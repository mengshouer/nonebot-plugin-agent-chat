"""Settings and profile editing, shared by the CLI flags and the session.

`nonebot-agent-chat --config-* / --profile-* / --reload` and the in-session
`:config` / `:profile` / `:reload` commands are the same actions with two
surfaces, so both live here: one implementation per action, one place that
knows how an outcome is worded. `cli.py` keeps the parser, the one-shot ask
path, and the process-level concerns (signals, exit).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from . import config_editor, env_file, profile_editor, setting_help, tui_state
from .config import Config
from .edits import EFFECT_HINT, EFFECT_HINT_CLI, EditChange, format_changes
from .env_file import read_values
from .errors import ConfigurationError, InputError
from .platforms import rule_label
from .profiles import ProfileRegistry
from .prompts import PromptRegistry
from .service import RELOAD_MARKER_FILE, AgentChatService, reload_marker
from .storage import AgentStore
from .tools import registry as tool_registry


def inspect_profiles(
    profile_dir: Path,
    selected: str | None,
    prompts: PromptRegistry,
    *,
    check_credentials: bool,
) -> int:
    registry = ProfileRegistry(profile_dir, selected, prompts)
    try:
        names = registry.load(required_profile=selected)
    except ConfigurationError as exc:
        print(f"profile error: {exc}", file=sys.stderr)
        return 2

    for name in names:
        loaded = registry.get(name)
        profile = loaded.config
        fallbacks = ",".join(profile.fallback_profiles) or "-"
        prompt = loaded.prompt.summary if loaded.prompt else "unknown"
        print(
            f"{name}: protocol={profile.protocol.value} model={profile.model} "
            f"search={profile.search_mode.value} fallback={fallbacks} "
            f"prompt={prompt}"
        )

    if not check_credentials:
        return 0

    errors: list[str] = []
    for loaded in registry.profiles.values():
        try:
            ProfileRegistry.resolve_api_key(loaded.config)
            if loaded.config.search_mode.value == "exa":
                # One resolver decides: an inline key counts, and the message is
                # spelled once (see ProfileRegistry.resolve_exa_api_key).
                ProfileRegistry.resolve_exa_api_key(loaded.config)
            tool_registry.for_profile(loaded.config)
        except ConfigurationError as exc:
            errors.append(f"{loaded.name}: {exc}")
    if errors:
        print("credential/tool errors:", file=sys.stderr)
        for error in errors:
            print(f"- {error}", file=sys.stderr)
        return 2
    print("profile check: ok")
    return 0


async def _apply_interactive_edit(
    service: AgentChatService,
    outcomes: list[_EditOutcome],
    *,
    bot_data_dir: Path,
    target: Path,
) -> None:
    """Report edit outcomes, request a bot reload, and reload this session."""

    changed = False
    for outcome in outcomes:
        for warning in outcome.warnings:
            print(f"⚠️ {warning}")
        for line in outcome.lines():
            print(line)
        for key in outcome.restart_keys:
            print(f"⚠️ {key} 需重启 bot 才生效")
        changed = changed or bool(outcome.changes)
    if not changed:
        return
    await _mark_reload(bot_data_dir)
    print(f"已写入 {target}；{EFFECT_HINT}")
    try:
        await service.reload_everything()
    except (ConfigurationError, InputError) as exc:
        print(f"本会话重载失败：{exc}", file=sys.stderr)
        return
    print("本会话已重载，后续提问直接使用新配置")


def _load_editor_module():
    """Import the full-screen editor on demand: textual is an optional extra."""

    try:
        from . import tui_app
    except ImportError as exc:  # pragma: no cover - optional extra
        raise InputError(
            '交互编辑器需要 textual：pip install "nonebot-plugin-agent-chat[tui]"'
        ) from exc
    return tui_app


async def _run_editor(
    *,
    env_file: Path | None,
    env_owned: set[str],
    config: Config,
    profile_dir: Path,
    bot_data_dir: Path,
    tab: str,
    focus_profile: str | None = None,
    service: AgentChatService | None = None,
) -> list[str]:
    """Open the full-screen editor; returns the actions it saved.

    The app writes the files and asks for the reload itself (marker plus, in a
    session, an immediate in-process reload), so callers only report.
    """

    editor = _load_editor_module()
    if tab == "settings" and env_file is None:
        raise InputError("交互编辑器需要 --env-file（--no-env 下没有默认路径）")
    settings = (
        tui_state.SettingsPanel(env_file, env_owned, config)
        if env_file is not None
        else None
    )

    async def request_reload() -> None:
        await _mark_reload(bot_data_dir)
        if service is not None:
            await service.reload_everything()

    context = tui_state.EditorContext(
        profiles=tui_state.ProfilesPanel(profile_dir),
        request_reload=request_reload,
        warnings_for_delete=lambda name: _collect_removal_warnings(
            name, profile_dir, bot_data_dir, config
        ),
        settings=settings,
    )
    await editor.run_editor(context, tab=tab, focus_profile=focus_profile)
    actions = format_changes(settings.last_changes) if settings is not None else []
    actions.extend(format_changes(context.profiles.last_changes))
    return actions


async def interactive_config(
    service: AgentChatService,
    command: str,
    *,
    env_file: Path | None,
    profile_dir: Path,
    bot_data_dir: Path,
) -> None:
    rest = command.partition(" ")[2].strip()
    action, _, argument = rest.partition(" ")
    argument = argument.strip()
    if action in ("", "list"):
        file_values = read_values(env_file) if env_file is not None else {}
        listing = config_editor.build_config_list(
            service.config, file_values=file_values, owned=service.env_owned
        )
        for line in _format_config_listing(listing, file_values):
            print(line)
        print()
        for line in setting_help.LEGEND:
            print(line)
        return
    if action == "explain" or action.startswith("AGENT_CHAT_"):
        # `:config <KEY>` is the shorthand for `:config explain <KEY>`.
        key = (argument if action == "explain" else action).strip().upper()
        if not key:
            raise InputError("用法：:config explain KEY")
        file_values = read_values(env_file) if env_file is not None else {}
        listing = config_editor.build_config_list(
            service.config, file_values=file_values, owned=service.env_owned
        )
        for line in _explain_lines(listing, key):
            print(line)
        return
    if action == "set":
        path = _require_env_file(env_file)
        key, raw = _parse_assignment(argument, usage=":config set")
        transaction, outcome = _plan_config_edits(
            path, [(key, raw)], service.config, service.env_owned
        )
        transaction.commit()
        await _apply_interactive_edit(
            service, [outcome], bot_data_dir=bot_data_dir, target=path
        )
        return
    if action == "unset":
        path = _require_env_file(env_file)
        if not argument:
            raise InputError("用法：:config unset KEY")
        transaction, outcome = _plan_config_edits(
            path, [(argument.strip(), None)], service.config, service.env_owned
        )
        transaction.commit()
        await _apply_interactive_edit(
            service, [outcome], bot_data_dir=bot_data_dir, target=path
        )
        return
    if action == "edit":
        path = _require_env_file(env_file)
        performed = await _run_editor(
            env_file=path,
            env_owned=service.env_owned,
            config=service.config,
            profile_dir=profile_dir,
            bot_data_dir=bot_data_dir,
            tab="settings",
            service=service,
        )
        for line in performed:
            print(line)
        if performed:
            print(EFFECT_HINT)
        return
    raise InputError("用法：:config [list | set KEY=VALUE | unset KEY | edit]")


async def interactive_profile(
    service: AgentChatService,
    command: str,
    *,
    env_file: Path | None,
    profile_dir: Path,
    bot_data_dir: Path,
) -> None:
    rest = command.partition(" ")[2].strip()
    action, _, argument = rest.partition(" ")
    argument = argument.strip()
    if action in ("", "list"):
        names = profile_editor.list_profiles(profile_dir)
        print("\n".join(names) or "没有 profile")
        return
    if action == "set":
        name, _, item = argument.partition(" ")
        if not name or not item:
            raise InputError("用法：:profile set NAME KEY=VALUE")
        transaction = profile_editor.ProfileFileTransaction(profile_dir)
        outcome = _plan_profile_set(transaction, name, item, usage=":profile set")
        transaction.commit()
        await _apply_interactive_edit(
            service,
            [outcome],
            bot_data_dir=bot_data_dir,
            target=profile_editor.profile_path(profile_dir, name),
        )
        return
    if action == "unset":
        name, _, field_name = argument.partition(" ")
        if not name or not field_name:
            raise InputError("用法：:profile unset NAME KEY")
        transaction = profile_editor.ProfileFileTransaction(profile_dir)
        outcome = _plan_profile_unset(transaction, name, field_name)
        transaction.commit()
        await _apply_interactive_edit(
            service,
            [outcome],
            bot_data_dir=bot_data_dir,
            target=profile_editor.profile_path(profile_dir, name),
        )
        return
    if action == "new":
        name, _, template = argument.partition(" ")
        template = template.removeprefix("--from").strip()
        if not name:
            raise InputError("用法：:profile new NAME [--from <模板>]")
        if not template:
            # Optional --from: the same first profile the editor prefills.
            template = profile_editor.preferred_template(
                profile_editor.list_profiles(profile_dir)
            )
            if not template:
                raise InputError("没有可复制的 profile：先创建或手写一个 profile 文件")
        transaction = profile_editor.ProfileFileTransaction(profile_dir)
        outcome = _plan_profile_new(transaction, name, template)
        transaction.commit()
        await _apply_interactive_edit(
            service,
            [outcome],
            bot_data_dir=bot_data_dir,
            target=profile_editor.profile_path(profile_dir, name),
        )
        return
    if action == "remove":
        if not argument:
            raise InputError("用法：:profile remove NAME")
        transaction = profile_editor.ProfileFileTransaction(profile_dir)
        outcome = await _plan_profile_remove(
            transaction, argument, profile_dir, bot_data_dir, service.config
        )
        transaction.commit()
        await _apply_interactive_edit(
            service,
            [outcome],
            bot_data_dir=bot_data_dir,
            target=profile_editor.profile_path(profile_dir, argument),
        )
        return
    if action == "edit":
        performed = await _run_editor(
            env_file=env_file,
            env_owned=service.env_owned,
            config=service.config,
            profile_dir=profile_dir,
            bot_data_dir=bot_data_dir,
            tab="profiles",
            focus_profile=argument or None,
            service=service,
        )
        for line in performed:
            print(line)
        if performed:
            print(EFFECT_HINT)
        return
    raise InputError(
        "用法：:profile [list | set NAME KEY=VALUE | unset NAME KEY | new NAME | "
        "remove NAME | edit [NAME]]"
    )


async def interactive_reload(service: AgentChatService, bot_data_dir: Path) -> None:
    await _mark_reload(bot_data_dir)
    report = await service.reload_now()
    applied = report["config"]["applied"]  # type: ignore[index]
    restart = report["config"]["restart_required"]  # type: ignore[index]
    print(f"已请求重载：profiles {len(report['profiles'])} 个")  # type: ignore[arg-type]
    for line in config_editor.format_report(applied, restart):
        print(line)
    if not applied and not restart:
        print("设置没有变化")


@dataclass
class _EditOutcome:
    """One editor action: what changed, plus what the caller must report."""

    changes: list[EditChange]
    warnings: list[str] = field(default_factory=list)
    restart_keys: list[str] = field(default_factory=list)
    config: Config | None = None

    def lines(self) -> list[str]:
        """How this outcome reads; callers never word a change themselves."""

        return format_changes(self.changes)


def _parse_assignment(item: str, *, usage: str) -> tuple[str, str]:
    key, separator, raw = item.partition("=")
    key = key.strip()
    if not separator or not key:
        raise InputError(f"{usage} 需要 KEY=VALUE：{item}")
    return key, raw


def _require_env_file(env_file: Path | None) -> Path:
    if env_file is None:
        raise InputError("编辑设置文件需要 --env-file（--no-env 下没有默认路径）")
    return env_file


def production_data_dir(args: argparse.Namespace, env_file: Path | None) -> Path:
    """The bot's store: environment first, then the dotenv file, then default."""

    file_values = read_values(env_file) if env_file is not None else {}
    return args.data_dir or Path(
        config_editor.environment_value(
            "AGENT_CHAT_DATA_DIR",
            file_values.get("AGENT_CHAT_DATA_DIR") or "data/agent_chat",
        )
    )


def _format_config_listing(
    listing: config_editor.ConfigList, file_values: Mapping[str, str]
) -> list[str]:
    lines = [config_editor.format_entry(entry) for entry in listing.entries]
    # Unmanaged keys (secrets) show their name only; nothing here edits them.
    lines.extend(f"{key}=•••（未管理）" for key in listing.unmanaged_keys)
    lines.extend(
        f"{key}={file_values.get(key, '')}（CLI 专用）" for key in listing.cli_only_keys
    )
    lines.extend(f"{key}=（未知键，将被忽略）" for key in listing.unknown_keys)
    return lines


def _explain_lines(listing: config_editor.ConfigList, key: str) -> list[str]:
    """One setting's explanation, with the value this process actually uses."""

    entry = next((item for item in listing.entries if item.key == key), None)
    return setting_help.explain_lines(
        key,
        current=entry.value if entry else None,
        source=entry.source if entry else None,
    )


def _config_listing_json(listing: config_editor.ConfigList) -> dict[str, object]:
    return {
        "entries": [
            {
                "key": entry.key,
                "value": entry.value,
                "source": entry.source,
                "restart_required": entry.restart_required,
            }
            for entry in listing.entries
        ],
        "cli_only_keys": listing.cli_only_keys,
        "unknown_keys": listing.unknown_keys,
        "unmanaged_keys": listing.unmanaged_keys,
    }


def _plan_config_edits(
    path: Path,
    edits: list[tuple[str, str | None]],
    config: Config,
    owned: set[str],
) -> tuple[config_editor.ConfigEditTransaction, _EditOutcome]:
    """Plan config-file edits in memory; the caller commits them once."""

    transaction = config_editor.ConfigEditTransaction(path, config, owned)
    changes: list[EditChange] = []
    restart: list[str] = []
    for key, raw in edits:
        canonical = key.upper()
        if raw is None:
            removed = transaction.unset(key)
            changes.append(EditChange("unset" if removed else "none", "setting", key))
            if removed and canonical in config_editor.RESTART_ONLY_KEYS:
                restart.append(canonical)
        else:
            transaction.set(key, raw)
            changes.append(
                EditChange("set", "setting", key, config_editor.display_token(key, raw))
            )
            if canonical in config_editor.RESTART_ONLY_KEYS:
                restart.append(canonical)
    return transaction, _EditOutcome(
        changes=changes,
        restart_keys=restart,
        config=transaction.config,
    )


def _commit_transactions(
    config_transaction: config_editor.ConfigEditTransaction | None,
    profile_transaction: profile_editor.ProfileFileTransaction | None,
) -> None:
    """Write every staged file at once; deletions happen after the writes."""

    files: dict[Path, str] = {}
    if profile_transaction is not None:
        files.update(profile_transaction.files)
    if config_transaction is not None:
        files[config_transaction.path] = config_transaction.text
    env_file.write_many_atomic(files)
    if profile_transaction is not None:
        for path in sorted(profile_transaction.deletions):
            path.unlink(missing_ok=True)


def _plan_profile_set(
    transaction: profile_editor.ProfileFileTransaction,
    name: str,
    item: str,
    *,
    usage: str,
) -> _EditOutcome:
    field, raw = _parse_assignment(item, usage=usage)
    transaction.set_field(name, field, raw)
    return _EditOutcome(
        changes=[
            # The value is masked here, so it never reaches a terminal unmasked.
            EditChange(
                "set",
                "profile_field",
                f"{name}.{field}",
                profile_editor.display_token(field, raw),
            )
        ]
    )


def _plan_profile_unset(
    transaction: profile_editor.ProfileFileTransaction,
    name: str,
    field: str,
) -> _EditOutcome:
    removed = transaction.unset_field(name, field.strip())
    return _EditOutcome(
        changes=[
            EditChange("unset", "profile_field", f"{name}.{field}")
            if removed
            else EditChange("none", "profile_field", name, field)
        ]
    )


def _plan_profile_new(
    transaction: profile_editor.ProfileFileTransaction,
    name: str,
    template: str,
) -> _EditOutcome:
    transaction.create(name, template)
    return _EditOutcome(changes=[EditChange("create", "profile", name, template)])


async def _plan_profile_remove(
    transaction: profile_editor.ProfileFileTransaction,
    name: str,
    profile_dir: Path,
    bot_data_dir: Path,
    config: Config,
) -> _EditOutcome:
    warnings = await _collect_removal_warnings(name, profile_dir, bot_data_dir, config)
    transaction.delete(name)
    return _EditOutcome(
        changes=[EditChange("remove", "profile", name)], warnings=warnings
    )


# Flags handled by this layer; the mutating subset is what the read-only flags
# refuse to be combined with. One list, so a new flag cannot drift between the
# routing check and the mutual-exclusion check.
_SETTINGS_FLAGS = (
    "config_list",
    "config_explain",
    "config_set",
    "config_unset",
    "config_edit",
    "profile_edit",
    "profile_set",
    "profile_unset",
    "profile_new",
    "profile_remove",
    "reload",
)
_READ_ONLY_SETTINGS_FLAGS = ("config_list", "config_explain")


# `--profile-edit` is `nargs="?"` with `const=""`: passing it *without* a name is
# meaningful, so for that flag presence alone counts.
_PRESENT_WHEN_EMPTY = frozenset({"profile_edit"})


def _flag_given(args: argparse.Namespace, name: str) -> bool:
    """Whether a flag was actually passed.

    Most flags are `append` (default `[]`) or `store_true`, so an empty value
    means "absent"; `profile_edit` is the exception above.
    """

    value = getattr(args, name)
    if value is None:
        return False
    if name in _PRESENT_WHEN_EMPTY:
        return True
    return bool(value)


def is_settings_action(args: argparse.Namespace) -> bool:
    return any(_flag_given(args, name) for name in _SETTINGS_FLAGS)


async def _mark_reload(data_dir: Path) -> None:
    """Write the reload marker into the bot's data directory (no bot needed)."""

    await asyncio.to_thread(
        env_file.write_atomic, data_dir / RELOAD_MARKER_FILE, reload_marker()
    )


async def _collect_removal_warnings(
    name: str,
    profile_dir: Path,
    data_dir: Path,
    config: Config,
) -> list[str]:
    """Reference warnings for a profile about to be deleted (no deletion)."""

    store = AgentStore(data_dir / "agent_chat.db")
    try:
        await store.initialize(interrupt_running=False)
        rules = await store.list_profile_rules()
        rooms = await store.rooms_using_profile(name)
    finally:
        await store.close()
    labels = [
        rule_label(str(rule["scope"]), str(rule["target_id"]))
        for rule in rules
        if rule.get("profile") == name
    ]
    return profile_editor.reference_warnings(
        name,
        rule_labels=labels,
        room_names=rooms,
        configured_default=config.agent_chat_default_profile,
    )


def _invalid_keys_notice(keys: Sequence[str]) -> str:
    """Tell the operator which broken values were ignored for this command."""

    return (
        "以下设置当前取值无效，本命令按默认值继续"
        "（来自 dotenv 文件时用 --config-unset 或编辑器修复；"
        "来自真实环境变量时请修改它的来源）：" + "、".join(keys)
    )


async def run_settings_command(
    args: argparse.Namespace,
    env_file: Path | None,
    env_owned: set[str],
    config: Config,
    profile_dir: Path,
    bot_data_dir: Path,
    *,
    invalid_keys: Sequence[str] = (),
) -> int:
    """--config-* / --profile-* / --reload without a bot: list, explain, edit."""

    file_values = read_values(env_file) if env_file is not None else {}
    mutating = any(
        _flag_given(args, name)
        for name in _SETTINGS_FLAGS
        if name not in _READ_ONLY_SETTINGS_FLAGS
    )
    if (args.config_list or args.config_explain) and (
        mutating or args.profile_list or args.check
    ):
        raise InputError(
            "--config-list / --config-explain 是只读的，"
            "不能与编辑动作、--reload 或 profile 检查同时使用"
        )
    actions: list[str] = []
    warnings: list[str] = []
    restart_required: list[str] = []
    changed = False

    if args.config_edit:
        editor_actions = await _run_editor(
            env_file=env_file,
            env_owned=env_owned,
            config=config,
            profile_dir=profile_dir,
            bot_data_dir=bot_data_dir,
            tab="settings",
        )
        actions.extend(editor_actions)
        if editor_actions:
            # The editor already wrote the marker; just report the outcome.
            actions.append(EFFECT_HINT_CLI)

    if args.profile_edit is not None:
        editor_actions = await _run_editor(
            env_file=env_file,
            env_owned=env_owned,
            config=config,
            profile_dir=profile_dir,
            bot_data_dir=bot_data_dir,
            tab="profiles",
            focus_profile=args.profile_edit or None,
        )
        actions.extend(editor_actions)
        if editor_actions:
            actions.append(EFFECT_HINT_CLI)

    if args.config_list or args.config_explain:
        listing = config_editor.build_config_list(
            config, file_values=file_values, owned=env_owned
        )
        if args.config_list:
            if args.json:
                payload = _config_listing_json(listing)
                payload["invalid_keys"] = list(invalid_keys)
                print(json.dumps(payload, ensure_ascii=False, indent=2))
                return 0
            print("\n".join(_format_config_listing(listing, file_values)))
            print()
            print("\n".join(setting_help.LEGEND))
            if invalid_keys:
                print()
                print("⚠️ " + _invalid_keys_notice(invalid_keys))
        for key_item in args.config_explain:
            if args.config_list:
                print()
            for line in _explain_lines(listing, key_item.strip().upper()):
                print(line)
        return 0

    def collect(outcome: _EditOutcome) -> None:
        nonlocal config, changed
        actions.extend(outcome.lines())
        warnings.extend(outcome.warnings)
        restart_required.extend(outcome.restart_keys)
        changed = changed or bool(outcome.changes)
        if outcome.config is not None:
            config = outcome.config

    config_transaction: config_editor.ConfigEditTransaction | None = None
    profile_transaction: profile_editor.ProfileFileTransaction | None = None

    if args.config_set or args.config_unset:
        path = _require_env_file(env_file)
        edits: list[tuple[str, str | None]] = [
            _parse_assignment(item, usage="--config-set") for item in args.config_set
        ]
        edits.extend((key.strip(), None) for key in args.config_unset)
        config_transaction, outcome = _plan_config_edits(path, edits, config, env_owned)
        collect(outcome)

    if (
        args.profile_set
        or args.profile_unset
        or args.profile_new
        or args.profile_remove
    ):
        profile_transaction = profile_editor.ProfileFileTransaction(profile_dir)
    if profile_transaction is not None:
        for name, item in args.profile_set:
            collect(
                _plan_profile_set(
                    profile_transaction, name, item, usage="--profile-set"
                )
            )
        for name, field_name in args.profile_unset:
            collect(_plan_profile_unset(profile_transaction, name, field_name))
        if args.profile_new:
            if not args.profile_from:
                raise InputError("--profile-new 需要 --from <模板 profile>")
            collect(
                _plan_profile_new(
                    profile_transaction, args.profile_new, args.profile_from
                )
            )
        if args.profile_remove:
            collect(
                await _plan_profile_remove(
                    profile_transaction,
                    args.profile_remove,
                    profile_dir,
                    bot_data_dir,
                    config,
                )
            )

    # Nothing has touched the disk yet: validate the whole batch, then write it.
    if config_transaction is not None or profile_transaction is not None:
        _commit_transactions(config_transaction, profile_transaction)

    if args.reload:
        changed = True
        actions.append("已请求重载")

    if restart_required:
        warnings.append(
            "以下设置需重启 bot 才生效：" + "、".join(sorted(set(restart_required)))
        )
    if invalid_keys:
        warnings.append(_invalid_keys_notice(invalid_keys))

    if changed:
        await _mark_reload(bot_data_dir)
        actions.append(EFFECT_HINT_CLI)

    if args.json:
        print(
            json.dumps(
                {
                    "actions": actions,
                    "warnings": warnings,
                    "restart_required": restart_required,
                    "invalid_keys": list(invalid_keys),
                    "reload_requested": changed,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        lines = [f"⚠️ {warning}" for warning in warnings] + actions
        print("\n".join(lines))
    return 0
