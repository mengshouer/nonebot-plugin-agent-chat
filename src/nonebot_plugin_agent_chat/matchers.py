from __future__ import annotations

import asyncio
import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from arclet.alconna import Arparma
from nonebot import (
    get_adapters,
    get_driver,
    get_plugin_config,
    logger,
    on_message,
    require,
)
from nonebot.adapters import Bot, Event
from nonebot.permission import SUPERUSER
from nonebot.rule import Rule

require("nonebot_plugin_alconna")

from nonebot_plugin_alconna import (
    Alconna,
    AlconnaMatcher,
    AlconnaMatches,
    Args,
    Image,
    MultiVar,
    UniMessage,
    on_alconna,
)

from . import answer_options, config_editor, env_file
from .alconna_ext import CommandMentionExtension
from .answer_delivery import send_answer_text
from .config import Config
from .delivery import loss_notice
from .errors import (
    BusyError,
    ConfigurationError,
    InputError,
    ProfileRuleError,
    ProviderError,
    RoomError,
)
from .formatting import append_sources
from .image_reply import (
    ImageReplySettings,
    deliver_answer,
)
from .input import (
    AGENT_ROOM_COMMAND,
    AGENTCTL_COMMAND,
    command_parts,
    has_trigger,
    is_other_bot_command,
    is_reserved_control_text,
    rule_target,
    strip_command_mention,
)
from .message_input import collect_message_input
from .platforms import (
    ChatIdentity,
    applicable_rule_rows,
    is_directed_at_bot,
    is_identity_allowed,
    resolve_chat_identity,
    uses_command_mentions,
)
from .render import close_shared_renderer, probe_renderer
from .service import AgentChatService

driver = get_driver()
agent_env_file = Path(os.getenv("AGENT_CHAT_ENV_FILE", ".env.agent_chat"))
# The file owns the keys it actually provided: reloads update only those, so a
# real environment variable keeps winning.
agent_env_owned = env_file.load_into_environ(agent_env_file)
plugin_config = get_plugin_config(Config)


def _resolve_secret(name: str) -> str | None:
    value = os.getenv(name)
    if value:
        return value
    configured = getattr(driver.config, name.lower(), None)
    reveal = getattr(configured, "get_secret_value", None)
    if reveal is not None:
        configured = reveal()
    return str(configured) if configured not in (None, "") else None


service = AgentChatService(
    plugin_config,
    secret_resolver=_resolve_secret,
    env_file=agent_env_file,
    env_owned=agent_env_owned,
)


def _sync_plugin_config() -> None:
    """Adopt a reloaded config into this module's snapshot."""

    global plugin_config
    plugin_config = service.config


async def _adopt_pending_reload() -> None:
    """Apply a CLI-requested reload; failures stay logged, not fatal."""

    await service.apply_pending_reload()
    _sync_plugin_config()


@driver.on_startup
async def initialize_agent_chat() -> None:
    installed = sorted(get_adapters())
    if installed:
        logger.info(
            "Agent chat installed adapters={}",
            ", ".join(installed),
        )
    await service.initialize()


@driver.on_shutdown
async def shutdown_agent_chat() -> None:
    await close_shared_renderer()
    await service.close()


def _bot_username(bot: Bot) -> str | None:
    username = getattr(bot, "username", None)
    return str(username) if username else None


async def _should_trigger(bot: Bot, event: Event) -> bool:
    # Apply a CLI-requested reload before routing: an added trigger or a
    # revoked ACL must govern this very message, which would otherwise be
    # filtered out (or accepted) under the stale snapshot.
    await _adopt_pending_reload()
    identity = resolve_chat_identity(bot, event)
    if identity is None:
        return False
    if not await SUPERUSER(bot, event) and not is_identity_allowed(
        identity,
        allowed_groups=plugin_config.agent_chat_allowed_groups,
        allowed_users=plugin_config.agent_chat_allowed_users,
    ):
        return False
    mentions = uses_command_mentions(identity.scope)
    username = _bot_username(bot)
    raw_text = event.get_plaintext()
    if mentions and is_other_bot_command(raw_text, username):
        return False
    text = strip_command_mention(raw_text, username) if mentions else raw_text
    if is_reserved_control_text(text, mentions=mentions):
        return False
    if has_trigger(text, plugin_config.agent_chat_triggers):
        return True
    if identity.private:
        return plugin_config.agent_chat_enable_private_auto_reply
    if not plugin_config.agent_chat_enable_at:
        return False
    return is_directed_at_bot(event)


llm = on_message(
    rule=Rule(_should_trigger),
    priority=plugin_config.agent_chat_priority,
    block=True,
)

# Control commands are parsed by Alconna and work on every supported adapter.
# The free-form ask matcher stays on_message: configurable substring triggers and
# image-bearing messages cannot be expressed faithfully as an Alconna command.
command_mention = CommandMentionExtension()


def _control_matcher(command: Alconna) -> type[AlconnaMatcher]:
    """Alconna matcher shared by the superuser control commands."""

    return on_alconna(
        command,
        permission=SUPERUSER,
        priority=5,
        block=True,
        auto_send_output=False,
        use_cmd_start=True,
        use_cmd_sep=False,
        extensions=[command_mention],
    )


_agentctl_command = Alconna(AGENTCTL_COMMAND, Args["parts", MultiVar(Any), ()])
_agent_room_command = Alconna(AGENT_ROOM_COMMAND, Args["parts", MultiVar(Any), ()])
agentctl = _control_matcher(_agentctl_command)
agent_room = _control_matcher(_agent_room_command)


def _describe_delivery_failure(failures: list[str]) -> str:
    """Summarize transport errors for logs without echoing message content."""

    if not failures:
        return "unknown"
    unique: list[str] = []
    for failure in failures:
        if failure not in unique:
            unique.append(failure)
    return "; ".join(unique[-3:])


async def _send_plain_text(bot: Bot, event: Event, text: str) -> None:
    await UniMessage.text(text).send(target=event, bot=bot)


async def _send_images(bot: Bot, event: Event, pages: Sequence[bytes]) -> None:
    await UniMessage([Image(raw=page) for page in pages]).send(
        target=event,
        bot=bot,
    )


async def _deliver_result(
    bot: Bot, event: Event, result, identity: ChatIdentity
) -> None:
    """Send one answer, as image pages when the profile and config allow it."""

    settings = ImageReplySettings.from_config(
        plugin_config,
        answer_options.image_reply_mode(
            plugin_config,
            service.profiles,
            profile_name=result.actual_profile,
            scope=identity.scope,
        ),
        scope=identity.scope,
    )
    failures: list[str] = []

    async def send_text(text: str) -> None:
        await send_answer_text(bot, event, text, identity.scope)

    async def send_images(pages: Sequence[bytes]) -> None:
        await _send_images(bot, event, pages)

    # Two independent switches: the text footer and the image footer. Inline
    # markdown links in the answer itself are part of the text and stay either
    # way; RunResult.sources keeps recording data regardless.
    show_text_sources = answer_options.show_sources(
        plugin_config,
        service.profiles,
        profile_name=result.actual_profile,
        scope=identity.scope,
        kind="text",
    )
    show_image_sources = answer_options.show_sources(
        plugin_config,
        service.profiles,
        profile_name=result.actual_profile,
        scope=identity.scope,
        kind="image",
    )
    report = await deliver_answer(
        text=result.text,
        # None (not just []) suppresses the whole rendered footer, including
        # the answer's own inline link targets.
        sources=(
            [source.url for source in result.sources] if show_image_sources else None
        ),
        settings=settings,
        # The text path appends the source footer only when its switch allows;
        # the rendered footer only exists on the image path.
        send_text=send_text,
        send_images=send_images,
        fallback_text=(
            append_sources(result.text, result.sources)
            if show_text_sources
            else result.text
        ),
        on_failure=failures.append,
    )
    _log_delivery(report, failures, settings.mode.value)
    if report.notice is not None:
        await _notify(bot, event, report.notice)
    if report.text_report is not None:
        await _report_delivery_loss(bot, event, report.text_report, failures)


def _log_delivery(report, failures: list[str], mode: str) -> None:
    if report.delivery == "text":
        if report.render_error is not None:
            logger.warning(
                "Agent chat image reply fell back to text: render_error={}",
                report.render_error,
            )
        return
    if report.delivery == "text_fallback":
        logger.warning(
            "Agent chat image delivery interrupted: sent_pages={} dropped_pages={} "
            "render_error={} errors={}",
            report.sent_pages,
            report.dropped_pages,
            report.render_error,
            _describe_delivery_failure(failures),
        )
        return
    if report.complete:
        logger.info(
            "Agent chat image delivery: mode={} messages={} pages={} attempts={}",
            mode,
            report.sent_messages,
            report.sent_pages,
            report.send_attempts,
        )
        return
    logger.warning(
        "Agent chat image delivery incomplete: sent_pages={} dropped_pages={} "
        "attempts={} errors={}",
        report.sent_pages,
        report.dropped_pages,
        report.send_attempts,
        _describe_delivery_failure(failures),
    )


async def _report_delivery_loss(
    bot: Bot,
    event: Event,
    report,
    failures: list[str],
) -> None:
    """Report lost text instead of letting a transport failure look like an error."""

    if report.complete:
        return
    logger.warning(
        "Agent chat delivery incomplete: sent={} dropped={} dropped_chars={} "
        "attempts={} errors={}",
        report.sent_chunks,
        report.dropped_chunks,
        report.dropped_chars,
        report.send_attempts,
        _describe_delivery_failure(failures),
    )
    notice = loss_notice(report)
    if notice is not None:
        await _notify(bot, event, notice)


async def _notify(bot: Bot, event: Event, text: str) -> None:
    try:
        await _send_plain_text(bot, event, text)
    except asyncio.CancelledError:
        raise
    except Exception as exc:  # noqa: BLE001 - transport boundary
        logger.warning(
            "Agent chat could not report the delivery failure: {}",
            type(exc).__name__,
        )


async def _report_failure(bot: Bot, event: Event, exc: BaseException) -> None:
    if isinstance(exc, ProfileRuleError):
        # Operator action is required, so the guidance goes to the chat too.
        logger.error("Agent chat profile rule error: {}", exc)
        await _send_plain_text(bot, event, str(exc))
    elif isinstance(exc, (BusyError, InputError, RoomError)):
        await _send_plain_text(bot, event, str(exc))
    elif isinstance(exc, ConfigurationError):
        logger.error("Agent chat configuration error: {}", exc)
        if await SUPERUSER(bot, event):
            await _send_plain_text(bot, event, f"LLM 配置错误：{exc}")
        else:
            await _send_plain_text(bot, event, "LLM 服务暂不可用，请联系管理员")
    elif isinstance(exc, ProviderError):
        logger.warning(
            "Agent chat provider error [{}]: {}",
            exc.error_type,
            exc,
        )
        if await SUPERUSER(bot, event):
            await _send_plain_text(
                bot,
                event,
                f"模型服务请求失败（{exc.error_type}）：{exc}",
            )
        else:
            await _send_plain_text(bot, event, "模型服务请求失败，请稍后再试")
    elif isinstance(exc, asyncio.CancelledError):
        await _send_plain_text(bot, event, "请求已取消")
    else:
        # The project logger enables diagnose=True; avoid traceback locals because
        # they may contain prompts or credentials.
        logger.error("Unexpected agent chat error: {}", type(exc).__name__)
        await _send_plain_text(bot, event, "处理请求时发生错误")


@llm.handle()
async def handle_llm(bot: Bot, event: Event) -> None:
    identity = resolve_chat_identity(bot, event)
    if identity is None:
        return
    try:
        message = await UniMessage.of(event.get_message(), bot=bot).attach_reply(
            event=event,
            bot=bot,
        )
        # A dedicated task, not a bare await: the run guard records the current
        # task so /agentctl cancel cancels the run while this handler survives to
        # report the outcome.
        # Mention stripping only happens where the platform declares that form;
        # other platforms keep the raw text.
        username = _bot_username(bot) if uses_command_mentions(identity.scope) else None
        task = asyncio.create_task(
            service.ask_deferred(
                lambda: collect_message_input(
                    message,
                    bot=bot,
                    event=event,
                    remove_triggers=plugin_config.agent_chat_triggers,
                    bot_username=username,
                    max_reply_chars=plugin_config.agent_chat_max_reply_chars,
                    max_images=plugin_config.agent_chat_max_images,
                    max_image_bytes=plugin_config.agent_chat_max_image_bytes,
                ),
                subject_key=identity.subject_key,
                context_key=identity.context_key,
                conversation=identity.conversation,
            )
        )
        result = await task
        await _deliver_result(bot, event, result, identity)
    except asyncio.CancelledError as exc:
        await _report_failure(bot, event, exc)
    except Exception as exc:  # noqa: BLE001 - matcher error boundary
        await _report_failure(bot, event, exc)


async def _execute_agentctl(
    bot: Bot,
    event: Event,
    command: str,
    value: str,
    identity: ChatIdentity,
) -> None:
    await _adopt_pending_reload()
    if command == "status":
        if identity.private:
            details = await service.status_details()
            details["image_reply"] = answer_options.image_reply_status(
                plugin_config, probe_renderer()
            )
            details["platforms"] = answer_options.platform_status(
                sorted(get_adapters())
            )
        else:
            # No cross-conversation diagnostics (recent_runs) for a group.
            details = await service.conversation_status(identity.conversation)
        await _send_plain_text(
            bot,
            event,
            json.dumps(details, ensure_ascii=False, indent=2),
        )
    elif command == "profiles":
        await _send_profile_list(bot, event, identity)
    elif command == "use":
        if not value:
            raise InputError("用法：/agentctl use <profile>")
        await service.use_profile(value)
        await _send_plain_text(bot, event, f"已切换到 profile: {value}")
    elif command == "reload":
        try:
            report = await service.reload_now()
        except (ConfigurationError, InputError) as exc:
            # The failure detail names files and keys: private chat only.
            logger.error("Agent chat reload failed: {}", exc)
            if identity.private:
                await _send_plain_text(bot, event, f"重载失败：{exc}")
            else:
                await _send_plain_text(
                    bot, event, "重载失败：配置或 profile 有误，详情见日志或私聊"
                )
            return
        _sync_plugin_config()
        report_profiles = report.get("profiles")
        names: list[str] = (
            list(report_profiles) if isinstance(report_profiles, list) else []
        )
        report_config = report.get("config")
        config = report_config if isinstance(report_config, dict) else {}
        applied = config.get("applied")
        restart = config.get("restart_required")
        if not isinstance(applied, dict):
            applied = {}
        if not isinstance(restart, dict):
            restart = {}
        if identity.private:
            lines = [f"已重载 {len(names)} 个 profile"]
            lines.extend(config_editor.format_report(applied, restart))
            if restart:
                lines.append("标 ⚠️ 的设置需重启 bot 才能生效")
            elif applied:
                lines.append("设置已生效")
            await _send_plain_text(bot, event, "\n".join(lines))
        else:
            # Operator detail cross-conversation: groups get the count only.
            summary = f"已重载：profiles {len(names)} 个"
            if applied:
                summary += f"；{len(applied)} 项设置生效"
            if restart:
                summary += f"（{len(restart)} 项需重启）"
            await _send_plain_text(bot, event, summary)
    elif command == "cancel":
        if not value:
            raise InputError("用法：/agentctl cancel <run_id|all>")
        count = await service.cancel_runs(value)
        await _send_plain_text(bot, event, f"已发送取消请求：{count} 个任务")
    elif command == "rule":
        await _execute_rule_command(bot, event, value, identity)
    else:
        raise InputError("可用命令：status、profiles、use、reload、cancel、rule")


PROFILE_SOURCE_LABELS = {
    "room": "Room 绑定",
    "use": "临时切换",
    "group-rule": "群规则",
    "platform-rule": "平台规则",
    "default": "全局默认",
    "unavailable": "不可用",
}


def _source_label(source: str) -> str:
    """Chinese label for a profile source in human-readable chat lines."""

    return PROFILE_SOURCE_LABELS.get(source, source)


async def _send_profile_list(bot: Bot, event: Event, identity: ChatIdentity) -> None:
    """Profile names are operator detail; groups only learn their own."""

    if not identity.private:
        profile, source = await service.effective_profile(identity.conversation)
        await _send_plain_text(
            bot, event, f"本会话 profile: {profile}（来源：{_source_label(source)}）"
        )
        return
    active = service.active_profile_name
    lines = [
        ("* " if name == active else "  ") + name for name in service.profiles.names()
    ]
    await _send_plain_text(bot, event, "\n".join(lines) or "没有可用 profile")


async def _execute_rule_command(
    bot: Bot,
    event: Event,
    value: str,
    identity: ChatIdentity,
) -> None:
    parts = value.split()
    if not parts:
        raise InputError(
            "用法：/agentctl rule set <profile> [platform|group <id>] | "
            "rule list | rule unset [platform|group <id>]"
        )
    action, *rest = parts
    if action == "list":
        await _send_rule_list(bot, event, identity)
        return
    if action in ("set", "unset"):
        if action == "set":
            if not rest:
                raise InputError("用法：/agentctl rule set <profile> [platform]")
            profile, target_tokens = rest[0], rest[1:]
        else:
            profile, target_tokens = "", rest
        scope, target_id = rule_target(target_tokens, identity)
        if action == "set":
            row, warnings = await service.set_profile_rule(
                scope=scope,
                target_id=target_id,
                profile=profile,
                updated_by=identity.operator_id,
                reveal_names=identity.private,
            )
            lines = service.format_rule_action(
                "set",
                scope,
                target_id,
                profile=str(row["profile"]) if row else profile,
                warnings=warnings,
            )
            await _send_plain_text(bot, event, "\n".join(lines))
            return
        removed = await service.unset_profile_rule(scope=scope, target_id=target_id)
        await _send_plain_text(
            bot,
            event,
            "\n".join(
                service.format_rule_action("unset", scope, target_id, removed=removed)
            ),
        )
        return
    raise InputError(f"未知 rule 子命令：{action}（可用 set、list、unset）")


async def _send_rule_list(bot: Bot, event: Event, identity: ChatIdentity) -> None:
    rules = await service.rule_status()
    if not identity.private:
        profile, source = await service.effective_profile(identity.conversation)
        lines = [f"本会话生效 profile：{profile}（来源：{_source_label(source)}）"]
        lines.extend(
            f"适用规则：{rule['label']} → {rule['profile']}"
            for rule in applicable_rule_rows(rules, identity.conversation)
        )
        await _send_plain_text(bot, event, "\n".join(lines))
        return
    if not rules:
        await _send_plain_text(bot, event, "没有规则")
        return
    lines = [service.format_rule_line(rule, with_audit=True) for rule in rules]
    lines.append(f"共 {len(rules)} 条")
    if plugin_config.agent_chat_room_enabled:
        lines.append("注意：Room 绑定优先于规则")
    await _send_plain_text(bot, event, "\n".join(lines))


async def _room_rule_warning(identity: ChatIdentity, profile_name: str) -> str | None:
    """Tell the operator when a Room binding shadows a profile rule."""

    rule = await service.applicable_rule(identity.conversation)
    if rule is None or str(rule["profile"]) == profile_name:
        return None
    return (
        f"注意：本会话规则 {rule['label']} → {rule['profile']}，"
        f"当前 Room 绑定 {profile_name} 会覆盖它"
    )


@agentctl.handle()
async def handle_agentctl(
    bot: Bot,
    event: Event,
    result: Arparma = AlconnaMatches(),  # noqa: B008 - NoneBot dependency injection
) -> None:
    command, value = command_parts(result.all_matched_args.get("parts") or ())
    command = command or "status"
    identity = resolve_chat_identity(bot, event)
    if identity is None:
        await _send_plain_text(bot, event, "当前会话类型不支持该命令")
        return
    try:
        await _execute_agentctl(bot, event, command, value, identity)
    except asyncio.CancelledError as exc:
        await _report_failure(bot, event, exc)
    except Exception as exc:  # noqa: BLE001 - command error boundary
        await _report_failure(bot, event, exc)


async def _execute_agent_room(
    bot: Bot,
    event: Event,
    message: UniMessage,
    command: str,
    value: str,
    identity: ChatIdentity,
) -> None:
    await _adopt_pending_reload()
    context = identity.context_key
    if command == "new":
        room = await service.create_room(context, value or "room")
        warning = await _room_rule_warning(identity, room.profile)
        lines = [f"Agent Room 已创建：{room.name}"]
        if warning:
            lines.append(warning)
        await _send_plain_text(bot, event, "\n".join(lines))
    elif command == "status":
        room = await service.room_status(context)
        if room is None:
            await _send_plain_text(bot, event, "当前没有活跃的 Agent Room")
        else:
            await _send_plain_text(
                bot,
                event,
                f"Room: {room.name}\nProfile: {room.profile}\n"
                f"Updated: {room.updated_at}",
            )
    elif command == "use":
        if not value:
            raise InputError("用法：/agent_room use <profile>")
        room = await service.set_room_profile(context, value)
        warning = await _room_rule_warning(identity, room.profile)
        lines = [f"Agent Room 已切换到 profile: {room.profile}"]
        if warning:
            lines.append(warning)
        await _send_plain_text(bot, event, "\n".join(lines))
    elif command == "clear":
        await service.clear_room(context)
        await _send_plain_text(bot, event, "Agent Room 历史已清空")
    elif command == "close":
        await service.close_room(context)
        await _send_plain_text(bot, event, "Agent Room 已关闭")
    elif command == "ask":
        task = asyncio.create_task(
            service.ask_room_deferred(
                lambda: collect_message_input(
                    message,
                    bot=bot,
                    event=event,
                    current_text_override=value,
                    max_reply_chars=plugin_config.agent_chat_max_reply_chars,
                    max_images=plugin_config.agent_chat_max_images,
                    max_image_bytes=plugin_config.agent_chat_max_image_bytes,
                ),
                subject_key=identity.subject_key,
                context_key=identity.context_key,
            )
        )
        result = await task
        await _deliver_result(bot, event, result, identity)
    else:
        raise InputError("可用命令：new、ask、use、status、clear、close")


@agent_room.handle()
async def handle_agent_room(
    bot: Bot,
    event: Event,
    result: Arparma = AlconnaMatches(),  # noqa: B008 - NoneBot dependency injection
) -> None:
    command, value = command_parts(result.all_matched_args.get("parts") or ())
    command = command or "status"
    identity = resolve_chat_identity(bot, event)
    if identity is None:
        await _send_plain_text(bot, event, "当前会话类型不支持该命令")
        return
    try:
        message = await UniMessage.of(event.get_message(), bot=bot).attach_reply(
            event=event,
            bot=bot,
        )
        await _execute_agent_room(bot, event, message, command, value, identity)
    except asyncio.CancelledError as exc:
        await _report_failure(bot, event, exc)
    except Exception as exc:  # noqa: BLE001 - command error boundary
        await _report_failure(bot, event, exc)
