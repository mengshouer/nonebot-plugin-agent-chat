from __future__ import annotations

import argparse
import asyncio
import json
import os
import signal
import sys
import threading
import time
from pathlib import Path

from . import config_editor
from .config import Config
from .env_file import load_into_environ
from .errors import AgentChatError, ConfigurationError, InputError
from .formatting import append_sources
from .images import ImageBudget, load_local_image
from .input import CollectedInput
from .models import AgentImage, ReasoningEffort, RunResult
from .platforms import ConversationRef, SupportScope, scope_value
from .prompts import PromptRegistry
from .service import AgentChatService

# The settings/profile layer lives in its own module; these are its entry points.
from .settings_cli import (
    inspect_profiles,
    interactive_config,
    interactive_profile,
    interactive_reload,
    is_settings_action,
    production_data_dir,
    run_settings_command,
)

_DEBUG_SUBJECT = "local-debug:user"
_DEBUG_CONTEXT = "local-debug:terminal"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m nonebot_plugin_agent_chat",
        description="在本地运行 Agent Chat，无需启动 NoneBot。",
        add_help=False,
    )
    parser._positionals.title = "位置参数"
    parser._optionals.title = "选项"
    parser.add_argument("-h", "--help", action="help", help="显示帮助并退出")
    parser.add_argument("prompt", nargs="*", help="单次提问的内容")
    parser.add_argument(
        "--platform-list",
        action="store_true",
        help="列出可用于配置的平台标识并退出，无需加载 Bot 配置",
    )
    parser.add_argument("--profile", help="Profile ID（文件名，不含扩展名）")
    parser.add_argument("--profiles-dir", type=Path, help="Profile 文件目录")
    parser.add_argument("--data-dir", type=Path, help="独立的本地调试数据目录")
    parser.add_argument(
        "--env-file",
        type=Path,
        help="要加载的配置文件（默认：.env.agent_chat）",
    )
    parser.add_argument(
        "--no-env",
        action="store_true",
        help="不加载配置文件",
    )
    parser.add_argument(
        "--image",
        action="append",
        default=[],
        type=Path,
        help="附加一张本地图片，可重复使用，最多四张",
    )
    parser.add_argument(
        "--room",
        action="store_true",
        help="使用新建的持久调试会话 Agent Room，不切换备用 Profile",
    )
    parser.add_argument(
        "--room-name", default="local-debug", help="调试会话名称（默认：local-debug）"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="校验 Profile、工具及其引用的凭据",
    )
    parser.add_argument(
        "--reasoning-effort",
        choices=[effort.value for effort in ReasoningEffort],
        help="覆盖所选 Profile 的推理强度，仅对当前进程生效",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="将运行元数据输出到标准错误流",
    )
    parser.add_argument(
        "--render-image",
        type=Path,
        metavar="OUT.png",
        help=(
            "将提问内容或 --render-input 文件渲染为 PNG，不发送模型请求；"
            "多页文件依次添加 -2、-3 等后缀"
        ),
    )
    parser.add_argument(
        "--render-input",
        type=Path,
        help="从此文件读取 --render-image 所需的 Markdown 内容",
    )
    parser.add_argument(
        "--rule-list",
        action="store_true",
        help="列出平台和群的默认 Profile 规则并退出",
    )
    parser.add_argument(
        "--rule-set",
        metavar="PROFILE",
        help="将 PROFILE 设为 --scope [--group] 的默认 Profile 并退出",
    )
    parser.add_argument(
        "--rule-unset",
        action="store_true",
        help="删除 --scope [--group] 对应的规则并退出",
    )
    parser.add_argument(
        "--scope",
        type=scope_value,
        metavar="PLATFORM",
        help="平台标识，区分大小写（如 QQClient、Telegram）；用 --platform-list 查询",
    )
    parser.add_argument(
        "--group",
        metavar="ID",
        help="--scope 平台内的群 ID；省略时选择整个平台",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="将 --scope 指定的会话视为私聊，私聊不应用 Profile 规则",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help=(
            "为 --platform-list / --rule-* / --config-* / --profile-* 输出 JSON；"
            "编辑操作输出执行结果，而不是列表"
        ),
    )
    parser.add_argument(
        "--config-list",
        action="store_true",
        help="列出 AGENT_CHAT_* 配置及其取值来源并退出",
    )
    parser.add_argument(
        "--config-set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="在配置文件中设置一个 AGENT_CHAT_* 配置项，可重复使用",
    )
    parser.add_argument(
        "--config-explain",
        action="append",
        default=[],
        metavar="KEY",
        help="查看配置项的说明、类型、默认值和重载方式",
    )
    parser.add_argument(
        "--config-unset",
        action="append",
        default=[],
        metavar="KEY",
        help="删除一个配置项以恢复默认值，可重复使用",
    )
    parser.add_argument(
        "--profile-list",
        action="store_true",
        help="列出 Profile 并退出",
    )
    parser.add_argument(
        "--profile-set",
        action="append",
        nargs=2,
        default=[],
        metavar=("NAME", "KEY=VALUE"),
        help="设置一个 Profile 字段，可重复使用",
    )
    parser.add_argument(
        "--profile-unset",
        action="append",
        nargs=2,
        default=[],
        metavar=("NAME", "KEY"),
        help="删除一个 Profile 字段，可重复使用",
    )
    parser.add_argument(
        "--profile-new",
        metavar="NAME",
        help="复制 --from 指定的 Profile，创建新的 Profile",
    )
    parser.add_argument(
        "--from",
        dest="profile_from",
        metavar="TEMPLATE",
        help="--profile-new 使用的模板 Profile",
    )
    parser.add_argument(
        "--profile-remove",
        metavar="NAME",
        help="删除指定 Profile 文件，输入名称即表示确认",
    )
    parser.add_argument(
        "--config-edit",
        action="store_true",
        help="打开交互式配置编辑器，需安装 tui 扩展依赖",
    )
    parser.add_argument(
        "--profile-edit",
        nargs="?",
        const="",
        metavar="NAME",
        help="打开交互式 Profile 编辑器，需安装 tui 扩展依赖",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="请求正在运行的 Bot 重载配置和 Profile",
    )
    parser.add_argument(
        "--render-sources",
        action="append",
        default=[],
        metavar="URL",
        help="在图片来源页脚中添加一个 URL，可重复使用",
    )
    return parser


def _render_source_text(args: argparse.Namespace) -> str:
    if args.render_input is not None:
        return args.render_input.read_text(encoding="utf-8")
    return " ".join(args.prompt)


async def _render_image_command(args: argparse.Namespace) -> int:
    """Local-only render so typography can be checked without sending anything."""

    from .render import close_shared_renderer, probe_renderer, shared_renderer

    available, detail = probe_renderer()
    if not available:
        print(f"renderer unavailable: {detail}", file=sys.stderr)
        return 2
    text = _render_source_text(args)
    if not text.strip():
        print("nothing to render: provide a prompt or --render-input", file=sys.stderr)
        return 2
    out = args.render_image
    assert out is not None
    max_height = config_editor.environment_config().agent_chat_image_reply_max_height
    renderer = shared_renderer(20.0)
    try:
        document = await renderer.render(text, sources=args.render_sources)
        top = 0
        page_number = 0
        while True:
            height = document.slice_height(top, max_height)
            if height <= 0:
                break
            page_number += 1
            page = await document.slice(top, height)
            target = (
                out
                if page_number == 1
                else out.with_name(f"{out.stem}-{page_number}{out.suffix}")
            )
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(page.data)
            print(f"{target}: {page.width}x{page.height} {len(page.data)} bytes")
            top += height
        await document.close()
    except AgentChatError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    finally:
        await close_shared_renderer()
    return 0


def _paths(args: argparse.Namespace, config: Config) -> tuple[Path, Path, str | None]:
    """Resolve profile/debug paths; the Config already read the environment."""

    profiles = args.profiles_dir or config.agent_chat_profile_dir
    # AGENT_CHAT_DEBUG_DATA_DIR is CLI-only: it has no Config field and is never
    # read by the bot itself.
    data = args.data_dir or Path(
        config_editor.environment_value(
            "AGENT_CHAT_DEBUG_DATA_DIR", "data/agent_chat/debug"
        )
    )
    profile = args.profile or config.agent_chat_default_profile
    return profiles, data, profile


async def _load_images(paths: list[Path], config: Config) -> list[AgentImage]:
    budget = ImageBudget(
        max_images=config.agent_chat_max_images,
        max_image_bytes=config.agent_chat_max_image_bytes,
    )
    budget.check_count(len(paths))
    images = []
    total = 0
    for path in paths:
        remaining = budget.remaining_bytes(total)
        if remaining <= 0:
            budget.check_room_for(total)
        image = await load_local_image(path, remaining)
        total += len(image.data)
        images.append(image)
    return images


def _debug_metadata(result: RunResult, elapsed: float) -> str:
    return json.dumps(
        {
            "run_id": result.run_id,
            "profile": result.actual_profile,
            "elapsed_seconds": round(elapsed, 3),
            "usage": {
                "input_tokens": result.usage.input_tokens,
                "output_tokens": result.usage.output_tokens,
                "total_tokens": result.usage.total_tokens,
                "reasoning_tokens": result.usage.reasoning_tokens,
            },
            "model_turns": result.model_turns,
            "local_tool_calls": result.local_tool_calls,
            "searches": result.searches,
        },
        ensure_ascii=False,
        indent=2,
    )


async def _rule_command(service: AgentChatService, args: argparse.Namespace) -> int:
    """Handle --rule-list / --rule-set / --rule-unset."""

    target_id = args.group or ""
    if (args.rule_set is not None or args.rule_unset) and not args.scope:
        raise InputError("--rule-set/--rule-unset 需要 --scope")
    if args.rule_set is not None:
        row, warnings = await service.set_profile_rule(
            scope=args.scope,
            target_id=target_id,
            profile=args.rule_set,
            updated_by="cli",
        )
        payload: dict[str, object] = {"rule": row, "warnings": warnings}
        human = service.format_rule_action(
            "set",
            args.scope,
            target_id,
            profile=str(row["profile"]) if row else args.rule_set,
            warnings=warnings,
        )
    elif args.rule_unset:
        removed = await service.unset_profile_rule(
            scope=args.scope, target_id=target_id
        )
        payload = {
            "scope": scope_value(args.scope),
            "target_id": target_id,
            "removed": removed,
        }
        human = service.format_rule_action(
            "unset", args.scope, target_id, removed=removed
        )
    else:
        rules = await service.rule_status()
        payload = {"rules": rules}
        human = [service.format_rule_line(rule, with_audit=True) for rule in rules]
        human.append(f"共 {len(rules)} 条")
        if service.config.agent_chat_room_enabled:
            human.append("注意：Room 绑定优先于规则")
    print(
        json.dumps(payload, ensure_ascii=False, indent=2)
        if args.json
        else "\n".join(human)
    )
    return 0


async def _ask(
    service: AgentChatService,
    prompt: str,
    images: list[AgentImage],
    *,
    room: bool,
    debug: bool,
    conversation: ConversationRef | None = None,
) -> None:
    text = prompt.strip()
    if not text and images:
        text = "Describe the provided image(s)."
    if not text and not images:
        raise InputError("请输入问题")

    collected = CollectedInput(text=text, images=list(images))
    started = time.monotonic()
    if room:
        result = await service.ask_room(
            collected,
            subject_key=_DEBUG_SUBJECT,
            context_key=_DEBUG_CONTEXT,
        )
    else:
        result = await service.ask(
            collected,
            subject_key=_DEBUG_SUBJECT,
            context_key=_DEBUG_CONTEXT,
            conversation=conversation,
        )
    elapsed = time.monotonic() - started
    print(append_sources(result.text, result.sources))
    if debug:
        print(_debug_metadata(result, elapsed), file=sys.stderr)


_INTERACTIVE_HELP = """可用命令：
  :config [list]                 列出设置（值 / 来源 / 需重启标记）
  :config <KEY>                  查看单条说明（等价 :config explain <KEY>）
  :config explain KEY            查看单条说明（说明 / 类型 / 默认 / 生效）
  :config set KEY=VALUE          修改设置，写入文件并请求重载
  :config unset KEY              删除设置行，恢复默认值
  :config edit                   打开交互式设置编辑器（需 tui 扩展依赖）
  :profile [list]                列出 Profile
  :profile set NAME KEY=VALUE    修改 Profile 字段
  :profile unset NAME KEY        删除 Profile 字段，恢复默认值
  :profile new NAME [--from T]   复制现有 Profile 新建
  :profile remove NAME           删除 Profile（先显示规则 / Room 引用警告）
  :profile edit [NAME]           打开交互式 Profile 编辑器（需 tui 扩展依赖）
  :reload                        重读配置文件与 Profile，并请求 Bot 重载
  :status                        查看当前状态
  :new [name]                    新建 Agent Room（需 --room）
  :clear                         清空 Agent Room 历史（需 --room）
  :help                          显示本帮助
  :quit / :q / :exit             退出
其他输入按问题发送。"""


# How long a Ctrl+C'd CLI may spend on cleanup before it is forced out.
_INTERRUPT_GRACE_SECONDS = 2.0


def _hard_exit(status: int) -> None:
    """Leave now, without atexit hooks: thread joins can block indefinitely."""

    try:
        sys.stdout.flush()
        sys.stderr.flush()
    finally:
        os._exit(status)


def _install_interrupt_escape(grace_seconds: float = _INTERRUPT_GRACE_SECONDS) -> None:
    """Guarantee the process exits after Ctrl+C.

    Normal cleanup usually finishes in milliseconds, in which case the timer
    never fires; if it is stuck (a worker thread the interpreter refuses to
    abandon), a second Ctrl+C or the timer ends the process anyway.
    """

    def force_quit(number: int, frame: object) -> None:
        print("\nforced exit", file=sys.stderr)
        _hard_exit(130)

    try:
        signal.signal(signal.SIGINT, force_quit)
    except ValueError:  # pragma: no cover - not the main thread
        pass
    timer = threading.Timer(grace_seconds, _hard_exit, args=(130,))
    timer.daemon = True
    timer.start()


def _release_stdin() -> None:
    """Unblock a pending ``input()`` so its worker thread can end.

    ``sys.stdin`` is usually opened read-only, so the newline is written to the
    controlling terminal instead; a pty's input queue is shared, which is what
    wakes the blocked ``read()``.
    """

    try:
        if sys.stdin is None or not sys.stdin.isatty():
            return
        with open("/dev/tty", "w", encoding="utf-8") as tty:
            tty.write("\n")
            tty.flush()
    except (OSError, ValueError):  # pragma: no cover - no controlling terminal
        pass


async def _interactive_command(
    service: AgentChatService,
    command: str,
    *,
    env_file: Path | None,
    profile_dir: Path,
    bot_data_dir: Path,
    room: bool,
    room_name: str,
) -> bool:
    """Handle one ``:`` command; True means the session should end."""

    head = command.split(maxsplit=1)[0]
    if head in {":quit", ":q", ":exit"}:
        return True
    if head in {":help", ":?"}:
        print(_INTERACTIVE_HELP)
        return False
    if head == ":status":
        print(json.dumps(await service.status_details(), ensure_ascii=False, indent=2))
        return False
    if head == ":new":
        if not room:
            print(":new requires --room", file=sys.stderr)
            return False
        name = command.partition(" ")[2].strip() or room_name
        await service.create_room(_DEBUG_CONTEXT, name)
        print(f"new room: {name}")
        return False
    if head == ":clear":
        if not room:
            print(":clear requires --room", file=sys.stderr)
            return False
        await service.clear_room(_DEBUG_CONTEXT)
        print("room cleared")
        return False
    if head == ":config":
        await interactive_config(
            service,
            command,
            env_file=env_file,
            profile_dir=profile_dir,
            bot_data_dir=bot_data_dir,
        )
        return False
    if head == ":profile":
        await interactive_profile(
            service,
            command,
            env_file=env_file,
            profile_dir=profile_dir,
            bot_data_dir=bot_data_dir,
        )
        return False
    if head == ":reload":
        await interactive_reload(service, bot_data_dir)
        return False
    print(f"未知命令：{head}（:help 查看可用命令）", file=sys.stderr)
    return False


async def _interactive(
    service: AgentChatService,
    initial_images: list[AgentImage],
    *,
    room: bool,
    room_name: str,
    debug: bool,
    conversation: ConversationRef | None = None,
    env_file: Path | None = None,
    profile_dir: Path,
    bot_data_dir: Path,
) -> None:
    # The session edits this file: let the in-process reload follow it too.
    if env_file is not None and service.env_file is None:
        service.env_file = env_file
    print(
        "Agent Chat 本地调试。输入 :help 查看命令"
        "（:config / :profile / :reload 可直接改配置）"
    )
    images = initial_images
    while True:
        try:
            prompt = await asyncio.to_thread(input, "you> ")
        except EOFError:
            print()
            return
        except KeyboardInterrupt:
            # Ctrl+C: `input()` runs in a worker thread that is still blocked on
            # stdin, so wake it before leaving -- otherwise the interpreter
            # hangs joining it at shutdown.
            _release_stdin()
            print()
            return
        command = prompt.strip()
        if command.startswith(":"):
            try:
                if await _interactive_command(
                    service,
                    command,
                    env_file=env_file,
                    profile_dir=profile_dir,
                    bot_data_dir=bot_data_dir,
                    room=room,
                    room_name=room_name,
                ):
                    return
            except AgentChatError as exc:
                print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
            continue
        if not command and not images:
            continue
        try:
            await _ask(
                service,
                prompt,
                images,
                room=room,
                debug=debug,
                conversation=conversation,
            )
        except AgentChatError as exc:
            print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        finally:
            images = []


async def run_cli(args: argparse.Namespace) -> int:
    if args.platform_list:
        rows = [{"name": scope.name, "scope": scope.value} for scope in SupportScope]
        if args.json:
            print(json.dumps(rows, ensure_ascii=False, indent=2))
        else:
            print("平台标识（区分大小写，可直接用于配置）：")
            print("此列表不代表当前已连接或已完成兼容性验证的平台。")
            for row in rows:
                print(row["scope"])
        return 0

    env_file: Path | None = None
    env_owned: set[str] = set()
    if not args.no_env:
        resolved: Path = (
            args.env_file
            if args.env_file is not None
            else Path(os.getenv("AGENT_CHAT_ENV_FILE", ".env.agent_chat"))
        )
        env_file = resolved
        env_owned = load_into_environ(resolved)
    else:
        # --no-env stops the loading, not the editing: an edit still needs a
        # concrete file, so only an explicit --env-file counts.
        env_file = args.env_file
    if args.render_image is not None:
        return await _render_image_command(args)

    # Edits and reload requests belong to the bot's store, not the debug one.
    bot_data_dir = production_data_dir(args, env_file)
    if is_settings_action(args):
        # A single invalid dotenv value must not lock the operator out of the
        # commands that repair it: edit actions run against a best-effort config
        # and report the keys that had to fall back to their defaults.
        editing_config, invalid_keys = config_editor.build_editing_config()
        profile_dir, _data_dir, _selected = _paths(args, editing_config)
        return await run_settings_command(
            args,
            env_file,
            env_owned,
            editing_config,
            profile_dir,
            bot_data_dir,
            invalid_keys=invalid_keys,
        )
    environment_config = config_editor.environment_config()
    profile_dir, data_dir, selected = _paths(args, environment_config)
    if args.profile_list or args.check:
        return inspect_profiles(
            profile_dir,
            selected,
            PromptRegistry(
                data_dir / "prompts",
                environment_config.agent_chat_default_system_prompt_file,
            ),
            check_credentials=args.check,
        )
    # Rules steer the running bot, so they belong in the bot's own store; the
    # debug dir stays for plain local asks and rooms. A --scope ask resolves
    # rules too, so it reads the same store unless --data-dir overrides it.
    rule_action = bool(args.rule_list or args.rule_set is not None or args.rule_unset)
    reads_bot_store = rule_action or (args.scope and not args.room)
    if reads_bot_store and args.data_dir is None:
        data_dir = environment_config.agent_chat_data_dir
    config = environment_config.model_copy(
        update={
            "agent_chat_data_dir": data_dir,
            "agent_chat_profile_dir": profile_dir,
            "agent_chat_default_profile": selected,
            "agent_chat_room_enabled": (
                args.room or environment_config.agent_chat_room_enabled
            ),
        }
    )
    service = AgentChatService(
        config,
        env_file=env_file,
        env_owned=env_owned,
    )
    try:
        # Managing rules or trialling them against the bot's store must never
        # touch the running bot's runs metadata.
        await service.initialize(interrupt_running=not reads_bot_store)
        runtime_profile = selected or service.active_profile_name
        if args.reasoning_effort:
            if runtime_profile is None:
                raise ConfigurationError(
                    "--reasoning-effort requires --profile or an active profile"
                )
            service.profiles.override_for_runtime(
                runtime_profile,
                reasoning_effort=args.reasoning_effort,
            )
        if runtime_profile and (selected or args.reasoning_effort):
            await service.use_profile(runtime_profile)
        if service.startup_error:
            raise ConfigurationError(service.startup_error)
        if rule_action:
            return await _rule_command(service, args)

        conversation = None
        if args.scope:
            conversation = ConversationRef(
                scope=args.scope,
                target_id=args.group or "",
                private=args.private,
            )
        images = await _load_images(args.image, config)
        prompt = " ".join(args.prompt).strip()
        if not prompt and not sys.stdin.isatty():
            prompt = sys.stdin.read().strip()

        if args.room:
            await service.create_room(_DEBUG_CONTEXT, args.room_name)
        if prompt or images:
            await _ask(
                service,
                prompt,
                images,
                room=args.room,
                debug=args.debug,
                conversation=conversation,
            )
        else:
            await _interactive(
                service,
                images,
                room=args.room,
                room_name=args.room_name,
                debug=args.debug,
                conversation=conversation,
                env_file=env_file,
                profile_dir=profile_dir,
                bot_data_dir=bot_data_dir,
            )
        return 0
    finally:
        await service.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return asyncio.run(run_cli(args))
    except KeyboardInterrupt:
        print("\ninterrupted", file=sys.stderr)
        # Ctrl+C must always leave the terminal usable, whatever the cleanup is
        # doing: wake a blocked stdin read, then guarantee an exit -- a second
        # Ctrl+C (or the grace timer) skips the interpreter's thread joins.
        _release_stdin()
        _install_interrupt_escape()
        return 130
    except AgentChatError as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        if getattr(args, "debug", False):
            raise
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
