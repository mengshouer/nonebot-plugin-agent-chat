import builtins
import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nonebot_plugin_agent_chat import env_file
from nonebot_plugin_agent_chat.cli import (
    _interactive_command,
    build_parser,
    inspect_profiles,
    run_cli,
)
from nonebot_plugin_agent_chat.config import Config
from nonebot_plugin_agent_chat.errors import InputError
from nonebot_plugin_agent_chat.models import RunResult, Usage
from nonebot_plugin_agent_chat.prompts import PromptRegistry
from nonebot_plugin_agent_chat.service import RELOAD_MARKER_FILE, AgentChatService
from nonebot_plugin_agent_chat.storage import AgentStore


class PlatformListTests(unittest.IsolatedAsyncioTestCase):
    def test_argument_help_is_chinese(self) -> None:
        parser = build_parser()
        self.assertRegex(parser.description, r"[一-鿿]")
        for action in parser._actions:
            with self.subTest(argument=action.dest):
                self.assertIsInstance(action.help, str)
                self.assertRegex(action.help, r"[一-鿿]")
        text = parser.format_help()
        for label in ("位置参数", "选项", "显示帮助并退出"):
            self.assertIn(label, text)

    def test_help_explains_platform_ids_without_component_names(self) -> None:
        text = build_parser().format_help()
        self.assertIn("--scope PLATFORM", text)
        self.assertIn("平台标识", text)
        self.assertIn("QQClient", text)
        self.assertIn("Telegram", text)
        for jargon in ("UniSeg", "SupportScope"):
            self.assertNotIn(jargon, text)

    def test_platform_list_json_works_in_a_fresh_process_before_setup(self) -> None:
        from nonebot_plugin_alconna.uniseg.constraint import SupportScope

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = root / ".env.agent_chat"
            config.write_text("AGENT_CHAT_ALLOWED_GROUPS=not-json\n")
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "nonebot_plugin_agent_chat",
                    "--platform-list",
                    "--json",
                ],
                cwd=root,
                env={"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stderr, "")
            self.assertEqual(
                json.loads(result.stdout),
                [{"name": scope.name, "scope": scope.value} for scope in SupportScope],
            )
            self.assertEqual(list(root.iterdir()), [config])

    async def test_lists_upstream_scopes_without_loading_runtime_state(self) -> None:
        from nonebot_plugin_alconna.uniseg import SupportScope

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            invalid_env = root / "invalid.env"
            invalid_env.write_text("AGENT_CHAT_MAX_SEARCHES=not-an-integer\n")
            for structured in (False, True):
                with self.subTest(json=structured):
                    argv = ["--platform-list", "--env-file", str(invalid_env)]
                    if structured:
                        argv.append("--json")
                    args = build_parser().parse_args(argv)
                    output = io.StringIO()
                    with (
                        patch.dict(
                            os.environ,
                            {"AGENT_CHAT_ALLOWED_GROUPS": "invalid"},
                            clear=True,
                        ),
                        patch(
                            "nonebot_plugin_agent_chat.cli.load_into_environ",
                            side_effect=AssertionError("dotenv must not be loaded"),
                        ),
                        patch(
                            "nonebot_plugin_agent_chat.cli._paths",
                            side_effect=AssertionError("profiles must not be read"),
                        ),
                        patch(
                            "nonebot_plugin_agent_chat.cli.AgentChatService",
                            side_effect=AssertionError("service must not be created"),
                        ),
                        patch(
                            "nonebot_plugin_agent_chat.cli.config_editor.environment_config",
                            side_effect=AssertionError("config must not be parsed"),
                        ),
                        contextlib.redirect_stdout(output),
                    ):
                        self.assertEqual(await run_cli(args), 0)
                    if structured:
                        self.assertEqual(
                            json.loads(output.getvalue()),
                            [
                                {"name": scope.name, "scope": scope.value}
                                for scope in SupportScope
                            ],
                        )
                    else:
                        lines = output.getvalue().splitlines()
                        self.assertIn("平台标识", lines[0])
                        self.assertIn("不代表当前已连接或已完成兼容性验证", lines[1])
                        self.assertEqual(
                            lines[2:], [scope.value for scope in SupportScope]
                        )
                        self.assertNotIn("UniSeg", output.getvalue())
            self.assertEqual(list(root.iterdir()), [invalid_env])


class CliTests(unittest.IsolatedAsyncioTestCase):
    def test_list_profiles_uses_filename_as_id(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "local.json").write_text(
                json.dumps(
                    {
                        "protocol": "openai-completions",
                        "model": "local-model",
                    }
                ),
                encoding="utf-8",
            )
            output = io.StringIO()
            with contextlib.redirect_stdout(output):
                status = inspect_profiles(
                    root,
                    None,
                    PromptRegistry(root / "prompts"),
                    check_credentials=False,
                )
            self.assertEqual(status, 0)
            self.assertIn("local: protocol=openai-completions", output.getvalue())

    async def test_list_profiles_uses_data_dir_and_configured_prompt_name(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            profiles = root / "profiles"
            prompt_dir = root / "data" / "prompts" / "personal"
            profiles.mkdir()
            prompt_dir.mkdir(parents=True)
            (profiles / "local.json").write_text(
                json.dumps({"protocol": "openai-completions", "model": "local-model"}),
                encoding="utf-8",
            )
            (prompt_dir / "zh.md").write_text("global prompt", encoding="utf-8")
            args = build_parser().parse_args(
                [
                    "--no-env",
                    "--profiles-dir",
                    str(profiles),
                    "--data-dir",
                    str(root / "data"),
                    "--profile-list",
                ]
            )
            output = io.StringIO()
            with (
                patch.dict(
                    os.environ,
                    {"AGENT_CHAT_DEFAULT_SYSTEM_PROMPT_FILE": "personal/zh.md"},
                ),
                contextlib.redirect_stdout(output),
            ):
                status = await run_cli(args)

            self.assertEqual(status, 0)
            self.assertIn("prompt=global:personal/zh.md", output.getvalue())

    async def test_store_redirect_and_interrupt_flag(self) -> None:
        """Rule commands and --scope asks use the bot store; debug asks don't."""

        captured: dict[str, object] = {}

        class FakeService:
            startup_error = None
            active_profile_name = "local"

            def __init__(self, config, **kwargs: object) -> None:
                self.config = config
                captured["data_dir"] = config.agent_chat_data_dir

            async def initialize(self, *, interrupt_running: bool = True) -> None:
                captured["interrupt_running"] = interrupt_running

            async def use_profile(self, name: str) -> None:
                return None

            async def ask(self, collected, **kwargs) -> RunResult:
                return RunResult(
                    text="ok",
                    sources=[],
                    usage=Usage(total_tokens=1, reasoning_tokens=0),
                    model_turns=1,
                    local_tool_calls=0,
                    searches=0,
                    actual_profile="local",
                    run_id="debug-run",
                )

            async def rule_status(self) -> list[dict[str, object]]:
                return []

            async def close(self) -> None:
                return None

        env = {
            "AGENT_CHAT_DATA_DIR": "/tmp/bot-store",
            "AGENT_CHAT_DEBUG_DATA_DIR": "/tmp/debug-store",
        }
        for argv, expect_bot_store in (
            (["--no-env", "--profile", "local", "hello"], False),
            (["--no-env", "--profile", "local", "--scope", "QQClient", "hi"], True),
            (["--no-env", "--profile", "local", "--rule-list"], True),
        ):
            with self.subTest(argv=argv):
                captured.clear()
                args = build_parser().parse_args(argv)
                with (
                    patch(
                        "nonebot_plugin_agent_chat.cli.AgentChatService", FakeService
                    ),
                    patch.dict(os.environ, env),
                    contextlib.redirect_stdout(io.StringIO()),
                    contextlib.redirect_stderr(io.StringIO()),
                ):
                    status = await run_cli(args)

                self.assertEqual(status, 0)
                data_dir = str(captured["data_dir"])
                expected = "/tmp/bot-store" if expect_bot_store else "/tmp/debug-store"
                self.assertTrue(data_dir.startswith(expected), data_dir)
                self.assertEqual(captured["interrupt_running"], not expect_bot_store)

    async def test_one_shot_cli_runs_without_nonebot(self) -> None:
        closed = False
        captured_config: Config | None = None
        captured_ask_kwargs: dict[str, object] | None = None

        captured_service_kwargs: dict[str, object] = {}

        class FakeService:
            startup_error = None

            def __init__(self, config, **kwargs: object) -> None:
                nonlocal captured_config
                self.config = config
                captured_config = config
                captured_service_kwargs.update(kwargs)

            async def initialize(self, *, interrupt_running: bool = True) -> None:
                return None

            async def use_profile(self, name: str) -> None:
                self.profile = name

            async def ask(self, collected, **kwargs) -> RunResult:
                nonlocal captured_ask_kwargs
                self.prompt = collected.text
                captured_ask_kwargs = kwargs
                return RunResult(
                    text="local answer",
                    sources=[],
                    usage=Usage(total_tokens=3, reasoning_tokens=2),
                    model_turns=1,
                    local_tool_calls=0,
                    searches=0,
                    actual_profile="local",
                    run_id="debug-run",
                )

            async def close(self) -> None:
                nonlocal closed
                closed = True

        args = build_parser().parse_args(
            ["--no-env", "--profile", "local", "--debug", "hello", "world"]
        )
        output = io.StringIO()
        error = io.StringIO()
        with (
            patch("nonebot_plugin_agent_chat.cli.AgentChatService", FakeService),
            patch.dict(os.environ, {"AGENT_CHAT_MAX_MODEL_TURNS": "9"}),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(error),
        ):
            status = await run_cli(args)

        self.assertEqual(status, 0)
        self.assertIn("local answer", output.getvalue())
        self.assertIn('"reasoning_tokens": 2', error.getvalue())
        self.assertIsNotNone(captured_config)
        assert captured_config is not None
        self.assertEqual(captured_config.agent_chat_max_model_turns, 9)
        self.assertIsNotNone(captured_ask_kwargs)
        assert captured_ask_kwargs is not None
        self.assertNotIn("enforce_cooldown", captured_ask_kwargs)
        # --no-env must not hand the service a dotenv file to reload from.
        self.assertIsNone(captured_service_kwargs.get("env_file"))
        self.assertTrue(closed)


class EditCommandTests(unittest.IsolatedAsyncioTestCase):
    """The settings/profile editors run without a bot process, then ask for a reload."""

    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.profiles = self.root / "profiles"
        self.profiles.mkdir()
        (self.profiles / "default.json").write_text(
            json.dumps({"protocol": "openai-responses", "model": "local"}),
            encoding="utf-8",
        )
        self.env_path = self.root / ".env.agent_chat"
        self.env_path.write_text(
            "# comment\nAGENT_CHAT_MAX_SEARCHES=3\n", encoding="utf-8"
        )
        self.data = self.root / "data"

    async def asyncTearDown(self) -> None:
        self.temp.cleanup()

    async def _run(
        self, *extra: str, no_env: bool = True, with_data_dir: bool = True
    ) -> str:
        argv = ["--env-file", str(self.env_path), "--profiles-dir", str(self.profiles)]
        if with_data_dir:
            argv += ["--data-dir", str(self.data)]
        argv += list(extra)
        if no_env:
            argv.insert(0, "--no-env")
        args = build_parser().parse_args(argv)
        output = io.StringIO()
        error = io.StringIO()
        # Loading the test dotenv mutates the process environment; restore it.
        with (
            patch.dict(os.environ, {}, clear=False),
            contextlib.redirect_stdout(output),
            contextlib.redirect_stderr(error),
        ):
            status = await run_cli(args)
        self.assertEqual(status, 0, error.getvalue())
        return output.getvalue()

    async def _store(self) -> AgentStore:
        store = AgentStore(self.data / "agent_chat.db")
        await store.initialize(interrupt_running=False)
        return store

    async def test_config_set_writes_file_and_requests_reload(self) -> None:
        out = await self._run("--config-set", "AGENT_CHAT_MAX_SEARCHES=9")
        self.assertIn("已设置 AGENT_CHAT_MAX_SEARCHES=9", out)
        text = self.env_path.read_text(encoding="utf-8")
        self.assertIn("# comment", text)
        self.assertIn("AGENT_CHAT_MAX_SEARCHES=9", text)
        self.assertTrue((self.data / RELOAD_MARKER_FILE).is_file())

    async def test_config_set_restart_only_key_warns(self) -> None:
        out = await self._run("--config-set", "AGENT_CHAT_DATA_DIR=/tmp/x")
        self.assertIn("需重启", out)

    async def test_config_set_rejects_unknown_key_and_bad_value(self) -> None:
        with self.assertRaises(InputError):
            await self._run("--config-set", "NOPE=1")
        with self.assertRaises(InputError):
            await self._run("--config-set", "AGENT_CHAT_MAX_SEARCHES=nope")

    async def test_config_unset_reports_and_removes(self) -> None:
        out = await self._run("--config-unset", "AGENT_CHAT_MAX_SEARCHES")
        self.assertIn("已删除", out)
        self.assertNotIn("AGENT_CHAT_MAX_SEARCHES", self.env_path.read_text())
        out = await self._run("--config-unset", "AGENT_CHAT_MAX_SEARCHES")
        self.assertIn("未改动", out)

    async def test_config_list_json_reports_sources(self) -> None:
        with patch.dict(os.environ, {"AGENT_CHAT_MAX_MODEL_TURNS": "7"}, clear=False):
            payload = json.loads(
                await self._run("--config-list", "--json", no_env=False)
            )
        entries = {item["key"]: item for item in payload["entries"]}
        # The file provided this one...
        self.assertEqual(entries["AGENT_CHAT_MAX_SEARCHES"]["source"], "file")
        self.assertEqual(entries["AGENT_CHAT_MAX_SEARCHES"]["value"], "3")
        # ...the real environment this one...
        self.assertEqual(entries["AGENT_CHAT_MAX_MODEL_TURNS"]["source"], "env")
        # ...and nothing provided the rest.
        self.assertEqual(entries["AGENT_CHAT_ENABLE_AT"]["source"], "default")
        self.assertTrue(entries["AGENT_CHAT_DATA_DIR"]["restart_required"])
        self.assertEqual(payload["unmanaged_keys"], [])

    async def test_config_list_json_reports_invalid_keys(self) -> None:
        self.env_path.write_text(
            "AGENT_CHAT_MESSAGE_CHUNK_CHARS=100\n", encoding="utf-8"
        )

        payload = json.loads(await self._run("--config-list", "--json", no_env=False))

        self.assertEqual(payload["invalid_keys"], ["AGENT_CHAT_MESSAGE_CHUNK_CHARS"])

    async def test_batch_edit_applies_every_change_together(self) -> None:
        out = await self._run(
            "--config-set",
            "AGENT_CHAT_MAX_SEARCHES=9",
            "--config-set",
            "AGENT_CHAT_MAX_MODEL_TURNS=4",
        )

        text = self.env_path.read_text(encoding="utf-8")
        self.assertIn("AGENT_CHAT_MAX_SEARCHES=9", text)
        self.assertIn("AGENT_CHAT_MAX_MODEL_TURNS=4", text)
        self.assertIn("已设置 AGENT_CHAT_MAX_SEARCHES=9", out)
        self.assertIn("已设置 AGENT_CHAT_MAX_MODEL_TURNS=4", out)

    async def test_batch_edit_validates_before_writing_anything(self) -> None:
        """A rejected batch must not leave earlier lines behind."""

        with self.assertRaises(InputError):
            await self._run(
                "--config-set",
                "AGENT_CHAT_MAX_SEARCHES=9",
                "--config-set",
                "AGENT_CHAT_MESSAGE_CHUNK_CHARS=100",
            )

        text = self.env_path.read_text(encoding="utf-8")
        self.assertIn("AGENT_CHAT_MAX_SEARCHES=3", text)
        self.assertNotIn("AGENT_CHAT_MAX_SEARCHES=9", text)

    async def test_profile_batch_validates_before_writing_anything(self) -> None:
        with self.assertRaises(InputError):
            await self._run(
                "--profile-set",
                "default",
                "model=better",
                "--profile-set",
                "missing",
                "model=x",
            )

        raw = json.loads((self.profiles / "default.json").read_text(encoding="utf-8"))
        self.assertEqual(raw["model"], "local")

    async def test_invalid_dotenv_value_can_still_be_repaired(self) -> None:
        """The command that fixes a bad value must not be blocked by it."""

        self.env_path.write_text(
            "AGENT_CHAT_MESSAGE_CHUNK_CHARS=100\n", encoding="utf-8"
        )

        out = await self._run(
            "--config-unset", "AGENT_CHAT_MESSAGE_CHUNK_CHARS", no_env=False
        )

        self.assertIn("无效", out)
        self.assertIn("已删除", out)
        self.assertEqual(self.env_path.read_text(encoding="utf-8"), "")

    async def test_profile_set_unset_and_new(self) -> None:
        await self._run("--profile-set", "default", "model=better")
        raw = json.loads((self.profiles / "default.json").read_text(encoding="utf-8"))
        self.assertEqual(raw["model"], "better")

        out = await self._run("--profile-unset", "default", "max_output_tokens")
        self.assertIn("没有设置", out)

        await self._run("--profile-new", "qq-safe", "--from", "default")
        copied = json.loads(
            (self.profiles / "qq-safe.json").read_text(encoding="utf-8")
        )
        self.assertEqual(copied["model"], "better")

    async def test_profile_set_rejects_bad_value(self) -> None:
        with self.assertRaises(InputError):
            await self._run("--profile-set", "default", "image_reply_mode=sometimes")

    async def test_profile_remove_warns_about_rule_references(self) -> None:
        store = await self._store()
        try:
            await store.set_profile_rule(
                scope="QQClient",
                target_id="598683145",
                profile="qq-safe",
                updated_by="test",
            )
        finally:
            await store.close()
        await self._run("--profile-new", "qq-safe", "--from", "default")

        out = await self._run("--profile-remove", "qq-safe")

        self.assertIn("规则", out)
        self.assertFalse((self.profiles / "qq-safe.json").exists())

    async def test_config_explain_prints_the_full_explanation(self) -> None:
        # Load the dotenv file so the source really is "来自 dotenv 文件".
        out = await self._run(
            "--config-explain", "AGENT_CHAT_MAX_SEARCHES", no_env=False
        )
        self.assertIn("说明：一次回答最多搜索几次", out)
        self.assertIn("类型：整数", out)
        self.assertIn("默认：10", out)
        self.assertIn("当前：3（来自 dotenv 文件）", out)
        self.assertIn("热生效", out)
        priority = await self._run("--config-explain", "AGENT_CHAT_PRIORITY")
        self.assertIn("需重启", priority)

    async def test_config_explain_names_unknown_keys(self) -> None:
        out = await self._run("--config-explain", "agent_chat_nope")
        self.assertIn("未知设置项", out)

    async def test_config_list_ends_with_the_marker_legend(self) -> None:
        out = await self._run("--config-list")
        self.assertIn("标记含义", out)
        self.assertIn("[需重启]", out)
        self.assertIn("未管理", out)
        self.assertIn("--config-explain", out)

    async def test_config_explain_cannot_be_combined_with_edits(self) -> None:
        with self.assertRaises(InputError):
            await self._run(
                "--config-explain",
                "AGENT_CHAT_MAX_SEARCHES",
                "--config-set",
                "AGENT_CHAT_MAX_SEARCHES=5",
            )

    async def test_config_list_cannot_be_combined_with_edits(self) -> None:
        with self.assertRaises(InputError):
            await self._run(
                "--config-list", "--config-set", "AGENT_CHAT_MAX_SEARCHES=5"
            )
        with self.assertRaises(InputError):
            await self._run("--config-list", "--reload")
        with self.assertRaises(InputError):
            await self._run("--config-list", "--profile-list")

    async def test_profile_set_echo_masks_credentials(self) -> None:
        out = await self._run(
            "--profile-set",
            "default",
            'extra_headers={"Authorization": "Bearer s3cret"}',
        )
        self.assertNotIn("s3cret", out)
        self.assertIn("•••", out)
        raw = json.loads((self.profiles / "default.json").read_text(encoding="utf-8"))
        self.assertEqual(raw["extra_headers"]["Authorization"], "Bearer s3cret")

    async def test_marker_follows_the_dotenv_data_dir(self) -> None:
        elsewhere = self.root / "from_env_file"
        self.env_path.write_text(f"AGENT_CHAT_DATA_DIR={elsewhere}\n", encoding="utf-8")
        out = await self._run("--reload", with_data_dir=False)
        self.assertIn("已请求重载", out)
        self.assertTrue((elsewhere / RELOAD_MARKER_FILE).is_file())
        self.assertFalse((self.data / "agent_chat.db").exists())

    async def test_editor_needs_a_terminal(self) -> None:
        with self.assertRaises(InputError):
            await self._run("--profile-edit", "default")

    async def test_reload_flag_requests_marker_only(self) -> None:
        out = await self._run("--reload")
        self.assertIn("已请求重载", out)
        self.assertTrue((self.data / RELOAD_MARKER_FILE).is_file())


_DEFAULT_ENV = object()


class InteractiveCommandTests(unittest.IsolatedAsyncioTestCase):
    """`:` commands let the operator configure without leaving the session."""

    async def asyncSetUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.profiles = self.root / "profiles"
        self.profiles.mkdir()
        (self.profiles / "default.json").write_text(
            json.dumps({"protocol": "openai-responses", "model": "local"}),
            encoding="utf-8",
        )
        self.env_path = self.root / ".env.agent_chat"
        self.env_path.write_text("AGENT_CHAT_MAX_SEARCHES=3\n", encoding="utf-8")
        self.bot_data = self.root / "bot_data"
        self.env_patch = patch.dict(os.environ, {}, clear=True)
        self.env_patch.start()
        self.service = AgentChatService(
            Config(
                _env_file=None,
                agent_chat_data_dir=self.root / "session",
                agent_chat_profile_dir=self.profiles,
                agent_chat_default_profile="default",
                agent_chat_cleanup_interval_seconds=0,
                agent_chat_max_searches=3,
            ),
            env_file=self.env_path,
        )
        self.service.env_owned = env_file.load_into_environ(self.env_path)
        await self.service.initialize()

    async def asyncTearDown(self) -> None:
        await self.service.close()
        self.env_patch.stop()
        self.temp.cleanup()

    async def _command(
        self, line: str, *, room: bool = False, env_path: object = _DEFAULT_ENV
    ) -> tuple[bool, str, str]:
        output = io.StringIO()
        error = io.StringIO()
        resolved_env = self.env_path if env_path is _DEFAULT_ENV else env_path
        with contextlib.redirect_stdout(output), contextlib.redirect_stderr(error):
            quit_ = await _interactive_command(
                self.service,
                line,
                env_file=resolved_env,  # type: ignore[arg-type]
                profile_dir=self.profiles,
                bot_data_dir=self.bot_data,
                room=room,
                room_name="local-debug",
            )
        return quit_, output.getvalue(), error.getvalue()

    async def _marker(self) -> str | None:
        path = self.bot_data / RELOAD_MARKER_FILE
        try:
            return path.read_text(encoding="utf-8").strip()
        except OSError:
            return None

    async def test_help_and_unknown_command(self) -> None:
        quit_, out, _ = await self._command(":help")
        self.assertFalse(quit_)
        self.assertIn(":config set KEY=VALUE", out)
        self.assertIn(":profile edit", out)
        # README documents `:config <KEY>` and `:config explain KEY`.
        self.assertIn(":config explain KEY", out)
        self.assertIn(":config <KEY>", out)
        _, _, err = await self._command(":nope")
        self.assertIn(":help", err)

    def test_json_help_covers_the_settings_flags(self) -> None:
        help_text = build_parser().format_help()
        self.assertIn("--config-*", help_text)
        self.assertIn("--profile-*", help_text)

    async def test_quit_ends_the_session(self) -> None:
        for line in (":quit", ":q", ":exit"):
            self.assertTrue((await self._command(line))[0])

    async def test_config_list_shows_value_and_source(self) -> None:
        _, out, _ = await self._command(":config")
        self.assertIn("AGENT_CHAT_MAX_SEARCHES=3", out)
        self.assertIn("[文件]", out)
        self.assertIn("标记含义", out)

    async def test_config_key_explains_one_setting(self) -> None:
        _, out, _ = await self._command(":config AGENT_CHAT_MAX_SEARCHES")
        self.assertIn("说明：一次回答最多搜索几次", out)
        self.assertIn("当前：3（来自 dotenv 文件）", out)

        _, out, _ = await self._command(":config explain agent_chat_nope")
        self.assertIn("未知设置项", out)

    async def test_config_set_writes_reloads_and_requests_the_bot(self) -> None:
        _, out, _ = await self._command(":config set AGENT_CHAT_MAX_SEARCHES=5")
        self.assertIn("已设置 AGENT_CHAT_MAX_SEARCHES=5", out)
        self.assertIn("已写入", out)
        self.assertIn("本会话已重载", out)
        self.assertIn("AGENT_CHAT_MAX_SEARCHES=5", self.env_path.read_text())
        self.assertEqual(self.service.config.agent_chat_max_searches, 5)
        self.assertIsNotNone(await self._marker())

    async def test_config_set_rejects_unknown_key_without_writing(self) -> None:
        with self.assertRaises(InputError):
            await self._command(":config set NOPE=1")
        self.assertNotIn("NOPE", self.env_path.read_text())
        self.assertIsNone(await self._marker())

    async def test_config_unset_falls_back_in_the_same_session(self) -> None:
        await self._command(":config set AGENT_CHAT_MAX_SEARCHES=5")
        _, out, _ = await self._command(":config unset AGENT_CHAT_MAX_SEARCHES")
        self.assertIn("已删除", out)
        self.assertEqual(self.service.config.agent_chat_max_searches, 10)
        self.assertNotIn("AGENT_CHAT_MAX_SEARCHES", self.env_path.read_text())

    async def test_config_edit_needs_a_terminal(self) -> None:
        with self.assertRaises(InputError):
            await self._command(":config edit")

    async def test_config_set_needs_an_env_file(self) -> None:
        with self.assertRaises(InputError) as caught:
            await self._command(":config set AGENT_CHAT_MAX_SEARCHES=5", env_path=None)
        self.assertIn("--env-file", str(caught.exception))

    async def test_profile_set_reloads_the_session(self) -> None:
        _, out, _ = await self._command(":profile set default model=better")
        self.assertIn("已设置 default.model=better", out)
        raw = json.loads((self.profiles / "default.json").read_text(encoding="utf-8"))
        self.assertEqual(raw["model"], "better")
        self.assertEqual(self.service.profiles.get("default").config.model, "better")

    async def test_profile_new_and_remove_warn_about_rules(self) -> None:
        store = AgentStore(self.bot_data / "agent_chat.db")
        await store.initialize(interrupt_running=False)
        try:
            await store.set_profile_rule(
                scope="QQClient",
                target_id="598683145",
                profile="qq-safe",
                updated_by="test",
            )
        finally:
            await store.close()

        await self._command(":profile new qq-safe --from default")
        self.assertTrue((self.profiles / "qq-safe.json").is_file())
        _, out, _ = await self._command(":profile remove qq-safe")
        self.assertIn("规则", out)
        self.assertFalse((self.profiles / "qq-safe.json").exists())

    async def test_profile_new_without_from_copies_the_first_profile(self) -> None:
        # `--from` is optional in the session (the :help brackets say so): the
        # first existing profile is the template, like the editor's prefill.
        _, out, _ = await self._command(":profile new mirrored")

        self.assertIn("mirrored", out)
        created = json.loads((self.profiles / "mirrored.json").read_text("utf-8"))
        self.assertEqual(
            created,
            json.loads((self.profiles / "default.json").read_text("utf-8")),
        )

    async def test_reload_requests_the_bot_and_reports(self) -> None:
        _, out, _ = await self._command(":reload")
        self.assertIn("已请求重载", out)
        self.assertIsNotNone(await self._marker())


class InterruptHandlingTests(unittest.TestCase):
    """Ctrl+C must always give the terminal back, whatever cleanup is doing."""

    def test_release_stdin_writes_to_the_controlling_terminal(self) -> None:

        from nonebot_plugin_agent_chat import cli as cli_module

        written = io.StringIO()

        class FakeTty:
            def write(self, text: str) -> None:
                written.write(text)

            def flush(self) -> None:
                pass

            def __enter__(self):
                return self

            def __exit__(self, *args: object) -> None:
                return None

        class TtyStdin:
            def isatty(self) -> bool:
                return True

        with (
            patch.object(sys, "stdin", TtyStdin()),
            patch.object(builtins, "open", lambda *a, **k: FakeTty()),
        ):
            cli_module._release_stdin()
        self.assertEqual(written.getvalue(), "\n")

    def test_release_stdin_ignores_pipes(self) -> None:
        import builtins

        from nonebot_plugin_agent_chat import cli as cli_module

        class PipeStdin:
            def isatty(self) -> bool:
                return False

        def fail_open(*args: object, **kwargs: object) -> None:
            raise AssertionError("a pipe has no terminal to wake")

        with (
            patch.object(sys, "stdin", PipeStdin()),
            patch.object(builtins, "open", fail_open),
        ):
            cli_module._release_stdin()

    def test_hard_exit_skips_atexit(self) -> None:

        from nonebot_plugin_agent_chat import cli as cli_module

        with patch.object(cli_module.os, "_exit") as exit_mock:
            cli_module._hard_exit(130)
        exit_mock.assert_called_once_with(130)

    def test_interrupt_escape_installs_handler_and_grace_timer(self) -> None:

        from nonebot_plugin_agent_chat import cli as cli_module

        signals: list[tuple[int, object]] = []
        timers: list[tuple[float, object, tuple[int, ...]]] = []

        class FakeTimer:
            def __init__(self, interval: float, function: object, args: tuple = ()):
                timers.append((interval, function, args))
                self.daemon = False

            def start(self) -> None:
                pass

        with (
            patch.object(
                cli_module.signal,
                "signal",
                lambda number, handler: signals.append((number, handler)),
            ),
            patch.object(cli_module.threading, "Timer", FakeTimer),
            patch.object(cli_module, "_hard_exit") as hard_exit,
        ):
            cli_module._install_interrupt_escape(grace_seconds=1.5)
            handler = signals[-1][1]
            handler(cli_module.signal.SIGINT, None)

        self.assertEqual(signals[-1][0], cli_module.signal.SIGINT)
        # The timer must call the same exit helper (patched here).
        self.assertEqual(timers, [(1.5, hard_exit, (130,))])
        hard_exit.assert_called_once_with(130)

    def test_main_returns_130_and_installs_the_escape(self) -> None:

        from nonebot_plugin_agent_chat import cli as cli_module

        released: list[str] = []
        escapes: list[str] = []

        async def boom(args: object) -> int:
            raise KeyboardInterrupt

        with (
            patch.object(cli_module, "run_cli", boom),
            patch.object(cli_module, "_release_stdin", lambda: released.append("x")),
            patch.object(
                cli_module,
                "_install_interrupt_escape",
                lambda *a, **k: escapes.append("x"),
            ),
            patch.object(cli_module.sys, "stderr", new=io.StringIO()),
        ):
            status = cli_module.main([])

        self.assertEqual(status, 130)
        self.assertEqual(released, ["x"])
        self.assertEqual(escapes, ["x"])


if __name__ == "__main__":
    unittest.main()
