"""Default directory resolution: the store dirs, with the legacy layout kept."""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from nonebot_plugin_agent_chat import config as config_module
from nonebot_plugin_agent_chat.config import Config

ENV_KEYS = ("AGENT_CHAT_DATA_DIR", "AGENT_CHAT_PROFILE_DIR")


class FakeStore(types.ModuleType):
    """Minimal stand-in for nonebot_plugin_localstore's directory helpers."""

    def __init__(self, data_dir: Path, config_dir: Path, error: Exception | None):
        super().__init__("nonebot_plugin_localstore")
        self.calls: list[str] = []
        self._data_dir = data_dir
        self._config_dir = config_dir
        self._error = error

    def get_plugin_data_dir(self) -> Path:
        self.calls.append("data")
        if self._error is not None:
            raise self._error
        return self._data_dir

    def get_plugin_config_dir(self) -> Path:
        self.calls.append("config")
        if self._error is not None:
            raise self._error
        return self._config_dir


class StoreDirTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        patched = mock.patch.dict(os.environ)
        patched.start()
        self.addCleanup(patched.stop)
        for key in ENV_KEYS:
            os.environ.pop(key, None)

    def fake_store(self, error: Exception | None = None) -> FakeStore:
        store = FakeStore(self.root / "store-data", self.root / "store-config", error)
        patched = mock.patch.dict(sys.modules, {"nonebot_plugin_localstore": store})
        patched.start()
        self.addCleanup(patched.stop)
        return store

    def patch_legacy(self, legacy: Path, *, exists: bool) -> None:
        if exists:
            (legacy / "profiles").mkdir(parents=True)
        patched = mock.patch.object(config_module, "LEGACY_DATA_DIR", legacy)
        patched.start()
        self.addCleanup(patched.stop)

    def test_localstore_dirs_are_used_when_the_legacy_layout_is_absent(self) -> None:
        legacy = self.root / "checkout" / "data" / "agent_chat"
        self.patch_legacy(legacy, exists=False)
        store = self.fake_store()

        config = Config()

        self.assertEqual(config.agent_chat_data_dir, self.root / "store-data")
        self.assertEqual(
            config.agent_chat_profile_dir, self.root / "store-config" / "profiles"
        )
        # One shared before-validator lookup fills both missing fields.
        self.assertEqual(store.calls.count("data"), 1)
        self.assertEqual(store.calls.count("config"), 1)

    def test_existing_legacy_directory_wins_over_localstore(self) -> None:
        legacy = self.root / "checkout" / "data" / "agent_chat"
        self.patch_legacy(legacy, exists=True)
        store = self.fake_store()

        config = Config()

        self.assertEqual(config.agent_chat_data_dir, legacy)
        self.assertEqual(config.agent_chat_profile_dir, legacy / "profiles")
        self.assertEqual(store.calls, [])

    def test_debug_only_directory_does_not_select_the_legacy_layout(self) -> None:
        legacy = self.root / "checkout" / "data" / "agent_chat"
        (legacy / "debug").mkdir(parents=True)
        self.patch_legacy(legacy, exists=False)
        store = self.fake_store()

        config = Config()

        self.assertEqual(config.agent_chat_data_dir, self.root / "store-data")
        self.assertEqual(
            config.agent_chat_profile_dir, self.root / "store-config" / "profiles"
        )
        self.assertEqual(store.calls, ["data", "config"])

    def test_debug_or_reload_marker_alone_does_not_select_the_legacy_layout(
        self,
    ) -> None:
        legacy = self.root / "checkout" / "data" / "agent_chat"
        legacy.mkdir(parents=True)
        (legacy / "debug").mkdir()
        (legacy / "reload.request").write_text("", encoding="utf-8")
        self.patch_legacy(legacy, exists=False)
        store = self.fake_store()

        config = Config()

        self.assertEqual(config.agent_chat_data_dir, self.root / "store-data")
        self.assertEqual(
            config.agent_chat_profile_dir, self.root / "store-config" / "profiles"
        )
        self.assertEqual(store.calls, ["data", "config"])

    def test_missing_localstore_falls_back_to_the_legacy_layout(self) -> None:
        legacy = self.root / "checkout" / "data" / "agent_chat"
        self.patch_legacy(legacy, exists=False)
        patched = mock.patch.dict(sys.modules, {"nonebot_plugin_localstore": None})
        patched.start()
        self.addCleanup(patched.stop)

        config = Config()

        self.assertEqual(config.agent_chat_data_dir, legacy)
        self.assertEqual(config.agent_chat_profile_dir, legacy / "profiles")

    def test_store_without_a_driver_falls_back_to_the_legacy_layout(self) -> None:
        legacy = self.root / "checkout" / "data" / "agent_chat"
        self.patch_legacy(legacy, exists=False)
        self.fake_store(RuntimeError("Cannot detect caller plugin"))

        config = Config()

        self.assertEqual(config.agent_chat_data_dir, legacy)
        self.assertEqual(config.agent_chat_profile_dir, legacy / "profiles")

    def test_explicit_settings_bypass_the_store(self) -> None:
        legacy = self.root / "checkout" / "data" / "agent_chat"
        self.patch_legacy(legacy, exists=False)
        store = self.fake_store()
        chosen = self.root / "chosen"

        config = Config(
            agent_chat_data_dir=chosen / "data",
            agent_chat_profile_dir=chosen / "profiles",
        )

        self.assertEqual(config.agent_chat_data_dir, chosen / "data")
        self.assertEqual(config.agent_chat_profile_dir, chosen / "profiles")
        self.assertEqual(store.calls, [])

    def test_standalone_import_keeps_working_with_the_store_installed(self) -> None:
        """``import nonebot_plugin_localstore`` needs a driver; the CLI has none."""

        code = """
from nonebot_plugin_agent_chat.config import Config

config = Config()
print(config.agent_chat_data_dir)
print(config.agent_chat_profile_dir)
"""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=directory,
                env={
                    "PATH": os.defpath,
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "HOME": directory,
                },
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(
            result.stdout.split(), ["data/agent_chat", "data/agent_chat/profiles"]
        )


if __name__ == "__main__":
    unittest.main()
