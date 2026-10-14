"""Storage directories come from nonebot-plugin-localstore."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from nonebot_plugin_agent_chat.paths import StoragePaths


class FakeStore(types.ModuleType):
    """Minimal stand-in for nonebot-plugin-localstore's directory helper."""

    def __init__(self, data_dir: Path, error: Exception | None):
        super().__init__("nonebot_plugin_localstore")
        self.calls: list[str] = []
        self._data_dir = data_dir
        self._error = error

    def get_plugin_data_dir(self) -> Path:
        self.calls.append("data")
        if self._error is not None:
            raise self._error
        return self._data_dir


class StoragePathTests(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def fake_store(self, error: Exception | None = None) -> FakeStore:
        store = FakeStore(self.root / "store-data", error)
        patched = mock.patch.dict(sys.modules, {"nonebot_plugin_localstore": store})
        patched.start()
        self.addCleanup(patched.stop)
        return store

    def test_localstore_data_dir_is_used_for_data_and_profiles(self) -> None:
        store = self.fake_store()

        paths = StoragePaths.resolve()
        expected_data = (self.root / "store-data").resolve()

        self.assertEqual(paths.data_dir, expected_data)
        self.assertEqual(paths.profile_dir, expected_data / "profiles")
        self.assertEqual(store.calls, ["data"])

    def test_explicit_overrides_skip_unneeded_store_calls(self) -> None:
        store = self.fake_store()
        custom_data = self.root / "custom-data"
        custom_profiles = self.root / "custom-profiles"

        only_data = StoragePaths.resolve(data_dir=custom_data)
        self.assertEqual(only_data.data_dir, custom_data.resolve())
        self.assertEqual(only_data.profile_dir, (custom_data / "profiles").resolve())
        self.assertEqual(store.calls, [])

        both = StoragePaths.resolve(data_dir=custom_data, profile_dir=custom_profiles)
        self.assertEqual(both.data_dir, custom_data.resolve())
        self.assertEqual(both.profile_dir, custom_profiles.resolve())
        self.assertEqual(store.calls, [])

        only_profiles = StoragePaths.resolve(profile_dir=custom_profiles)
        self.assertEqual(only_profiles.data_dir, (self.root / "store-data").resolve())
        self.assertEqual(only_profiles.profile_dir, custom_profiles.resolve())
        self.assertEqual(store.calls, ["data"])

    def test_localstore_error_is_not_silently_fallback(self) -> None:
        self.fake_store(RuntimeError("Cannot detect caller plugin"))

        with self.assertRaisesRegex(RuntimeError, "localstore"):
            StoragePaths.resolve()

    def test_missing_localstore_is_not_hidden(self) -> None:
        patched = mock.patch.dict(sys.modules, {"nonebot_plugin_localstore": None})
        patched.start()
        self.addCleanup(patched.stop)

        with self.assertRaisesRegex(RuntimeError, "localstore"):
            StoragePaths.resolve()

    def test_derived_paths_hang_off_the_resolved_data_dir(self) -> None:
        paths = StoragePaths(self.root / "rel" / ".." / "data")
        resolved_data = (self.root / "data").resolve()

        self.assertEqual(paths.data_dir, resolved_data)
        self.assertEqual(paths.profile_dir, resolved_data / "profiles")
        self.assertEqual(paths.database_path, resolved_data / "agent_chat.db")
        self.assertEqual(paths.prompt_dir, resolved_data / "prompts")
        self.assertEqual(paths.image_dir, resolved_data / "images")
        self.assertEqual(paths.debug_dir, resolved_data / "debug")

    def test_standalone_resolution_requires_a_driver_or_explicit_paths(self) -> None:
        code = """
from nonebot_plugin_agent_chat.paths import StoragePaths
StoragePaths.resolve()
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
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("localstore", result.stderr.lower())

    def test_cli_resolution_matches_the_bot(self) -> None:
        """The CLI and Bot must resolve identical base and per-plugin directories."""

        cli_code = """
from nonebot_plugin_agent_chat.paths import StoragePaths
paths = StoragePaths.resolve(initialize_driver=True)
print(paths.data_dir)
print(paths.profile_dir)
"""
        bot_code = """
import nonebot

nonebot.init(driver="~none")
nonebot.load_plugin("nonebot_plugin_agent_chat")
from nonebot_plugin_agent_chat.paths import StoragePaths

paths = StoragePaths.resolve()
print(paths.data_dir)
print(paths.profile_dir)
"""
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            base_data = root / "base-data"
            plugin_data = root / "plugin-data"
            cases = (
                (
                    "base",
                    {"LOCALSTORE_DATA_DIR": str(base_data)},
                    (
                        str(base_data / "nonebot_plugin_agent_chat"),
                        str(base_data / "nonebot_plugin_agent_chat" / "profiles"),
                    ),
                ),
                (
                    "plugin_override",
                    {
                        "LOCALSTORE_PLUGIN_DATA_DIR": json.dumps(
                            {"nonebot_plugin_agent_chat": str(plugin_data)}
                        ),
                    },
                    (
                        str(plugin_data),
                        str(plugin_data / "profiles"),
                    ),
                ),
            )
            for label, extra_env, expected in cases:
                with self.subTest(case=label):
                    env = {
                        "PATH": os.defpath,
                        "PYTHONDONTWRITEBYTECODE": "1",
                        "HOME": directory,
                        **extra_env,
                    }
                    runs = {
                        name: subprocess.run(
                            [sys.executable, "-c", code],
                            cwd=directory,
                            env=env,
                            capture_output=True,
                            text=True,
                            timeout=60,
                            check=False,
                        )
                        for name, code in (("cli", cli_code), ("bot", bot_code))
                    }
                    for name, result in runs.items():
                        self.assertEqual(
                            result.returncode,
                            0,
                            f"{name}: {result.stdout}{result.stderr}",
                        )
                    self.assertEqual(tuple(runs["cli"].stdout.split()), expected)
                    self.assertEqual(
                        [
                            line
                            for line in runs["bot"].stdout.splitlines()
                            if line.startswith("/")
                        ],
                        list(expected),
                    )


if __name__ == "__main__":
    unittest.main()
