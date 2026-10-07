import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from dotenv import dotenv_values

from nonebot_plugin_agent_chat import env_file


class TextEditingTests(unittest.TestCase):
    def test_set_keeps_comments_order_and_neighbours(self) -> None:
        text = (
            "# header comment\n"
            "AGENT_CHAT_MAX_SEARCHES=3\n"
            "\n"
            "AGENT_CHAT_DEFAULT_PROFILE=default  # inline\n"
        )
        updated = env_file.set_value(text, "AGENT_CHAT_MAX_SEARCHES", "5")
        self.assertIn("# header comment", updated)
        self.assertIn("AGENT_CHAT_MAX_SEARCHES=5", updated)
        self.assertNotIn("AGENT_CHAT_MAX_SEARCHES=3", updated)
        # The untouched line keeps its own inline comment.
        self.assertIn("AGENT_CHAT_DEFAULT_PROFILE=default  # inline", updated)

    def test_set_appends_missing_keys(self) -> None:
        updated = env_file.set_value("A=1\n", "B", "2")
        self.assertEqual(updated, "A=1\nB=2\n")

    def test_values_round_trip_through_dotenv(self) -> None:
        for value in (
            "3",
            "true",
            '["a", "b"]',
            '{"Telegram": "off"}',
            "with space",
            "has'quote",
            "",
        ):
            with self.subTest(value=value):
                text = env_file.set_value("", "AGENT_CHAT_X", value)
                with tempfile.TemporaryDirectory() as tmp:
                    path = Path(tmp) / ".env"
                    path.write_text(text, encoding="utf-8")
                    parsed = dotenv_values(path)
                self.assertEqual(parsed["AGENT_CHAT_X"], value)

    def test_unset_removes_only_the_target_line(self) -> None:
        text = "A=1\nB=2\n"
        updated, removed = env_file.unset_value(text, "A")
        self.assertTrue(removed)
        self.assertEqual(updated, "B=2\n")
        again, removed_again = env_file.unset_value(updated, "A")
        self.assertFalse(removed_again)
        self.assertEqual(again, updated)

    def test_write_atomic_replaces_and_keeps_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / ".env"
            path.write_text("old\n", encoding="utf-8")
            path.chmod(0o600)
            env_file.write_atomic(path, "new\n")
            self.assertEqual(path.read_text(encoding="utf-8"), "new\n")
            self.assertEqual(path.stat().st_mode & 0o777, 0o600)
            leftovers = list(Path(tmp).iterdir())
            self.assertEqual(leftovers, [path])


class EnvironRefreshTests(unittest.TestCase):
    def _env(self, path: Path, content: str) -> Path:
        path.write_text(content, encoding="utf-8")
        return path

    def test_initial_load_tracks_file_owned_keys_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._env(Path(tmp) / ".env", "A=file\nB=file\n")
            with patch.dict(os.environ, {"B": "real"}, clear=True):
                owned = env_file.load_into_environ(path)
                self.assertEqual(owned, {"A"})
                self.assertEqual(os.environ["A"], "file")
                self.assertEqual(os.environ["B"], "real")

    def test_stage_updates_removes_and_adds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._env(Path(tmp) / ".env", "A=1\nB=2\n")
            with patch.dict(os.environ, {}, clear=True):
                owned = env_file.load_into_environ(path)
                self.assertEqual(owned, {"A", "B"})
                path.write_text("A=9\nC=3\n", encoding="utf-8")
                staged = env_file.stage_environ(path, owned)
                env_file.commit_environ(staged, owned)
                self.assertEqual(set(staged.owned), {"A", "C"})
                self.assertEqual(os.environ["A"], "9")
                self.assertNotIn("B", os.environ)
                self.assertEqual(os.environ["C"], "3")

    def test_stage_never_overwrites_real_environment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._env(Path(tmp) / ".env", "A=file\n")
            with patch.dict(os.environ, {"A": "real"}, clear=True):
                owned: set[str] = set()
                env_file.load_into_environ(path)
                staged = env_file.stage_environ(path, owned)
                env_file.commit_environ(staged, owned)
                self.assertEqual(os.environ["A"], "real")
                self.assertEqual(set(staged.owned), set())

    def test_staging_is_pure_until_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = self._env(Path(tmp) / ".env", "A=file\nB=file\n")
            with patch.dict(os.environ, {"B": "real"}, clear=True):
                first = env_file.stage_environ(path, set())
                # Staging must not change the live environment at all.
                self.assertNotIn("A", os.environ)
                env_file.commit_environ(first, set())
                self.assertEqual(os.environ["A"], "file")
                self.assertEqual(set(first.owned), {"A"})

                path.write_text("A=9\nC=3\n", encoding="utf-8")
                staged = env_file.stage_environ(path, set(first.owned))
                # Staging the edit still leaves the live value alone.
                self.assertEqual(os.environ["A"], "file")
                self.assertEqual(staged.values["A"], "9")
                self.assertEqual(staged.values["C"], "3")

                env_file.commit_environ(staged, set(first.owned))
                self.assertEqual(os.environ["A"], "9")
                self.assertEqual(os.environ["C"], "3")
                self.assertEqual(os.environ["B"], "real")

                path.write_text("", encoding="utf-8")
                empty = env_file.stage_environ(path, set(staged.owned))
                env_file.commit_environ(empty, set(staged.owned))
                self.assertNotIn("A", os.environ)
                self.assertNotIn("C", os.environ)
                self.assertEqual(os.environ["B"], "real")

    def test_write_many_atomic_replaces_every_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            first = root / "a.env"
            second = root / "b.env"
            first.write_text("old-a\n", encoding="utf-8")
            second.write_text("old-b\n", encoding="utf-8")

            env_file.write_many_atomic({first: "new-a\n", second: "new-b\n"})

            self.assertEqual(first.read_text(encoding="utf-8"), "new-a\n")
            self.assertEqual(second.read_text(encoding="utf-8"), "new-b\n")
            # No temp files survive a successful batch.
            self.assertEqual(
                sorted(path.name for path in root.iterdir()),
                ["a.env", "b.env"],
            )

    def test_missing_file_reads_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "none"
            self.assertEqual(env_file.read_values(missing), {})
            self.assertEqual(env_file.load_into_environ(missing), set())


if __name__ == "__main__":
    unittest.main()
