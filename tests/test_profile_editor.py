import json
import tempfile
import unittest
from pathlib import Path

from nonebot_plugin_agent_chat import profile_editor
from nonebot_plugin_agent_chat.errors import InputError

BASE = {"protocol": "openai-responses", "model": "test"}


def set_field(path: Path, field: str, token: str) -> None:
    """One atomic field edit, the shape the CLI and editor both use."""

    transaction = profile_editor.ProfileFileTransaction(path.parent)
    transaction.set_field(path.stem, field, token)
    transaction.commit()


def unset_field(path: Path, field: str) -> bool:
    transaction = profile_editor.ProfileFileTransaction(path.parent)
    removed = transaction.unset_field(path.stem, field)
    if removed:
        transaction.commit()
    return removed


def create_profile(directory: Path, name: str, template: str) -> Path:
    transaction = profile_editor.ProfileFileTransaction(directory)
    transaction.create(name, template)
    transaction.commit()
    return profile_editor.profile_path(directory, name)


def delete_profile(directory: Path, name: str) -> None:
    transaction = profile_editor.ProfileFileTransaction(directory)
    transaction.delete(name)
    transaction.commit()


class ProfileEditorTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.directory = Path(self.temp.name)
        self.path = self.directory / "primary.json"
        self.path.write_text(json.dumps(BASE), encoding="utf-8")

    def tearDown(self) -> None:
        self.temp.cleanup()

    def test_list_and_describe_fields(self) -> None:
        self.assertEqual(profile_editor.list_profiles(self.directory), ["primary"])
        fields = {
            field.name: field for field in profile_editor.describe_fields(self.path)
        }
        self.assertEqual(fields["model"].value, "test")
        self.assertEqual(fields["model"].kind, "text")
        self.assertEqual(fields["model"].value, "test")
        # Fields the file never mentioned show the default marker.
        self.assertEqual(fields["search_mode"].value, "(默认)")
        self.assertEqual(fields["search_mode"].kind, "enum")
        self.assertIn("off", fields["search_mode"].options)
        self.assertEqual(fields["show_sources_text"].kind, "bool")

    def test_sensitive_values_are_masked(self) -> None:
        path = self.directory / "masked.json"
        path.write_text(
            json.dumps({**BASE, "extra_headers": {"Authorization": "Bearer x"}}),
            encoding="utf-8",
        )
        fields = {field.name: field for field in profile_editor.describe_fields(path)}
        self.assertEqual(fields["extra_headers"].value, "•••")

    def test_saved_maps_use_the_same_mask_as_staged_values(self) -> None:
        canary = "fixture-private-value"
        values = {
            "extra_headers": {"Cookie": f"session={canary}"},
            "extra_query": {"credentials": {"session": canary}},
            "extra_body": {"metadata": {"auth": canary}},
        }
        self.path.write_text(json.dumps({**BASE, **values}), encoding="utf-8")
        fields = {
            field.name: field for field in profile_editor.describe_fields(self.path)
        }
        for name, value in values.items():
            with self.subTest(field=name):
                self.assertEqual(fields[name].value, "•••")
                self.assertEqual(
                    fields[name].value,
                    profile_editor.display_token(name, json.dumps(value)),
                )
        self.assertEqual(json.loads(self.path.read_text()), {**BASE, **values})

    def test_set_field_writes_and_validates(self) -> None:
        set_field(self.path, "model", "other")
        set_field(self.path, "show_sources_text", "false")
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertEqual(raw["model"], "other")
        self.assertIs(raw["show_sources_text"], False)

    def test_set_field_rejects_unknown_and_invalid(self) -> None:
        with self.assertRaises(InputError):
            set_field(self.path, "nope", "1")
        with self.assertRaises(InputError):
            # model must be a non-empty string (empty token = empty value)
            set_field(self.path, "model", "")
        with self.assertRaises(InputError):
            set_field(self.path, "image_reply_mode", "sometimes")

    def test_set_field_keeps_unrelated_keys(self) -> None:
        path = self.directory / "extra.json"
        path.write_text(
            json.dumps({**BASE, "max_output_tokens": 1234}), encoding="utf-8"
        )
        set_field(path, "model", "kept")
        raw = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(raw["max_output_tokens"], 1234)

    def test_unset_field_falls_back_to_default(self) -> None:
        set_field(self.path, "temperature", "0.5")
        self.assertTrue(unset_field(self.path, "temperature"))
        self.assertFalse(unset_field(self.path, "temperature"))
        raw = json.loads(self.path.read_text(encoding="utf-8"))
        self.assertNotIn("temperature", raw)

    def test_unset_required_field_is_rejected(self) -> None:
        with self.assertRaises(InputError):
            unset_field(self.path, "model")

    def test_create_from_template_and_delete(self) -> None:
        created = create_profile(self.directory, "qq-safe", "primary")
        self.assertEqual(created.name, "qq-safe.json")
        self.assertEqual(json.loads(created.read_text(encoding="utf-8")), BASE)
        with self.assertRaises(InputError):
            create_profile(self.directory, "qq-safe", "primary")
        with self.assertRaises(InputError):
            create_profile(self.directory, "bad name", "primary")
        with self.assertRaises(InputError):
            create_profile(self.directory, "x", "missing")

        delete_profile(self.directory, "qq-safe")
        self.assertFalse(created.exists())
        with self.assertRaises(InputError):
            delete_profile(self.directory, "qq-safe")

    def test_delete_cancels_a_create_staged_in_the_same_batch(self) -> None:
        """`--profile-new X --from Y --profile-remove X` must net out to nothing."""

        transaction = profile_editor.ProfileFileTransaction(self.directory)
        transaction.create("qq-safe", "primary")
        transaction.delete("qq-safe")

        transaction.commit()

        self.assertFalse((self.directory / "qq-safe.json").exists())
        self.assertEqual(transaction.files, {})
        self.assertEqual(transaction.deletions, set())

    def test_reference_warnings(self) -> None:
        warnings = profile_editor.reference_warnings(
            "primary",
            rule_labels=["QQClient:group:1"],
            room_names=["demo"],
            configured_default="primary",
        )
        self.assertEqual(len(warnings), 3)
        self.assertTrue(any("规则" in line for line in warnings))
        self.assertTrue(any("Room" in line for line in warnings))
        self.assertTrue(any("默认" in line for line in warnings))
        self.assertEqual(profile_editor.reference_warnings("primary"), [])


if __name__ == "__main__":
    unittest.main()
