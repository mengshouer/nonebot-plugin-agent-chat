import ast
import inspect
import unittest
from pathlib import Path

from nonebot_plugin_agent_chat.service import AgentChatService

SOURCES = {
    name: Path(__file__).resolve().parents[1]
    / f"src/nonebot_plugin_agent_chat/{name}.py"
    for name in ("matchers", "cli")
}


class MatcherContractTests(unittest.TestCase):
    def test_pending_reload_applies_before_routing(self) -> None:
        """A reloaded config must govern the message that triggered the reload."""

        tree = ast.parse(SOURCES["matchers"].read_text(encoding="utf-8"))
        function = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.AsyncFunctionDef) and node.name == "_should_trigger"
        )
        first = function.body[0]
        self.assertIsInstance(first, ast.Expr)
        assert isinstance(first, ast.Expr)
        self.assertIsInstance(first.value, ast.Await)
        assert isinstance(first.value, ast.Await)
        call = first.value.value
        self.assertIsInstance(call, ast.Call)
        assert isinstance(call, ast.Call)
        self.assertIsInstance(call.func, ast.Name)
        assert isinstance(call.func, ast.Name)
        self.assertEqual(call.func.id, "_adopt_pending_reload")

    def test_matchers_do_not_infer_scopes_from_adapter_names(self) -> None:
        source = SOURCES["matchers"].read_text(encoding="utf-8")
        for removed in (
            "_warn_about_unknown_scopes",
            "_block_ambiguous_adapters",
            "ambiguous_fallback_scopes",
            "normalized_scope",
        ):
            self.assertNotIn(removed, source)

    def test_no_unsolicited_processing_notice(self) -> None:
        source = SOURCES["matchers"].read_text(encoding="utf-8")
        self.assertNotIn("正在处理，请稍候", source)

    def test_nonebot_logger_uses_loguru_placeholders(self) -> None:
        source = SOURCES["matchers"].read_text(encoding="utf-8")
        self.assertIn("logger.", source)
        self.assertNotIn("%s", source)

    def test_service_calls_only_use_declared_keywords(self) -> None:
        """Matchers/CLI must not pass keywords the service does not accept.

        Guards against regressions like an ``ask_room_deferred`` call growing a
        ``conversation=`` argument that lives on ``ask_deferred`` only. The
        walker only sees direct ``service.X(...)`` calls; attribute chains such
        as ``service.profiles.Y(...)`` are intentionally out of scope (they
        target the registry, not the service).
        """

        problems = []
        for name, path in SOURCES.items():
            calls = [
                node
                for node in ast.walk(ast.parse(path.read_text(encoding="utf-8")))
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "service"
            ]
            self.assertTrue(calls, f"no service calls found in {name}.py")
            for call in calls:
                method = getattr(AgentChatService, call.func.attr, None)
                if method is None:
                    problems.append(f"{name}: unknown method: service.{call.func.attr}")
                    continue
                accepted = inspect.signature(method).parameters
                for keyword in call.keywords:
                    if keyword.arg not in accepted:
                        problems.append(
                            f"{name}: service.{call.func.attr}(...) got "
                            f"{keyword.arg}=, accepted: {sorted(accepted)}"
                        )
        self.assertEqual(problems, [])


class ProfileFieldContractTests(unittest.TestCase):
    """The getattr field names in answer_options must be real profile fields."""

    def test_profile_field_names_exist(self) -> None:
        import re

        from nonebot_plugin_agent_chat.models import ProviderProfile

        source = (
            Path(__file__).resolve().parents[1]
            / "src/nonebot_plugin_agent_chat/answer_options.py"
        ).read_text(encoding="utf-8")
        used = set(re.findall(r'profile_field\([^)]*"([a-z_]+)"\)', source))
        self.assertTrue(used)
        fields = set(ProviderProfile.model_fields)
        missing = used - fields
        self.assertFalse(missing)

    def test_source_switches_gate_the_right_payloads(self) -> None:
        """The image switch gates render sources; the text switch gates text.

        The wiring lives in matchers.py, which the suite cannot import, so it
        is pinned through the single deliver_answer(...) call: a swapped pair
        or an empty-list fallback must fail this test.
        """

        tree = ast.parse(SOURCES["matchers"].read_text(encoding="utf-8"))
        call = next(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "deliver_answer"
                and any(kw.arg == "fallback_text" for kw in node.keywords)
            ),
            None,
        )
        self.assertIsNotNone(call)
        assert call is not None
        keywords = {kw.arg: kw.value for kw in call.keywords}

        sources = keywords["sources"]
        self.assertIsInstance(sources, ast.IfExp)
        self.assertIn("show_image_sources", ast.dump(sources.test))
        self.assertNotIn("show_text_sources", ast.dump(sources))
        self.assertIsInstance(sources.orelse, ast.Constant)
        self.assertIsNone(sources.orelse.value)  # None, not []: whole footer off

        fallback = keywords["fallback_text"]
        self.assertIsInstance(fallback, ast.IfExp)
        self.assertIn("show_text_sources", ast.dump(fallback.test))
        self.assertNotIn("show_image_sources", ast.dump(fallback))


if __name__ == "__main__":
    unittest.main()
