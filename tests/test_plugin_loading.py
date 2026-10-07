import os
import subprocess
import sys
import tempfile
import unittest


class PluginLoadingTests(unittest.TestCase):
    def test_adapter_warnings_are_filtered_only_during_standalone_import(self) -> None:
        code = """
import sys
import warnings
from unittest.mock import patch

import nonebot

initialized = sys.argv[1] == "bot"
if initialized:
    nonebot.init(driver="~none")

get_adapters = nonebot.get_adapters

def probe_adapters():
    warnings.warn("unrelated adapter warning", RuntimeWarning)
    return get_adapters()

expected = [
    "Failed to get nonebot adapters: NoneBot has not been initialized.",
    "No adapters found, please make sure you have installed at least one adapter.",
]
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    filters = list(warnings.filters)
    with patch("nonebot.get_adapters", side_effect=probe_adapters):
        if initialized:
            assert nonebot.load_plugin("nonebot_plugin_agent_chat") is not None
        else:
            import nonebot_plugin_agent_chat.cli

    messages = [str(item.message) for item in caught]
    assert "unrelated adapter warning" in messages, messages
    if initialized:
        assert expected[1] in messages, messages
    else:
        assert not any(message in messages for message in expected), messages
        try:
            nonebot.get_driver()
        except ValueError:
            pass
        else:
            raise AssertionError("standalone import initialized NoneBot")

    assert warnings.filters == filters, "import changed global warning filters"
    caught.clear()
    for message in expected:
        warnings.warn(message, RuntimeWarning)
    assert [str(item.message) for item in caught] == expected
"""
        for mode in ("standalone", "bot"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                result = subprocess.run(
                    [sys.executable, "-c", code, mode],
                    cwd=directory,
                    env={"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1"},
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_dependency_is_registered_before_scope_import(self) -> None:
        code = """
import nonebot

nonebot.init(driver="~none")
plugin = nonebot.load_plugin("nonebot_plugin_agent_chat")
assert plugin is not None, "agent-chat failed to load"
assert nonebot.get_plugin("nonebot_plugin_alconna") is not None
"""
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                [sys.executable, "-c", code],
                cwd=directory,
                env={"PATH": os.defpath, "PYTHONDONTWRITEBYTECODE": "1"},
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
