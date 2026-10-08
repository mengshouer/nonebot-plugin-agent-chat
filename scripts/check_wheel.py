from __future__ import annotations

import re
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path


def fail(message: str) -> None:
    raise SystemExit(f"wheel check failed: {message}")


# `[project] version = "..."` is one top-level line; a regex keeps this script
# runnable on Python 3.10, where tomllib does not exist.
_VERSION = re.compile(r'^version\s*=\s*"([^"]+)"', re.MULTILINE)


def declared_version(repository: Path) -> str:
    """The version from pyproject.toml, so a release is a one-place edit."""

    text = (repository / "pyproject.toml").read_text(encoding="utf-8")
    matched = _VERSION.search(text)
    if matched is None:
        fail("pyproject.toml has no project version")
    return matched.group(1)


def main() -> None:
    if len(sys.argv) != 2:
        fail("usage: check_wheel.py <wheel>")
    wheel = Path(sys.argv[1])
    if not wheel.is_file():
        fail(f"not found: {wheel}")

    key_pattern = re.compile(rb"sk-[A-Za-z0-9_-]{20,}")
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        required = {
            "nonebot_plugin_agent_chat/__init__.py",
            "nonebot_plugin_agent_chat/__main__.py",
            "nonebot_plugin_agent_chat/py.typed",
            "nonebot_plugin_agent_chat/profiles.example/openai-chat.json",
            "nonebot_plugin_agent_chat/profiles.example/openai-responses.json",
            "nonebot_plugin_agent_chat/profiles.example/anthropic-exa.json",
            "nonebot_plugin_agent_chat/profiles.example/anthropic-web-search.json",
            "nonebot_plugin_agent_chat/prompts.example/default.md",
            "nonebot_plugin_agent_chat/env.agent_chat.example",
        }
        missing = sorted(required - names)
        if missing:
            fail(f"missing package data: {missing}")
        if any(name.startswith("tests/") for name in names):
            fail("tests must not be shipped in the wheel")

        metadata_names = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        entry_names = [
            name for name in names if name.endswith(".dist-info/entry_points.txt")
        ]
        if len(metadata_names) != 1 or len(entry_names) != 1:
            fail("wheel metadata or console entry point is missing")
        metadata = BytesParser().parsebytes(archive.read(metadata_names[0]))
        entries = archive.read(entry_names[0]).decode("utf-8")
        if metadata["Metadata-Version"] != "2.4":
            fail("unexpected core metadata version")
        if metadata["Name"] != "nonebot-plugin-agent-chat":
            fail("unexpected distribution name")
        if metadata["Version"] != declared_version(Path(__file__).resolve().parents[1]):
            fail("wheel version does not match pyproject.toml")
        requires_python = metadata["Requires-Python"] or ""
        if ">=3.10" not in requires_python or "<4.0" not in requires_python:
            fail("unexpected Python requirement")
        requires_dist = metadata.get_all("Requires-Dist") or []
        if not any(item.startswith("nonebot-plugin-alconna") for item in requires_dist):
            fail("nonebot-plugin-alconna is not a declared dependency")
        if not any(item.startswith("nonebot-adapter-onebot") for item in requires_dist):
            fail("the onebot-v11 adapter extra is missing")
        if not any(
            item.startswith("nonebot-adapter-telegram") for item in requires_dist
        ):
            fail("the telegram adapter extra is missing")
        if "nonebot-agent-chat = nonebot_plugin_agent_chat.cli:main" not in entries:
            fail("console script is missing")

        for name in sorted(names):
            if name.endswith("/"):
                continue
            data = archive.read(name)
            if key_pattern.search(data):
                fail(f"key-shaped literal in {name}")

    print(f"wheel content check: ok ({wheel.name})")


if __name__ == "__main__":
    main()
