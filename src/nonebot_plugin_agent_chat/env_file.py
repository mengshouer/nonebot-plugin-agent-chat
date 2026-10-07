"""The plugin's dotenv file: text-level editing and environment refreshes.

Two layers live here:

- text editing that keeps comments, ordering, and every other line untouched
  (used by the CLI editors, which validate before calling it);
- environment bookkeeping that only ever touches keys the file itself provided
  (used by reloads), so a real environment variable always wins.
"""

from __future__ import annotations

import io
import logging
import os
import tempfile
from collections.abc import Mapping
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from dotenv import dotenv_values

logger = logging.getLogger(__name__)

_SAFE_BARE = frozenset(" \t\"'#")


def read_values(path: Path) -> dict[str, str]:
    """Effective key/value pairs of the file, layout ignored."""

    if not path.is_file():
        return {}
    try:
        values = dotenv_values(path)
    except OSError as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return {}
    return {key: value for key, value in values.items() if value is not None}


@dataclass(frozen=True)
class EnvironmentSnapshot:
    """A candidate process environment staged without changing global state."""

    values: dict[str, str]
    owned: frozenset[str]


def stage_values(
    values: Mapping[str, str],
    owned: set[str],
) -> EnvironmentSnapshot:
    """Merge dotenv values into an environment copy.

    Keys already supplied by the real environment are never adopted. Keys that
    the dotenv file owned on the previous load follow the file, including being
    removed when their line disappears. This is the pure half of reload: callers
    can validate a complete candidate before committing it to ``os.environ``.
    """

    candidate = dict(os.environ)
    previous_owned = set(owned)
    next_owned = previous_owned & set(values)
    for key in previous_owned - set(values):
        candidate.pop(key, None)
    for key, value in values.items():
        if key in previous_owned or key not in candidate:
            candidate[key] = value
            next_owned.add(key)
    return EnvironmentSnapshot(candidate, frozenset(next_owned))


def stage_environ(path: Path, owned: set[str]) -> EnvironmentSnapshot:
    """Stage the current dotenv file without mutating process environment."""

    return stage_values(read_values(path), owned)


def commit_environ(snapshot: EnvironmentSnapshot, previous_owned: set[str]) -> None:
    """Publish a previously staged environment snapshot."""

    for key in set(previous_owned) - set(snapshot.owned):
        os.environ.pop(key, None)
    for key in snapshot.owned:
        os.environ[key] = snapshot.values[key]


def load_into_environ(path: Path) -> set[str]:
    """Load the file once and return the keys the file now owns."""

    snapshot = stage_environ(path, set())
    commit_environ(snapshot, set())
    return set(snapshot.owned)


def write_many_atomic(files: Mapping[Path, str]) -> None:
    """Replace several files, staging every new content before any replace.

    Each file keeps its permissions and is replaced atomically. A failure while
    staging leaves every target untouched, so a batch edit is either fully
    written or not written at all.
    """

    staged: list[tuple[Path, Path]] = []
    try:
        for path, text in files.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            mode: int | None = path.stat().st_mode if path.exists() else None
            handle_fd, tmp_name = tempfile.mkstemp(
                dir=path.parent, prefix=f".{path.name}.", suffix=".tmp"
            )
            tmp_path = Path(tmp_name)
            staged.append((tmp_path, path))
            with os.fdopen(handle_fd, "w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            if mode is not None:
                tmp_path.chmod(mode)
        for tmp_path, path in staged:
            os.replace(tmp_path, path)
    except BaseException:
        for tmp_path, _ in staged:
            with suppress(FileNotFoundError):
                tmp_path.unlink()
        raise


def parse_text(text: str) -> dict[str, str]:
    """Parse dotenv text without touching the filesystem."""

    values = dotenv_values(stream=io.StringIO(text))
    return {key: value for key, value in values.items() if value is not None}


def format_value(value: str) -> str:
    """Quote a value the way python-dotenv reads it back verbatim."""

    if value and not (set(value) & _SAFE_BARE) and not value.startswith(("$", "`")):
        return value
    if "'" not in value:
        return f"'{value}'"
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _find_line(lines: list[str], key: str) -> int | None:
    for index, line in enumerate(lines):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        name, separator, _ = stripped.partition("=")
        if separator and name.strip() == key:
            return index
    return None


def set_value(text: str, key: str, value: str) -> str:
    """Return ``text`` with ``KEY=VALUE`` set, comments and order kept."""

    line = f"{key}={format_value(value)}"
    lines = text.splitlines()
    index = _find_line(lines, key)
    if index is None:
        lines.append(line)
    else:
        lines[index] = line
    return "\n".join(lines) + "\n"


def unset_value(text: str, key: str) -> tuple[str, bool]:
    """Return ``text`` without ``KEY``; the flag reports whether it was there."""

    lines = text.splitlines()
    index = _find_line(lines, key)
    if index is None:
        return text, False
    del lines[index]
    return "\n".join(lines) + ("\n" if lines else ""), True


def write_atomic(path: Path, text: str) -> None:
    """Replace one file atomically, keeping its permissions when it exists."""

    write_many_atomic({path: text})
