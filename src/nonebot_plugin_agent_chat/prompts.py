from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .errors import ConfigurationError

DEFAULT_PROMPT_FILE = "default.md"

BUILTIN_DEFAULT_PROMPT = (
    "You are a helpful and reliable assistant.\n"
    "Answer the user's request clearly and accurately.\n"
    "If you are uncertain, say so instead of inventing facts.\n"
    "When tools or search results provide sources, cite them."
)

SOURCE_BUILTIN = "builtin"
SOURCE_GLOBAL = "global"
SOURCE_PROFILE_INLINE = "profile-inline"
SOURCE_PROFILE_FILE = "profile-file"


@dataclass(frozen=True)
class ResolvedPrompt:
    """One immutable system-prompt decision taken while staging profiles."""

    text: str
    source: str
    label: str
    digest: str

    @classmethod
    def create(cls, text: str, source: str, label: str) -> ResolvedPrompt:
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        return cls(text=text, source=source, label=label, digest=digest)

    @property
    def summary(self) -> str:
        """Metadata for logs and superuser status; never the prompt text."""

        if not self.text:
            return f"{self.label} (empty)"
        return f"{self.label} sha256={self.digest[:12]}"


def normalize_prompt_text(raw: bytes, label: str) -> str:
    """Decode a prompt file: UTF-8, BOM stripped, CRLF/CR to LF, blank edges trimmed."""

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ConfigurationError(f"{label}: prompt file is not valid UTF-8") from exc
    text = text.lstrip("\ufeff").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    while lines and not lines[0].strip():
        lines.pop(0)
    while lines and not lines[-1].strip():
        lines.pop()
    return "\n".join(lines)


class PromptRegistry:
    """Reads prompt files confined to one directory; holds no mutable state."""

    def __init__(self, root: Path, default_file: str = DEFAULT_PROMPT_FILE) -> None:
        self.root = Path(root)
        self.default_file = default_file.strip() or DEFAULT_PROMPT_FILE

    def confined_path(self, name: str, label: str) -> Path:
        """Resolve a prompt name inside the prompt root, rejecting escapes."""

        candidate = name.strip()
        if not candidate:
            raise ConfigurationError(f"{label}: prompt file name is empty")
        path = Path(candidate)
        if path.is_absolute():
            raise ConfigurationError(
                f"{label}: prompt file must be relative, got {candidate}"
            )
        if ".." in path.parts:
            raise ConfigurationError(
                f"{label}: prompt file must not contain '..', got {candidate}"
            )
        try:
            root = self.root.resolve()
            resolved = (root / path).resolve()
        except OSError as exc:
            raise ConfigurationError(
                f"{label}: cannot resolve prompt path {candidate}"
            ) from exc
        if resolved != root and root not in resolved.parents:
            raise ConfigurationError(
                f"{label}: prompt file escapes the prompt directory: {candidate}"
            )
        return resolved

    def read(self, path: Path, label: str) -> str | None:
        """Return normalized prompt text, or None when the file does not exist."""

        try:
            exists = path.exists()
        except OSError as exc:
            raise ConfigurationError(
                f"{label}: cannot inspect prompt file {path.name}"
            ) from exc
        if not exists:
            return None
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise ConfigurationError(
                f"{label}: cannot read prompt file {path.name} ({type(exc).__name__})"
            ) from exc
        return normalize_prompt_text(raw, label)

    def global_path(self) -> Path:
        return self.confined_path(self.default_file, "global prompt")

    def resolve_global(self) -> ResolvedPrompt:
        """Resolve the global default; a missing file legitimately means builtin."""

        path = self.global_path()
        text = self.read(path, f"global prompt {self.default_file}")
        if text is None:
            return ResolvedPrompt.create(
                BUILTIN_DEFAULT_PROMPT,
                SOURCE_BUILTIN,
                f"builtin (no {self.default_file})",
            )
        return ResolvedPrompt.create(
            text,
            SOURCE_GLOBAL,
            f"global:{self.default_file}",
        )

    def resolve_profile(
        self,
        profile: Any,
        global_prompt: ResolvedPrompt,
    ) -> ResolvedPrompt:
        """Apply inline > profile file > global precedence for one profile."""

        inline = str(getattr(profile, "system_prompt", "") or "").strip()
        if inline:
            return ResolvedPrompt.create(
                inline,
                SOURCE_PROFILE_INLINE,
                "profile:inline",
            )
        name = getattr(profile, "system_prompt_file", None)
        if name and name.strip():
            label = f"profile:file:{name.strip()}"
            path = self.confined_path(name, name.strip())
            text = self.read(path, label)
            if text is None:
                raise ConfigurationError(
                    f"{label}: prompt file not found in {self.root}"
                )
            return ResolvedPrompt.create(text, SOURCE_PROFILE_FILE, label)
        return global_prompt
