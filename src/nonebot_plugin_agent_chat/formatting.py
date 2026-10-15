from __future__ import annotations

from .models import Source


def append_sources(text: str, sources: list[Source], limit: int = 10) -> str:
    if not sources:
        return text
    lines = [text.rstrip(), "", "来源："]
    for index, source in enumerate(sources[:limit], 1):
        url = "".join(source.url.splitlines()).strip()
        label = " ".join(source.title.split())[:200] or url
        lines.append(f"[{index}] {label} - {url}")
    return "\n".join(lines).strip()


def format_token_count(tokens: int) -> str:
    if tokens >= 1000:
        return f"{tokens / 1000:.1f}k"
    return str(max(0, tokens))
