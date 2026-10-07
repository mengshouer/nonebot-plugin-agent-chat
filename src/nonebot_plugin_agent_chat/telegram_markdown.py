"""Convert Markdown answers into the HTML subset Telegram renders.

Telegram does not parse Markdown on its own: text sent without ``parse_mode``
arrives as literal characters. This module converts one answer (or one delivery
chunk, since long answers are split before sending) into balanced Telegram HTML:
headings become bold lines, lists get bullets, fenced code becomes ``<pre>`` when
the fence closes inside the chunk (otherwise per-line ``<code>``), pipe tables
become ``<pre>`` so columns stay aligned, and inline markup becomes
bold/italic/strikethrough/code/spoiler/link tags.

Every function is stateless and always emits balanced tags, because the
delivery layer may retry or re-split a chunk at any time.
"""

from __future__ import annotations

import re
from html import escape

_FENCE = re.compile(r"^\s*```")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEPARATOR = re.compile(r"^\s*\|[\s:\-|]+\|\s*$")
_HEADING = re.compile(r"^#{1,6}\s+(.*)$")
_BLOCKQUOTE = re.compile(r"^>\s?(.*)$")
_BULLET = re.compile(r"^(\s*)[-*+]\s+")
_SAFE_LINK = re.compile(r"^(?:https?://|tg://|mailto:)", re.IGNORECASE)

# One pass over an escaped line: a construct is never re-scanned, so markup
# inside inline code stays literal and emitted tags are never re-interpreted.
_INLINE = re.compile(
    r"`(?P<code>[^`\n]+)`"
    r"|\*\*(?P<bold>[^*\n]+)\*\*"
    r"|__(?P<bold_underscore>[^_\n]+)__"
    r"|~~(?P<strike>[^~\n]+)~~"
    r"|\|\|(?P<spoiler>[^|\n]+)\|\|"
    r"|\[(?P<label>[^\]\n]+)\]\((?P<href>[^)\s\"]+)\)"
    r"|(?<![\w*])\*(?P<italic>[^*\n]+)\*(?![\w*])"
    r"|(?<!\w)_(?P<italic_underscore>[^_\n]+)_(?!\w)"
)

_WRAPPERS = {
    "code": ("<code>", "</code>"),
    "bold": ("<b>", "</b>"),
    "bold_underscore": ("<b>", "</b>"),
    "strike": ("<s>", "</s>"),
    "spoiler": ('<span class="tg-spoiler">', "</span>"),
    "italic": ("<i>", "</i>"),
    "italic_underscore": ("<i>", "</i>"),
}


def _replace_inline(match: re.Match[str]) -> str:
    kind = match.lastgroup or ""
    value = match.group(kind)
    if kind in _WRAPPERS:
        opening, closing = _WRAPPERS[kind]
        return f"{opening}{value}{closing}"
    if kind == "href":
        # The whole match was escaped before matching, so an unsafe scheme is
        # rendered as its literal text instead of a link.
        if not _SAFE_LINK.match(value):
            return match.group(0)
        return f'<a href="{value}">{match.group("label")}</a>'
    return match.group(0)


def _inline(line: str) -> str:
    return _INLINE.sub(_replace_inline, escape(line, quote=False))


def _code_line(line: str) -> str:
    escaped = escape(line, quote=False)
    return f"<code>{escaped}</code>" if escaped else ""


def _table(lines: list[str], index: int) -> tuple[str, int] | None:
    """Return one ``<pre>`` table starting at ``index``, or ``None``."""

    if not _TABLE_ROW.match(lines[index]):
        return None
    if index + 1 >= len(lines) or not _TABLE_SEPARATOR.match(lines[index + 1]):
        return None
    rows: list[str] = []
    while index < len(lines) and _TABLE_ROW.match(lines[index]):
        rows.append(lines[index].strip())
        index += 1
    body = "\n".join(escape(row, quote=False) for row in rows)
    return f"<pre>{body}</pre>", index


def _heading_text(content: str) -> str:
    """Inline-convert heading text with heading-specific rules.

    A heading is already bold, so an edge ``**`` pair is a redundant wrapper,
    and ``__init__``-style identifiers stay literal instead of being read as
    underscore emphasis. Single-underscore italics still work, matching the
    body-text rules.
    """

    text = content.strip()
    if len(text) > 4 and text.startswith("**") and text.endswith("**"):
        text = text[2:-2].strip()

    stashed: list[str] = []

    def stash(match: re.Match[str]) -> str:
        stashed.append(escape(match.group(0), quote=False))
        return f"\x00{len(stashed) - 1}\x00"

    text = re.sub(r"__[^_\n]+__", stash, text)
    converted = _inline(text)
    return re.sub(r"\x00(\d+)\x00", lambda m: stashed[int(m.group(1))], converted)


def markdown_to_telegram_html(text: str) -> str:
    """Convert one Markdown chunk to balanced Telegram HTML."""

    if not text:
        return ""
    lines = text.split("\n")
    output: list[str] = []
    index = 0
    while index < len(lines):
        line = lines[index]

        if _FENCE.match(line):
            index += 1
            code: list[str] = []
            closed = False
            while index < len(lines):
                if _FENCE.match(lines[index]):
                    closed = True
                    index += 1
                    break
                code.append(lines[index])
                index += 1
            if closed:
                body = "\n".join(escape(part, quote=False) for part in code)
                output.append(f"<pre>{body}</pre>")
            else:
                # The fence continues in the next chunk; per-line code keeps
                # every message balanced without carrying state across chunks.
                output.append("\n".join(_code_line(part) for part in code))
            continue

        table = _table(lines, index)
        if table is not None:
            rendered, index = table
            output.append(rendered)
            continue

        heading = _HEADING.match(line)
        if heading:
            content = _heading_text(heading.group(1))
            # The heading itself is bold, so drop nested bold tags.
            content = content.replace("<b>", "").replace("</b>", "")
            output.append(f"<b>{content}</b>")
            index += 1
            continue

        quote = _BLOCKQUOTE.match(line)
        if quote:
            output.append(f"<blockquote>{_inline(quote.group(1))}</blockquote>")
            index += 1
            continue

        bullet = _BULLET.match(line)
        if bullet:
            output.append(f"{bullet.group(1)}• {_inline(line[bullet.end() :])}")
            index += 1
            continue

        output.append(_inline(line))
        index += 1
    return "\n".join(output)
