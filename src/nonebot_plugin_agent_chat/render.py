"""Render a Markdown answer to PNG pages for chat transports.

Untrusted model output is treated as data, never as markup:

* ``markdown-it-py`` runs with ``html=False``, so raw HTML in an answer is
  escaped instead of becoming live DOM.
* The Chromium context runs with JavaScript disabled and the page is loaded
  offline; every subresource request is aborted, so no remote or ``file://``
  fetch can happen while rendering.
* Only locally generated CSS, local fonts, and Pygments-highlighted code are
  injected into the page.

The Chromium process is not kept around: it starts with the first document, and
the batch that owns the last document asks for the teardown with
``close_if_idle()`` — a render that fails or is cancelled takes the browser down
itself — so a bot that rendered once keeps no browser resident. The next
document pays one cold start.

Rendering is optional: the renderer lives behind the ``render-image`` extra and
callers must degrade to text when it is unavailable.
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import os
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .errors import RenderError

logger = logging.getLogger(__name__)

PAGE_WIDTH = 900
MIN_SLICE_HEIGHT = 200
_MAX_FOOTER_LINKS = 20

# Page breaks prefer these block edges: a break inside a line is never wanted,
# while a table row or list item may break away from its parent block.
_BREAK_SELECTORS = (
    "body main > *",
    "body main table > tbody > tr",
    "body main ul > li",
    "body main ol > li",
    "body main pre",
    "body .sources",
)

_FONT_STACK = (
    '"Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans", '
    '"DejaVu Sans", system-ui, sans-serif'
)

_CSS = f"""
:root {{ color-scheme: light; }}
* {{ box-sizing: border-box; }}
html, body {{ margin: 0; padding: 0; background: #ffffff; }}
body {{
  width: {PAGE_WIDTH}px;
  padding: 28px 30px;
  color: #1f2328;
  font-family: {_FONT_STACK};
  font-size: 16px;
  line-height: 1.65;
  word-break: break-word;
}}
main > :first-child {{ margin-top: 0; }}
h1, h2, h3, h4, h5, h6 {{ line-height: 1.3; margin: 1.15em 0 0.5em; }}
h1 {{ font-size: 26px; border-bottom: 1px solid #d8dee4; padding-bottom: 6px; }}
h2 {{ font-size: 22px; border-bottom: 1px solid #d8dee4; padding-bottom: 4px; }}
h3 {{ font-size: 19px; }}
h4, h5, h6 {{ font-size: 17px; }}
p, li {{ margin: 0.5em 0; }}
ul, ol {{ padding-left: 1.6em; }}
li > ul, li > ol {{ margin: 0.2em 0; }}
table {{ border-collapse: collapse; width: 100%; margin: 0.8em 0; font-size: 15px; }}
th, td {{
  border: 1px solid #d8dee4;
  padding: 6px 10px;
  text-align: left;
  vertical-align: top;
}}
th {{ background: #f6f8fa; }}
blockquote {{
  margin: 0.8em 0;
  padding: 2px 0 2px 14px;
  color: #59636e;
  border-left: 4px solid #d8dee4;
}}
a {{ color: #0969da; text-decoration: none; }}
hr {{ border: none; border-top: 1px solid #d8dee4; margin: 1.4em 0; }}
code {{
  font-family: "DejaVu Sans Mono", "Noto Sans Mono", monospace;
  font-size: 14px;
}}
:not(pre) > code {{
  background: #f6f8fa;
  border-radius: 4px;
  padding: 1px 5px;
}}
pre {{
  white-space: pre-wrap;
  word-break: break-word;
  margin: 0.8em 0;
}}
.highlight {{
  background: #f6f8fa;
  border: 1px solid #d8dee4;
  border-radius: 6px;
  padding: 10px 12px;
  overflow-x: hidden;
}}
.highlight pre {{ margin: 0; }}
.sources {{
  margin-top: 26px;
  padding-top: 10px;
  border-top: 1px solid #d8dee4;
  font-size: 13px;
  color: #59636e;
}}
.sources ol {{ margin: 6px 0 0; padding-left: 22px; }}
.sources a {{ word-break: break-all; }}
img {{ display: none; }}
"""


@dataclass(frozen=True)
class RenderedPage:
    data: bytes
    width: int
    height: int


class RenderedDocument:
    """A laid-out document that can be sliced into bounded pages.

    Slicing is aligned to block boundaries so a page break never cuts through a
    line of text; only a single block taller than the whole page budget is split
    at a pixel offset.
    """

    def __init__(
        self,
        page: Any,
        renderer: MarkdownRenderer,
        total_height: int,
        boundaries: list[int],
    ) -> None:
        self.total_height = total_height
        self._boundaries = boundaries
        self._page = page
        self._renderer = renderer
        self._closed = False

    def slice_height(
        self,
        top: int,
        max_height: int,
        min_height: int = MIN_SLICE_HEIGHT,
    ) -> int:
        """Height of the next slice from ``top``, or ``0`` when finished."""

        if top >= self.total_height or max_height < 1:
            return 0
        remaining = self.total_height - top
        if remaining <= max_height:
            # Never leave a sliver page behind: the tail fits on one page.
            return remaining
        limit = min(top + max_height, self.total_height)
        floor = min(top + max(min_height, 1), limit)
        end = limit
        for boundary in self._boundaries:
            if boundary > limit:
                break
            if boundary >= floor:
                end = boundary
        return max(1, end - top)

    async def slice(self, top: int, height: int) -> RenderedPage:
        """Screenshot one slice, bounded by ``height`` device pixels."""

        height = max(1, min(height, max(1, self.total_height - top)))
        clip = {"x": 0, "y": top, "width": PAGE_WIDTH, "height": height}
        try:
            data = await self._page.screenshot(clip=clip, full_page=True)
        except Exception as exc:
            raise RenderError(f"Screenshot failed: {type(exc).__name__}") from exc
        return RenderedPage(data=data, width=PAGE_WIDTH, height=height)

    async def close(self) -> None:
        """Close the page and give the browser lease back; repeated calls are safe."""

        if self._closed:
            return
        self._closed = True
        # Bookkeeping first: a cancellation at the page close would strand the
        # lease, and a stranded lease keeps ``close_if_idle()`` from ever
        # releasing that browser again.
        self._renderer._release()
        await _safe_close(self._page)


class MarkdownRenderer:
    """Renders Markdown into pages through a browser leased per document.

    Chromium starts with the first document and lives while a document is open;
    the batch that owns the last one asks for the teardown with
    ``close_if_idle()``, while a render that fails or is cancelled takes the
    browser down itself. An idle bot therefore runs no browser process.
    """

    def __init__(self, *, timeout_seconds: float = 20.0) -> None:
        self.timeout_seconds = timeout_seconds
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._lock = asyncio.Lock()
        self._leases = 0
        self.last_error: str | None = None

    async def _lease_context(self) -> Any:
        """Launch on demand and take one lease, released by the caller.

        Launching and counting happen under the same lock, so a concurrent
        ``close_if_idle()`` cannot tear the browser down between the two: the
        returned context stays valid until the matching ``_release()``.
        """

        async with self._lock:
            if self._context is None:
                try:
                    await self._launch_context()
                except BaseException:
                    # Best effort, and no lease is taken either way. A launch
                    # interrupted inside a Playwright call can leave its driver
                    # process behind until this process exits — Playwright
                    # cannot be interrupted mid-call — but nothing that has a
                    # context is ever left unreachable by close_if_idle().
                    await self._discard_browser()
                    raise
            self._leases += 1
            return self._context

    async def _launch_context(self) -> None:
        """Start Playwright and the Chromium context; the caller holds the lock."""

        try:
            from playwright.async_api import async_playwright
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RenderError("Playwright is not installed") from exc
        try:
            self._playwright = await async_playwright().start()
            self._browser = await self._playwright.chromium.launch(
                args=["--no-sandbox", "--disable-dev-shm-usage"],
            )
            self._context = await self._browser.new_context(
                viewport={"width": PAGE_WIDTH, "height": 1200},
                device_scale_factor=1,
                java_script_enabled=False,
                offline=True,
            )
            # Belt and braces on top of offline mode: no subresource of any
            # scheme may load, so nothing can be fetched or exfiltrated.
            await self._context.route("**/*", _abort_route)
        except RenderError:
            raise
        except Exception as exc:
            await self._discard_browser()
            raise RenderError(f"Chromium launch failed: {type(exc).__name__}") from exc

    def _release(self) -> None:
        """Give back one lease taken by ``_lease_context()``."""

        if self._leases <= 0:
            # Never count below zero: a negative count would let
            # close_if_idle() tear the browser down under a live document.
            logger.debug("Browser lease released without a matching lease")
            return
        self._leases -= 1

    async def close_if_idle(self) -> bool:
        """Tear the browser down once no document is open.

        Called when a delivery batch ends, so the browser lives exactly as long
        as rendering is in flight instead of for the whole process. Returns
        whether a running browser was actually closed.
        """

        async with self._lock:
            if self._context is None or self._leases > 0:
                return False
            await self._discard_browser()
            return True

    async def _discard_browser(self) -> None:
        """Drop context, browser and driver; the caller serializes the teardown."""

        for closer in (
            getattr(self._context, "close", None),
            getattr(self._browser, "close", None),
            getattr(self._playwright, "stop", None),
        ):
            if closer is None:
                continue
            try:
                await closer()
            except Exception as exc:  # noqa: BLE001 - browser teardown boundary
                logger.debug("Browser teardown failed: %s", type(exc).__name__)
        self._context = None
        self._browser = None
        self._playwright = None

    async def render(
        self,
        markdown_text: str,
        *,
        sources: Iterable[str] | None = (),
    ) -> RenderedDocument:
        """Lay out ``markdown_text`` once; pages are sliced from the result.

        ``sources=None`` suppresses the footer entirely: not just the search
        sources, the answer's own inline links stay label-only too.
        """

        try:
            html = build_document_html(markdown_text, sources)
        except RenderError:
            raise
        except Exception as exc:
            raise RenderError(f"Markdown parsing failed: {type(exc).__name__}") from exc

        context = await self._lease_context()
        page = None
        handed_over = False
        try:
            page = await context.new_page()
            page.set_default_timeout(self.timeout_seconds * 1000)
            await asyncio.wait_for(
                page.set_content(html, wait_until="load"),
                timeout=self.timeout_seconds,
            )
            height = await self._document_height(page)
            boundaries = await self._block_boundaries(page, height)
            self.last_error = None
            document = RenderedDocument(page, self, height, boundaries)
            handed_over = True
            return document
        except RenderError:
            raise
        except asyncio.TimeoutError as exc:
            raise RenderError("Rendering timed out") from exc
        except Exception as exc:
            raise RenderError(f"Rendering failed: {type(exc).__name__}") from exc
        finally:
            if not handed_over:
                # Nothing owns the lease any more. Release it before any await,
                # so a cancellation cannot strand it, then drop the browser and
                # the page: a render that failed or was cancelled is exactly the
                # case that would otherwise leave a browser resident with no
                # document to release it later.
                self._release()
                await self.close_if_idle()
                if page is not None:
                    # The page exists as soon as new_page() returned, so
                    # re-raising without this would leak one tab per failure.
                    await _safe_close(page)

    async def _block_boundaries(self, page: Any, total_height: int) -> list[int]:
        """Y offsets where a page may break without cutting a text line."""

        found: set[int] = set()
        for selector in _BREAK_SELECTORS:
            try:
                locators = await page.locator(selector).all()
            except Exception as exc:  # noqa: BLE001 - browser boundary
                logger.debug(
                    "Break boundary probe failed for %s: %s",
                    selector,
                    type(exc).__name__,
                )
                continue
            for locator in locators:
                try:
                    box = await locator.bounding_box()
                except Exception as exc:  # noqa: BLE001 - browser boundary
                    logger.debug(
                        "Break boundary measurement failed: %s",
                        type(exc).__name__,
                    )
                    continue
                if not box:
                    continue
                for edge in (box["y"], box["y"] + box["height"]):
                    value = round(edge)
                    if 0 < value < total_height:
                        found.add(value)
        return sorted(found)

    async def _document_height(self, page: Any) -> int:
        try:
            box = await page.locator("body").bounding_box()
        except Exception as exc:
            raise RenderError(f"Measuring failed: {type(exc).__name__}") from exc
        if not box:
            raise RenderError("Rendered document has no layout box")
        return max(1, int(-(-box["height"] // 1)))

    async def close(self) -> None:
        """Force teardown (shutdown, CLI); open documents lose their browser."""

        async with self._lock:
            await self._discard_browser()


async def _abort_route(route: Any) -> None:
    try:
        await route.abort()
    except Exception as exc:  # noqa: BLE001 - browser teardown boundary
        logger.debug("Renderer cleanup failed: %s", type(exc).__name__)


async def _safe_close(page: Any) -> None:
    try:
        await page.close()
    except Exception as exc:  # noqa: BLE001 - browser teardown boundary
        logger.debug("Renderer cleanup failed: %s", type(exc).__name__)


@lru_cache(maxsize=1)
def markdown_parser() -> Any:
    """A Markdown parser that escapes raw HTML instead of emitting live DOM."""

    from markdown_it import MarkdownIt

    parser = MarkdownIt("commonmark", {"html": False}).enable(
        ["table", "strikethrough"]
    )
    parser.options.highlight = _highlight_code
    return parser


def build_document_html(markdown_text: str, sources: Iterable[str] | None = ()) -> str:
    """Render trusted Markdown to the standalone HTML document we screenshot.

    ``sources=None`` renders no footer at all (the switch-off case); otherwise
    the footer lists the answer's inline link targets plus ``sources``.
    """

    parser = markdown_parser()
    tokens = parser.parse(markdown_text)
    body = parser.renderer.render(tokens, parser.options, {})
    links = None if sources is None else _footnote_links(tokens, sources)
    return _document_html(body, links or [])


def _highlight_code(code: str, lang: str, attrs: str) -> str:
    del attrs
    try:
        from pygments import highlight
        from pygments.formatters import HtmlFormatter
        from pygments.lexers import get_lexer_by_name, guess_lexer
        from pygments.util import ClassNotFound
    except ImportError:  # pragma: no cover - optional dependency
        return ""
    try:
        lexer = get_lexer_by_name(lang) if lang else guess_lexer(code)
    except ClassNotFound:
        return ""
    try:
        return highlight(code, lexer, HtmlFormatter(style="friendly"))
    except Exception:  # noqa: BLE001 - third-party formatter boundary
        return ""


def _footnote_links(tokens: Iterable[Any], sources: Iterable[str]) -> list[str]:
    """Collect link targets, because a rendered image cannot be clicked."""

    links: list[str] = []
    for token in tokens:
        if token.type != "inline":
            continue
        for child in token.children or []:
            if child.type != "link_open":
                continue
            href = child.attrGet("href")
            if href and href.startswith(("http://", "https://")):
                links.append(href)
    for source in sources:
        if source.startswith(("http://", "https://")):
            links.append(source)
    unique: list[str] = []
    for link in links:
        if link not in unique:
            unique.append(link)
    return unique[:_MAX_FOOTER_LINKS]


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _pygments_css() -> str:
    """Stylesheet for highlighted code; empty when Pygments is unavailable."""

    try:
        from pygments.formatters import HtmlFormatter
    except ImportError:  # pragma: no cover - optional dependency
        return ""
    try:
        return HtmlFormatter(style="friendly").get_style_defs(".highlight")
    except Exception as exc:  # noqa: BLE001 - third-party boundary
        logger.debug("Pygments stylesheet unavailable: %s", type(exc).__name__)
        return ""


def _document_html(body: str, links: list[str]) -> str:
    footer = ""
    if links:
        items = "".join(
            f'<li><a href="{_escape(link)}">{_escape(link)}</a></li>' for link in links
        )
        footer = (
            '<section class="sources"><strong>来源 / 链接</strong>'
            f"<ol>{items}</ol></section>"
        )
    return (
        '<!DOCTYPE html><html lang="zh"><head><meta charset="utf-8">'
        f"<style>{_pygments_css()}{_CSS}</style></head>"
        f"<body><main>{body}</main>{footer}</body></html>"
    )


def _browser_roots() -> list[Path]:
    """Directories Playwright may keep its browser builds in."""

    configured = os.getenv("PLAYWRIGHT_BROWSERS_PATH")
    if configured and configured != "0":
        return [Path(configured)]
    if configured == "0":
        # Playwright can keep browsers inside its own package instead of the
        # shared cache; the manifest still names the expected revision.
        spec = importlib.util.find_spec("playwright")
        if spec is None or not spec.submodule_search_locations:
            return []
        package = Path(next(iter(spec.submodule_search_locations)))
        return [package / "driver" / "package" / ".local-browsers"]
    if sys.platform == "win32":
        local = os.getenv("USERPROFILE", "~")
        return [Path(local).expanduser() / "AppData" / "Local" / "ms-playwright"]
    if sys.platform == "darwin":
        return [Path("~/Library/Caches/ms-playwright").expanduser()]
    cache = os.getenv("XDG_CACHE_HOME") or "~/.cache"
    return [Path(cache).expanduser() / "ms-playwright"]


def _expected_browser_dirs() -> list[str]:
    """Directory names the *installed* Playwright revision expects.

    The manifest is authoritative: globbing for any ``chromium-*`` directory
    reports a stale revision as ready, which is exactly the false positive this
    function avoids.

    说明：按浏览器清单里的 revision 精确匹配，避免旧版本目录让探测误报 ready。
    """

    spec = importlib.util.find_spec("playwright")
    if spec is None or not spec.submodule_search_locations:
        return []
    manifest = (
        Path(next(iter(spec.submodule_search_locations)))
        / "driver"
        / "package"
        / "browsers.json"
    )
    if not manifest.is_file():
        return []
    try:
        data = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    names: list[str] = []
    for entry in data.get("browsers", []):
        name = str(entry.get("name", ""))
        revision = entry.get("revision")
        if name in {"chromium", "chromium-headless-shell"} and revision:
            names.append(f"{name.replace('-', '_')}-{revision}")
    return names


def _browsers_present() -> bool:
    expected = _expected_browser_dirs()
    if not expected:
        return False
    # Headless launches use the headless shell when this Playwright ships one.
    headless = [
        name for name in expected if name.startswith("chromium_headless_shell-")
    ]
    required = headless or expected
    for root in _browser_roots():
        for name in required:
            candidate = root / name
            try:
                if candidate.is_dir() and any(candidate.iterdir()):
                    return True
            except OSError:
                continue
    return False


def probe_renderer() -> tuple[bool, str]:
    """Cheap availability check used by status output and the CLI."""

    if importlib.util.find_spec("playwright") is None:
        return False, "playwright is not installed"
    if importlib.util.find_spec("markdown_it") is None:
        return False, "markdown-it-py is not installed"
    if importlib.util.find_spec("pygments") is None:
        return False, "pygments is not installed"
    if not _browsers_present():
        return False, "no Chromium build found for Playwright"
    return True, "ready"


_shared_renderer: MarkdownRenderer | None = None


def shared_renderer(timeout_seconds: float) -> MarkdownRenderer:
    """One reusable renderer for the whole process.

    The renderer outlives reloads, so the live timeout setting wins on every
    call instead of freezing at the value it was created with. Its browser does
    not outlive a batch: the delivery path releases it with ``close_if_idle()``
    as soon as the last document closes.
    """

    global _shared_renderer
    if _shared_renderer is None:
        _shared_renderer = MarkdownRenderer(timeout_seconds=timeout_seconds)
    else:
        _shared_renderer.timeout_seconds = timeout_seconds
    return _shared_renderer


async def close_shared_renderer() -> None:
    global _shared_renderer
    renderer, _shared_renderer = _shared_renderer, None
    if renderer is not None:
        await renderer.close()
