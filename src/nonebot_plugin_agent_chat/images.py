from __future__ import annotations

import asyncio
import base64
import binascii
import ipaddress
import socket
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpcore
import httpx

from .errors import InputError
from .models import AgentImage


def detect_media_type(data: bytes) -> str:
    signatures = (
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"\xff\xd8\xff", "image/jpeg"),
        (b"GIF87a", "image/gif"),
        (b"GIF89a", "image/gif"),
    )
    for signature, media_type in signatures:
        if data.startswith(signature):
            return media_type
    if len(data) >= 12 and data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def _is_global_address(value: str) -> bool:
    try:
        return ipaddress.ip_address(value.split("%", 1)[0]).is_global
    except ValueError:
        return False


async def _resolve_global_addresses(host: str, port: int) -> list[str]:
    try:
        literal = ipaddress.ip_address(host.split("%", 1)[0])
    except ValueError:
        literal = None
    if literal is not None:
        addresses = [str(literal)]
    else:
        try:
            info = await asyncio.get_running_loop().getaddrinfo(
                host,
                port,
                type=socket.SOCK_STREAM,
            )
        except (OSError, UnicodeError, ValueError) as exc:
            raise InputError("无法解析图片地址") from exc
        addresses = list(dict.fromkeys(str(item[4][0]) for item in info))
    if not addresses or any(not _is_global_address(item) for item in addresses):
        raise InputError("图片地址不能指向本机或私有网络")
    return addresses


async def validate_remote_url(url: str) -> None:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"} or not parsed.hostname:
        raise InputError("只支持有效的 HTTP(S) 图片地址")
    if parsed.username or parsed.password:
        raise InputError("图片地址不能包含用户凭证")
    host = parsed.hostname.rstrip(".").lower()
    if host == "localhost":
        raise InputError("图片地址不能指向本机或私有网络")
    try:
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
    except ValueError as exc:
        raise InputError("图片地址端口无效") from exc
    await _resolve_global_addresses(host, port)


class _PinnedPublicNetworkBackend(httpcore.AsyncNetworkBackend):
    """Resolve once, reject non-public results, then connect to the checked IP."""

    def __init__(self) -> None:
        self._backend: Any = httpcore.AnyIOBackend()

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        addresses = await _resolve_global_addresses(host, port)
        last_error: Exception | None = None
        for address in addresses[:4]:
            try:
                return await self._backend.connect_tcp(
                    address,
                    port,
                    timeout=timeout,
                    local_address=local_address,
                    socket_options=socket_options,
                )
            except Exception as exc:  # noqa: BLE001 - try the next validated IP
                last_error = exc
        assert last_error is not None
        raise last_error

    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Any = None,
    ) -> httpcore.AsyncNetworkStream:
        raise OSError("Unix sockets are disabled for remote images")

    async def sleep(self, seconds: float) -> None:
        await self._backend.sleep(seconds)


class _PublicImageTransport(httpx.AsyncHTTPTransport):
    def __init__(self) -> None:
        super().__init__()
        self._pool = httpcore.AsyncConnectionPool(
            network_backend=_PinnedPublicNetworkBackend()
        )


async def load_remote_image(
    source: str,
    client: httpx.AsyncClient,
    max_bytes: int,
) -> AgentImage:
    if source.startswith("base64://"):
        encoded = source[len("base64://") :]
        if len(encoded) > (max_bytes * 4 // 3) + 4:
            raise InputError("图片总大小超过限制")
        try:
            data = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise InputError("图片数据无法解析") from exc
        if len(data) > max_bytes:
            raise InputError("图片总大小超过限制")
        media_type = detect_media_type(data)
        if media_type == "application/octet-stream":
            raise InputError("无法识别图片格式")
        return AgentImage(media_type=media_type, data=data)

    current_url = source
    chunks: list[bytes] = []
    try:
        for _ in range(4):
            await validate_remote_url(current_url)
            async with client.stream("GET", current_url) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location:
                        raise InputError("图片重定向缺少目标地址")
                    current_url = urljoin(current_url, location)
                    continue
                response.raise_for_status()
                content_type = response.headers.get("content-type")
                normalized_type = (content_type or "").split(";", 1)[0].lower()
                if normalized_type and not (
                    normalized_type.startswith("image/")
                    or normalized_type == "application/octet-stream"
                ):
                    raise InputError("引用内容不是图片")
                content_length = response.headers.get("content-length")
                if (
                    content_length
                    and content_length.isdigit()
                    and int(content_length) > max_bytes
                ):
                    raise InputError("图片总大小超过限制")
                size = 0
                async for chunk in response.aiter_bytes():
                    size += len(chunk)
                    if size > max_bytes:
                        raise InputError("图片总大小超过限制")
                    chunks.append(chunk)
                break
        else:
            raise InputError("图片重定向次数过多")
    except InputError:
        raise
    except httpx.HTTPError as exc:
        raise InputError("下载图片失败") from exc

    data = b"".join(chunks)
    media_type = detect_media_type(data)
    if media_type == "application/octet-stream":
        raise InputError("仅支持有效的 JPEG、PNG、GIF 和 WebP 图片")
    return AgentImage(media_type=media_type, data=data)


async def load_local_image(path: Path, max_bytes: int) -> AgentImage:
    try:
        size = path.stat().st_size
    except OSError as exc:
        raise InputError(f"无法读取图片：{path}") from exc
    if size > max_bytes:
        raise InputError("图片总大小超过限制")
    try:
        data = await asyncio.to_thread(path.read_bytes)
    except OSError as exc:
        raise InputError(f"无法读取图片：{path}") from exc
    media_type = detect_media_type(data)
    if media_type == "application/octet-stream":
        raise InputError(f"无法识别图片格式：{path}")
    return AgentImage(media_type=media_type, data=data)


@dataclass(frozen=True)
class ImageBudget:
    """Bounds for the images one message may carry.

    Keeping the count and the byte total together means every entry point
    (CLI, chat input, Room history) enforces the same limits and reports the
    same wording.
    """

    max_images: int
    max_image_bytes: int

    def check_count(self, count: int) -> None:
        if count > self.max_images:
            raise InputError(f"每次最多处理 {self.max_images} 张图片")

    def check_total(self, total_bytes: int) -> None:
        if total_bytes > self.max_image_bytes:
            raise InputError("图片总大小超过限制")

    def remaining_bytes(self, used_bytes: int) -> int:
        """Bytes still available after ``used_bytes``; never negative."""

        return max(0, self.max_image_bytes - used_bytes)

    def check_room_for(self, used_bytes: int) -> None:
        if self.remaining_bytes(used_bytes) <= 0:
            raise InputError("图片总大小超过限制")


@dataclass(frozen=True)
class ImageSource:
    """One inbound image reference from a chat adapter.

    Exactly one field is set: inline ``data``, a public ``url``, or an opaque
    ``media`` handle that only the adapter can resolve (for example a Telegram
    ``file_id``).
    """

    data: bytes | None = None
    url: str | None = None
    media: Any = None


def _image_from_bytes(data: bytes, remaining: int) -> AgentImage:
    if not data:
        raise InputError("图片数据为空")
    if len(data) > remaining:
        raise InputError("图片总大小超过限制")
    media_type = detect_media_type(data)
    if media_type == "application/octet-stream":
        raise InputError("仅支持有效的 JPEG、PNG、GIF 和 WebP 图片")
    return AgentImage(media_type=media_type, data=data)


async def load_image_sources(
    sources: Sequence[ImageSource],
    *,
    max_images: int,
    max_image_bytes: int,
    fetch_media: Callable[[Any], Awaitable[bytes | None]] | None = None,
) -> list[AgentImage]:
    """Load inline, remote, and adapter-resolved images under one byte budget.

    Remote URLs keep the existing SSRF-safe, redirect-free transport. Adapter
    media handles are resolved by ``fetch_media`` and their failures are
    replaced with a sanitized error so platform download URLs (which may embed
    credentials) never reach logs or the user.
    """

    if len(sources) > max_images:
        raise InputError(f"每次最多处理 {max_images} 张图片")
    if not sources:
        return []

    budget = ImageBudget(max_images=max_images, max_image_bytes=max_image_bytes)
    images: list[AgentImage] = []
    total = 0
    timeout = httpx.Timeout(20.0, connect=10.0)
    needs_client = any(source.url for source in sources)
    client = (
        httpx.AsyncClient(
            timeout=timeout,
            follow_redirects=False,
            transport=_PublicImageTransport(),
            trust_env=False,
        )
        if needs_client
        else None
    )
    try:
        for source in sources:
            remaining = budget.remaining_bytes(total)
            if remaining <= 0:
                budget.check_room_for(total)
            if source.data is not None:
                image = _image_from_bytes(source.data, remaining)
            elif source.url:
                assert client is not None
                image = await load_remote_image(source.url, client, remaining)
            elif source.media is not None and fetch_media is not None:
                try:
                    fetched = await fetch_media(source.media)
                except asyncio.CancelledError:
                    raise
                except InputError:
                    raise
                except Exception:  # noqa: BLE001 - adapter transport boundary
                    raise InputError("下载图片失败") from None
                if not fetched:
                    raise InputError("下载图片失败")
                image = _image_from_bytes(fetched, remaining)
            else:
                raise InputError("无法识别的图片来源")
            total += len(image.data)
            images.append(image)
    finally:
        if client is not None:
            await client.aclose()
    return images
