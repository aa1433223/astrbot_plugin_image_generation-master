"""
图片处理模块 - 下载、提取、缓存管理
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import os
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import unquote, urlparse

import astrbot.api.message_components as Comp
from astrbot.api import logger
from astrbot.core.utils.io import download_image_by_url

if TYPE_CHECKING:
    from astrbot.api.event import AstrMessageEvent


@dataclass(slots=True)
class CachedImage:
    """已成功落盘缓存并读回的参考图。"""

    data: bytes
    mime_type: str
    cache_path: str
    source_key: str


@dataclass(slots=True)
class FetchedImages:
    """消息参考图提取结果。"""

    images: list[tuple[bytes, str]]
    candidate_count: int
    cache_paths: list[str]
    failed_count: int = 0

    @property
    def has_candidates(self) -> bool:
        return self.candidate_count > 0


class ImageProcessor:
    """图片处理器 - 负责图片下载、提取和缓存管理。"""

    def __init__(self, cache_dir: str, max_image_size_mb: int, max_cache_count: int):
        self._cache_dir = cache_dir
        self._max_image_size_mb = max_image_size_mb
        self._max_cache_count = max_cache_count
        self._ensure_cache_dir()

    def _ensure_cache_dir(self) -> None:
        """确保缓存目录存在。"""
        os.makedirs(self._cache_dir, exist_ok=True)

    def update_settings(
        self, max_image_size_mb: int | None = None, max_cache_count: int | None = None
    ) -> None:
        """更新设置。"""
        if max_image_size_mb is not None:
            self._max_image_size_mb = max_image_size_mb
        if max_cache_count is not None:
            self._max_cache_count = max_cache_count

    @property
    def cache_dir(self) -> str:
        """获取缓存目录路径。"""
        return self._cache_dir

    def _cache_path(self) -> Path:
        return Path(self._cache_dir)

    def _source_key(self, source: str, *, kind: str = "image") -> str:
        """生成稳定来源键；直接图和引用图使用同一个 URL/路径键以便去重。"""
        normalized = (source or "").strip()
        if os.path.exists(normalized):
            normalized = str(Path(normalized).resolve())
            return f"{kind}:{normalized}"
        parsed = urlparse(normalized)
        if parsed.scheme == "file":
            local_path = self._file_uri_to_path(normalized)
            if local_path:
                normalized = str(local_path.resolve())
        return f"{kind}:{normalized}"

    def _source_digest(self, source_key: str) -> str:
        return hashlib.md5(source_key.encode("utf-8")).hexdigest()[:16]

    def _safe_source_label(self, source_key: str) -> str:
        return self._source_digest(source_key)

    def _looks_like_local_source(self, source: str) -> bool:
        if not source:
            return False
        if source.startswith(("file:///", "base64://")):
            return True
        return os.path.exists(source)

    def _is_image_component(self, component: Any) -> bool:
        return isinstance(component, Comp.Image)

    def _is_reply_component(self, component: Any) -> bool:
        return isinstance(component, Comp.Reply)

    def _is_at_component(self, component: Any) -> bool:
        return isinstance(component, Comp.At)

    def _is_file_component(self, component: Any) -> bool:
        file_cls = getattr(Comp, "File", None)
        return bool(file_cls and isinstance(component, file_cls))

    def _get_file_component_source(self, component: Any) -> str:
        return str(
            getattr(component, "file", "")
            or getattr(component, "file_", "")
            or getattr(component, "url", "")
            or ""
        ).strip()

    def _get_reply_message_id(self, component: Any) -> str:
        for attr in ("id", "message_id", "msg_id"):
            value = getattr(component, attr, None)
            if value:
                return str(value).strip()
        return ""

    async def _call_bot_action(
        self,
        event: AstrMessageEvent,
        action: str,
        **params: Any,
    ) -> dict[str, Any] | None:
        bot = getattr(event, "bot", None)
        if not bot or not hasattr(bot, "call_action"):
            return None
        try:
            result = await bot.call_action(action=action, **params)
        except TypeError:
            try:
                result = await bot.call_action(action, **params)
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"[ImageGen] call_action {action} 失败: params={params}, error={exc}")
                return None
        except Exception as exc:  # noqa: BLE001
            logger.debug(f"[ImageGen] call_action {action} 失败: params={params}, error={exc}")
            return None

        if not isinstance(result, dict):
            return None
        data = result.get("data")
        if isinstance(data, dict):
            return data
        return result

    def _delete_source_cache(self, digest: str) -> None:
        """删除同源旧参考图缓存，防止失败下载复用旧文件。"""
        cache_dir = self._cache_path()
        if not cache_dir.exists():
            return
        prefix = f"ref_{digest}"
        deleted = 0
        for path in cache_dir.glob(f"{prefix}*"):
            if not path.is_file():
                continue
            try:
                path.unlink()
                deleted += 1
                logger.debug(f"[ImageGen] 已删除同源旧参考图缓存: {path}")
            except OSError as exc:
                logger.warning(f"[ImageGen] 删除旧参考图缓存失败: {path} - {exc}")
        if deleted:
            logger.info(f"[ImageGen] 已清理同源旧参考图缓存 {deleted} 个 (source={digest})")

    def _mime_extension(self, mime: str) -> str | None:
        return {
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "image/gif": ".gif",
            "image/webp": ".webp",
            "image/heic": ".heic",
            "image/heif": ".heif",
        }.get(mime)

    def _file_uri_to_path(self, source: str) -> Path | None:
        parsed = urlparse(source)
        if parsed.scheme != "file":
            return None
        raw_path = unquote(parsed.path or "")
        if parsed.netloc:
            raw_path = f"//{parsed.netloc}{raw_path}"
        # Windows file URI commonly looks like file:///D:/path/to/file.png.
        if os.name == "nt" and len(raw_path) >= 3 and raw_path[0] == "/" and raw_path[2] == ":":
            raw_path = raw_path[1:]
        return Path(raw_path)

    def _decode_base64_source(self, source: str) -> bytes:
        payload = source[len("base64://") :] if source.startswith("base64://") else source
        if payload.startswith("data:") and ";base64," in payload:
            payload = payload.split(";base64,", 1)[1]
        return base64.b64decode(payload, validate=True)

    async def _write_source_to_tmp(self, source: str, tmp_path: Path) -> bytes:
        """把来源写入临时文件并返回临时文件 bytes。"""
        parsed = urlparse(source)
        source_path: Path | None = None

        if source.startswith("base64://") or (
            source.startswith("data:image/") and ";base64," in source
        ):
            try:
                data = self._decode_base64_source(source)
            except (binascii.Error, ValueError) as exc:
                raise ValueError(f"base64 图片解码失败: {exc}") from exc
            tmp_path.write_bytes(data)
            return data

        if os.path.exists(source):
            source_path = Path(source)
        elif parsed.scheme == "file":
            source_path = self._file_uri_to_path(source)
        elif not parsed.scheme:
            source_path = Path(source)

        if source_path is not None:
            if not source_path.exists() or not source_path.is_file():
                raise FileNotFoundError(f"本地图片不存在: {source_path}")
            data = source_path.read_bytes()
            tmp_path.write_bytes(data)
            return data

        if parsed.scheme not in ("http", "https"):
            raise ValueError(f"不支持的图片来源协议: {parsed.scheme or 'unknown'}")

        result = await download_image_by_url(source, path=str(tmp_path))
        if isinstance(result, (str, os.PathLike)):
            downloaded_path = Path(result)
            if downloaded_path.exists() and downloaded_path.is_file():
                data = downloaded_path.read_bytes()
                if downloaded_path != tmp_path:
                    tmp_path.write_bytes(data)
                return data
        if isinstance(result, bytes):
            tmp_path.write_bytes(result)
            return result
        if tmp_path.exists() and tmp_path.is_file():
            return tmp_path.read_bytes()
        raise RuntimeError("下载器未返回有效缓存文件")

    def _validate_image_bytes(self, data: bytes, source_label: str) -> str | None:
        if not data:
            logger.warning(f"[ImageGen] 参考图缓存校验失败: 空文件 (source={source_label})")
            return None

        if len(data) > self._max_image_size_mb * 1024 * 1024:
            logger.warning(
                f"[ImageGen] 参考图缓存校验失败: 图片超过大小限制 "
                f"({self._max_image_size_mb}MB, source={source_label})"
            )
            return None

        mime = self._detect_mime_type(data)
        if not self._mime_extension(mime):
            logger.warning(
                f"[ImageGen] 参考图缓存校验失败: 无法识别图片格式 "
                f"(mime={mime}, source={source_label})"
            )
            return None
        return mime

    async def cache_image(
        self,
        source: str,
        *,
        source_key: str | None = None,
        kind: str = "image",
    ) -> CachedImage | None:
        """确认图片成功写入插件缓存并读回后，再返回可用于生图的 bytes。"""
        source = (source or "").strip()
        if not source:
            return None

        self._ensure_cache_dir()
        source_key = source_key or self._source_key(source, kind=kind)
        digest = self._source_digest(source_key)
        source_label = self._safe_source_label(source_key)
        cache_dir = self._cache_path()
        tmp_path = cache_dir / f"tmp_ref_{digest}_{uuid.uuid4().hex}.tmp"

        self._delete_source_cache(digest)

        try:
            data = await self._write_source_to_tmp(source, tmp_path)
            logger.debug(
                f"[ImageGen] 参考图临时文件已写入: {tmp_path} "
                f"(source={source_label}, bytes={len(data) if data else 0})"
            )

            mime = self._validate_image_bytes(data, source_label)
            if not mime:
                return None

            ext = self._mime_extension(mime)
            if not ext:
                return None
            final_path = cache_dir / f"ref_{digest}{ext}"
            os.replace(tmp_path, final_path)
            cached_data = final_path.read_bytes()
            cached_mime = self._validate_image_bytes(cached_data, source_label)
            if not cached_mime:
                try:
                    final_path.unlink()
                except OSError:
                    pass
                return None

            logger.info(
                f"[ImageGen] 参考图已缓存: path={final_path}, "
                f"mime={cached_mime}, bytes={len(cached_data)}, source={source_label}"
            )
            return CachedImage(
                data=cached_data,
                mime_type=cached_mime,
                cache_path=str(final_path),
                source_key=source_key,
            )
        except Exception as exc:
            logger.error(f"[ImageGen] 参考图缓存失败 (source={source_label}): {exc}")
            return None
        finally:
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                    logger.debug(f"[ImageGen] 已删除参考图临时文件: {tmp_path}")
                except OSError as exc:
                    logger.debug(f"[ImageGen] 删除参考图临时文件失败: {tmp_path} - {exc}")

    async def download_image(self, url: str) -> tuple[bytes, str] | None:
        """下载图片并返回二进制数据和 MIME 类型。"""
        cached = await self.cache_image(url)
        if not cached:
            return None
        return cached.data, cached.mime_type

    def _detect_mime_type(self, data: bytes) -> str:
        """检测图片 MIME 类型。"""
        if data.startswith(b"\xff\xd8"):
            return "image/jpeg"
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
            return "image/gif"
        if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
            return "image/webp"
        if len(data) > 12 and data[4:8] == b"ftyp":
            brand = data[8:12]
            if brand in (b"heic", b"heix", b"heim", b"heis"):
                return "image/heic"
            if brand in (b"mif1", b"msf1", b"heif"):
                return "image/heif"
        return "application/octet-stream"

    async def get_avatar_cached(self, user_id: str) -> CachedImage | None:
        """获取用户头像并确认写入缓存。"""
        url = f"https://q4.qlogo.cn/headimg_dl?dst_uin={user_id}&spec=640"
        return await self.cache_image(
            url,
            source_key=f"avatar:{user_id}",
            kind="avatar",
        )

    async def get_avatar(self, user_id: str) -> bytes | None:
        """获取用户头像。"""
        cached = await self.get_avatar_cached(user_id)
        return cached.data if cached else None

    def _extract_raw_image_file_ids(self, event: AstrMessageEvent) -> list[str]:
        message_obj = getattr(event, "message_obj", None)
        raw_message = getattr(message_obj, "raw_message", None)
        raw_segments = getattr(raw_message, "message", None)
        if not isinstance(raw_segments, list):
            return []

        file_ids: list[str] = []
        for segment in raw_segments:
            if not isinstance(segment, dict) or segment.get("type") != "image":
                continue
            data = segment.get("data") or {}
            if not isinstance(data, dict):
                continue
            file_id = data.get("file") or data.get("file_id")
            if file_id:
                file_ids.append(str(file_id))
        return file_ids

    async def _get_aiocqhttp_original_image_source(
        self,
        event: AstrMessageEvent,
        file_id: str,
    ) -> tuple[str, str] | None:
        result = await self._call_bot_action(event, "get_image", file=file_id)
        if not result:
            return None

        source = result.get("file") or result.get("path") or result.get("url")
        if not source:
            return None
        source_type = "aiocqhttp_get_image_file" if result.get("file") or result.get("path") else "aiocqhttp_get_image_url"
        return str(source), source_type

    async def _get_quoted_message_segments(
        self,
        event: AstrMessageEvent,
        reply_component: Any,
    ) -> tuple[list[dict[str, Any]], str]:
        """通过 QQ/OneBot get_msg 拉取被引用消息的原始消息段。"""
        message_id = self._get_reply_message_id(reply_component)
        if not message_id:
            return [], ""

        action_message_id: int | str = (
            int(message_id) if message_id.isdigit() else message_id
        )
        result = await self._call_bot_action(
            event,
            "get_msg",
            message_id=action_message_id,
        )
        if not result:
            return [], message_id

        segments = result.get("message")
        if isinstance(segments, str):
            try:
                import json

                decoded = json.loads(segments)
                segments = decoded
            except Exception:  # noqa: BLE001
                segments = []

        if not isinstance(segments, list):
            raw_message = result.get("raw_message")
            segments = raw_message if isinstance(raw_message, list) else []

        valid_segments = [segment for segment in segments if isinstance(segment, dict)]
        if valid_segments:
            logger.info(
                f"[ImageGen] 已通过 get_msg 获取引用消息: message_id={message_id}, "
                f"segments={len(valid_segments)}"
            )
        return valid_segments, message_id

    async def _iter_raw_image_segment_sources(
        self,
        segment: dict[str, Any],
        event: AstrMessageEvent,
    ) -> tuple[list[tuple[str, str]], str | None]:
        if segment.get("type") != "image":
            return [], None

        data = segment.get("data") or {}
        if not isinstance(data, dict):
            return [], None

        candidates: list[tuple[str, str]] = []
        file_id = data.get("file") or data.get("file_id")
        path = data.get("path") or data.get("file_path")
        url = data.get("url")

        if path:
            candidates.append((str(path), "raw_image_path"))
        if file_id:
            original = await self._get_aiocqhttp_original_image_source(
                event,
                str(file_id),
            )
            if original:
                candidates.append(original)
            if self._looks_like_local_source(str(file_id)):
                candidates.append((str(file_id), "raw_image_file_local"))
        if url:
            candidates.append((str(url), "raw_image_url"))

        deduped: list[tuple[str, str]] = []
        seen: set[str] = set()
        for source, source_type in candidates:
            if not source or source in seen:
                continue
            seen.add(source)
            deduped.append((source, source_type))

        return deduped, str(file_id) if file_id else None

    async def _iter_image_component_sources(
        self,
        component: Any,
        event: AstrMessageEvent,
        *,
        raw_file_id: str | None = None,
    ) -> list[tuple[str, str]]:
        """按更接近原图的顺序返回图片来源候选。"""
        candidates: list[tuple[str, str]] = []

        path = str(getattr(component, "path", "") or "").strip()
        if path:
            candidates.append((path, "component_path"))

        file_value = str(getattr(component, "file", "") or "").strip()
        if file_value and self._looks_like_local_source(file_value):
            candidates.append((file_value, "component_file_local"))

        if raw_file_id:
            original = await self._get_aiocqhttp_original_image_source(event, raw_file_id)
            if original:
                candidates.append(original)

        convert_to_file_path = getattr(component, "convert_to_file_path", None)
        if callable(convert_to_file_path):
            try:
                converted = await convert_to_file_path()
                if converted:
                    candidates.append((str(converted), "component_convert_to_file_path"))
            except Exception as exc:  # noqa: BLE001
                logger.debug(f"[ImageGen] convert_to_file_path 获取图片失败: {exc}")

        if file_value and not self._looks_like_local_source(file_value):
            candidates.append((file_value, "component_file_remote"))

        url = str(getattr(component, "url", "") or "").strip()
        if url:
            candidates.append((url, "component_url_fallback"))

        deduped: list[tuple[str, str]] = []
        seen: set[str] = set()
        for source, source_type in candidates:
            if not source or source in seen:
                continue
            seen.add(source)
            deduped.append((source, source_type))
        return deduped

    async def _cache_first_valid_source(
        self,
        sources: list[tuple[str, str]],
        *,
        source_key: str,
        kind: str = "image",
    ) -> CachedImage | None:
        for source, source_type in sources:
            logger.debug(
                f"[ImageGen] 尝试缓存参考图来源: type={source_type}, "
                f"source={self._safe_source_label(source_key)}"
            )
            cached = await self.cache_image(source, source_key=source_key, kind=kind)
            if cached:
                logger.info(
                    f"[ImageGen] 参考图来源已采用: type={source_type}, "
                    f"path={cached.cache_path}"
                )
                return cached
            if source_type == "component_url_fallback":
                logger.warning(
                    "[ImageGen] 只能尝试平台图片 URL 兜底，可能不是原图"
                )
        return None

    async def _cache_file_image_if_valid(self, source: str) -> CachedImage | None:
        cached = await self.cache_image(source, kind="file_image")
        if cached:
            logger.info(f"[ImageGen] 文件消息已识别为参考图: path={cached.cache_path}")
            return cached
        logger.debug("[ImageGen] 文件消息不是可识别图片，已跳过")
        return None

    async def fetch_images_from_event(
        self, event: AstrMessageEvent
    ) -> list[tuple[bytes, str]]:
        """从消息事件中提取图片（包括直接发送的图片、引用消息中的图片、被@用户的头像）。"""
        result = await self.fetch_images_from_event_with_status(event)
        return result.images

    async def fetch_images_from_event_with_status(
        self, event: AstrMessageEvent
    ) -> FetchedImages:
        """从消息事件中提取参考图，并返回候选数、成功缓存路径等状态。"""
        cached_by_source: dict[str, CachedImage | None] = {}
        candidate_count = 0
        failed_count = 0

        if not event.message_obj or not event.message_obj.message:
            return FetchedImages(images=[], candidate_count=0, cache_paths=[])

        # 预扫描：记录引用消息的发送者以及各个 @ 出现次数，用于过滤自动 @
        reply_sender_id = None
        at_counts: dict[str, int] = {}

        for component in event.message_obj.message:
            if self._is_reply_component(component):
                if hasattr(component, "sender_id") and component.sender_id:
                    reply_sender_id = str(component.sender_id)
            elif self._is_at_component(component):
                if hasattr(component, "qq") and component.qq != "all":
                    uid = str(component.qq)
                    at_counts[uid] = at_counts.get(uid, 0) + 1

        async def remember_candidate(
            source: str,
            *,
            source_key: str | None = None,
            kind: str = "image",
        ) -> bool:
            nonlocal candidate_count, failed_count
            candidate_count += 1
            key = source_key or self._source_key(source, kind=kind)
            if key in cached_by_source:
                logger.info(
                    f"[ImageGen] 检测到重复参考图来源，使用最后一次缓存结果覆盖: "
                    f"source={self._safe_source_label(key)}"
                )
                cached_by_source.pop(key, None)

            cached = await self.cache_image(source, source_key=key, kind=kind)
            if cached:
                cached_by_source[key] = cached
                return True
            else:
                failed_count += 1
                cached_by_source[key] = None
                return False

        async def remember_sources(
            sources: list[tuple[str, str]],
            *,
            source_key: str,
            kind: str = "image",
        ) -> bool:
            nonlocal candidate_count, failed_count
            candidate_count += 1
            key = source_key
            if key in cached_by_source:
                logger.info(
                    f"[ImageGen] 检测到重复参考图来源，使用最后一次缓存结果覆盖: "
                    f"source={self._safe_source_label(key)}"
                )
                cached_by_source.pop(key, None)

            cached = await self._cache_first_valid_source(
                sources,
                source_key=key,
                kind=kind,
            )
            if cached:
                cached_by_source[key] = cached
                return True
            else:
                failed_count += 1
                cached_by_source[key] = None
                return False

        async def remember_file_image(source: str) -> bool:
            nonlocal candidate_count, failed_count
            cached = await self._cache_file_image_if_valid(source)
            if not cached:
                return False
            candidate_count += 1
            key = cached.source_key
            if key in cached_by_source:
                logger.info(
                    f"[ImageGen] 检测到重复文件参考图，使用最后一次缓存结果覆盖: "
                    f"source={self._safe_source_label(key)}"
                )
                cached_by_source.pop(key, None)
            cached_by_source[key] = cached
            return True

        raw_file_ids = self._extract_raw_image_file_ids(event)
        raw_file_index = 0

        for component in event.message_obj.message:
            try:
                if self._is_image_component(component):
                    # 处理直接发送的图片
                    raw_file_id = (
                        raw_file_ids[raw_file_index]
                        if raw_file_index < len(raw_file_ids)
                        else None
                    )
                    raw_file_index += 1
                    sources = await self._iter_image_component_sources(
                        component,
                        event,
                        raw_file_id=raw_file_id,
                    )
                    if sources:
                        stable_source = raw_file_id or sources[-1][0]
                        await remember_sources(
                            sources,
                            source_key=f"image:{stable_source}",
                        )
                elif self._is_reply_component(component):
                    # 处理引用消息中的图片
                    reply_image_cached = False
                    chain = getattr(component, "chain", None)
                    if chain:
                        for sub_comp in chain:
                            if self._is_image_component(sub_comp):
                                sources = await self._iter_image_component_sources(
                                    sub_comp,
                                    event,
                                )
                                if sources:
                                    if await remember_sources(
                                        sources,
                                        source_key=f"image:{sources[-1][0]}",
                                    ):
                                        reply_image_cached = True
                            elif self._is_file_component(sub_comp):
                                source = self._get_file_component_source(sub_comp)
                                if source:
                                    if await remember_file_image(source):
                                        reply_image_cached = True

                    # OneBot get_msg 只作为兜底：部分 QQ 引用消息没有填充 Reply.chain。
                    # 如果 chain 中的图片已经成功缓存，再拉原始消息会把同一张图算两次。
                    if not reply_image_cached:
                        raw_segments, reply_message_id = await self._get_quoted_message_segments(
                            event,
                            component,
                        )
                        for segment in raw_segments:
                            sources, file_id = await self._iter_raw_image_segment_sources(
                                segment,
                                event,
                            )
                            if not sources:
                                continue
                            stable_source = file_id or sources[-1][0]
                            await remember_sources(
                                sources,
                                source_key=f"reply:{reply_message_id}:image:{stable_source}",
                            )
                elif self._is_file_component(component):
                    source = self._get_file_component_source(component)
                    if source:
                        await remember_file_image(source)
                elif self._is_at_component(component):
                    # 处理 @ 用户的头像
                    if hasattr(component, "qq") and component.qq != "all":
                        uid = str(component.qq)
                        # 引用消息带来的单次自动 @ 默认忽略头像，除非用户再次显式 @
                        if reply_sender_id and uid == reply_sender_id:
                            if at_counts.get(uid, 0) == 1:
                                continue
                        self_id = str(event.get_self_id()).strip()
                        # 机器人单次被 @ 多为触发前缀，默认不取机器人头像
                        if self_id and uid == self_id and at_counts.get(uid, 0) == 1:
                            continue
                        await remember_candidate(
                            f"https://q4.qlogo.cn/headimg_dl?dst_uin={uid}&spec=640",
                            source_key=f"avatar:{uid}",
                            kind="avatar",
                        )
            except Exception as e:
                logger.error(f"[ImageGen] 提取消息组件图片失败: {e}")
                continue

        cached_images = [item for item in cached_by_source.values() if item]
        logger.info(
            f"[ImageGen] 参考图提取完成: 候选={candidate_count}, "
            f"成功={len(cached_images)}, 失败={failed_count}, cache_dir={self._cache_dir}"
        )
        return FetchedImages(
            images=[(item.data, item.mime_type) for item in cached_images],
            candidate_count=candidate_count,
            cache_paths=[item.cache_path for item in cached_images],
            failed_count=failed_count,
        )

    async def cleanup_cache(self) -> None:
        """执行缓存清理。"""
        if not os.path.exists(self._cache_dir):
            return

        files = []
        for f in os.listdir(self._cache_dir):
            path = os.path.join(self._cache_dir, f)
            if os.path.isfile(path):
                files.append((path, os.path.getmtime(path)))

        # 按修改时间排序（旧的在前）
        files.sort(key=lambda x: x[1])

        # 按数量清理
        if len(files) > self._max_cache_count:
            to_delete = files[: len(files) - self._max_cache_count]
            deleted_count = 0
            for path, _ in to_delete:
                try:
                    os.remove(path)
                    deleted_count += 1
                except OSError as e:
                    logger.debug(f"[ImageGen] 删除缓存文件失败: {path} - {e}")
            logger.info(
                f"[ImageGen] 已清理 {deleted_count}/{len(to_delete)} 个旧缓存文件 (按数量)"
            )

    def save_generated_image(self, task_id: str, img_bytes: bytes) -> str | None:
        """保存生成的图片到缓存目录，返回文件路径。"""
        try:
            import time

            file_name = f"gen_{task_id}_{int(time.time())}_{hashlib.md5(img_bytes).hexdigest()[:6]}.png"
            file_path = os.path.join(self._cache_dir, file_name)
            with open(file_path, "wb") as f:
                f.write(img_bytes)
            return file_path
        except Exception as exc:
            logger.error(f"[ImageGen] 保存图片失败: {exc}")
            return None
