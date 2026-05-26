from __future__ import annotations

import asyncio
import base64
import json
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path
import time
from typing import Any
from io import BytesIO
from urllib.parse import urlparse

import aiohttp
from astrbot.api import logger
from PIL import Image

from ..core.base_adapter import BaseImageAdapter
from ..core.types import GenerationRequest, ImageCapability, ImageData


class _OpenAIConnectStageError(Exception):
    def __init__(self, original: Exception, *, duration: float, used_proxy: bool):
        super().__init__(str(original))
        self.original = original
        self.duration = duration
        self.used_proxy = used_proxy


class OpenAIAdapter(BaseImageAdapter):
    """OpenAI / NewAPI image generation adapter."""

    MAX_EXACT_EDGE = 3840
    POLL_INTERVAL_SECONDS = 3
    TRACE_LOG_MAX_BYTES = 20 * 1024 * 1024
    TRACE_LOG_BACKUP_COUNT = 5
    _trace_loggers: dict[str, logging.Logger] = {}

    def get_capabilities(self) -> ImageCapability:
        return self._get_configured_capabilities() | ImageCapability.RESOLUTION

    def _is_gpt_image_model(self) -> bool:
        model_family = str(self.config.extra.get("model_family", "auto"))
        if model_family == "gpt-image":
            return True
        if model_family == "dall-e":
            return False
        return bool(self.model and "gpt-image" in self.model)

    def _prefer_url_response(self) -> bool:
        return bool(self.config.extra.get("prefer_url_response", False))

    def _trace_mode(self) -> bool:
        return bool(
            self.config.extra.get(
                "trace_mode", self.config.extra.get("enable_trace_log", False)
            )
        )

    def _trace_log_path(self) -> str | None:
        value = self.config.extra.get("trace_log_path")
        if value:
            return str(value)
        # 回退默认路径，确保 trace_mode 开启时日志一定能写入
        return "data/openai_trace.log"

    def _get_trace_file_logger(self) -> logging.Logger | None:
        path = self._trace_log_path()
        if not path:
            return None

        resolved = str(Path(path).resolve())
        existing = self._trace_loggers.get(resolved)
        if existing:
            return existing

        try:
            log_path = Path(resolved)
            log_path.parent.mkdir(parents=True, exist_ok=True)

            file_logger = logging.getLogger(
                f"astrbot_plugin_image_generation.openai_trace.{abs(hash(resolved))}"
            )
            file_logger.setLevel(logging.DEBUG)
            file_logger.propagate = False

            if not file_logger.handlers:
                handler = RotatingFileHandler(
                    log_path,
                    maxBytes=self.TRACE_LOG_MAX_BYTES,
                    backupCount=self.TRACE_LOG_BACKUP_COUNT,
                    encoding="utf-8",
                )
                handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
                file_logger.addHandler(handler)

            self._trace_loggers[resolved] = file_logger
            logger.info(f"[ImageGen] OpenAI trace 日志已初始化: {resolved}")
            return file_logger
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[ImageGen] OpenAI trace 日志初始化失败 (path={resolved}): {exc}")
            return None

    # ------------------------------------------------------------------
    # Trace 日志：脱敏处理
    # ------------------------------------------------------------------

    def _sanitize_value(self, obj: Any) -> Any:
        """递归脱敏 JSON 值：替换超长 base64 数据为摘要标记。"""
        if isinstance(obj, dict):
            return {k: self._sanitize_value(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._sanitize_value(v) for v in obj]
        if isinstance(obj, str) and len(obj) > 500:
            # data:image/...;base64,... URI
            if obj.startswith("data:image/") and ";base64," in obj:
                prefix, _, b64 = obj.partition(";base64,")
                size = len(b64) * 3 // 4
                return f"[DATA_URI {prefix} ~{size} bytes]"
            # 纯 base64 检测（采样前100字符）
            sample = obj[:100]
            if len(obj) > 1000 and all(
                c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r "
                for c in sample
            ):
                size = len(obj) * 3 // 4
                return f"[BASE64 ~{size} bytes]"
            return obj[:500] + f"...[TRUNCATED total={len(obj)}]"
        return obj

    def _sanitize_response_body(self, text: str) -> Any:
        """脱敏响应体：替换超长 base64 数据为摘要标记，保留结构信息。"""
        try:
            obj = json.loads(text)
            return self._sanitize_value(obj)
        except json.JSONDecodeError:
            if len(text) > 2000:
                return text[:2000] + f"...[TRUNCATED total={len(text)}]"
            return text

    def _trace_file(self, event: str, task_id: str | None = None, **fields: Any) -> None:
        if not self._trace_mode():
            return

        file_logger = self._get_trace_file_logger()
        if not file_logger:
            return

        # 对 body 字段自动脱敏
        if "body" in fields:
            body = fields["body"]
            if isinstance(body, str):
                fields["body"] = self._sanitize_response_body(body)

        payload = {
            "event": event,
            "task_id": task_id,
            **fields,
        }
        try:
            file_logger.debug(json.dumps(payload, ensure_ascii=False, default=str))
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"{self._get_log_prefix(task_id)} 写入 OpenAI trace 日志失败: {exc}")

    def _request_summary(
        self,
        request: GenerationRequest,
        response_format: str | None,
        *,
        async_wait: bool,
        use_edit: bool,
        submit_mode: str,
        reference_image_count: int,
    ) -> dict[str, Any]:
        return {
            "model": self.model or ("gpt-image-1" if use_edit else "dall-e-3"),
            "size": self._resolve_size(request),
            "response_format": response_format or "default",
            "wait": False if async_wait else None,
            "use_edit": use_edit,
            "submit_mode": submit_mode,
            "image_count": len(request.images),
            "reference_image_count": reference_image_count,
            "prompt_length": len(request.prompt or ""),
            "aspect_ratio": request.aspect_ratio,
            "resolution": request.resolution,
        }

    def _trace_reference_image_details(
        self,
        event: str,
        images: list[ImageData],
        *,
        task_id: str | None = None,
        submit_mode: str | None = None,
    ) -> None:
        self._trace_file(
            event,
            task_id,
            submit_mode=submit_mode,
            reference_image_count=len(images),
            reference_images=[
                {
                    "index": index,
                    "mime_type": img.mime_type,
                    "bytes": len(img.data),
                    "filename": self._image_filename(index, img.mime_type),
                }
                for index, img in enumerate(images)
            ],
        )

    def _get_reference_images(
        self, request: GenerationRequest, *, task_id: str | None = None
    ) -> list[str]:
        reference_images: list[str] = []
        prepared_images = self._prepare_reference_images(request, task_id=task_id)
        self._trace_reference_image_details(
            "reference_images_prepared",
            prepared_images,
            task_id=task_id,
            submit_mode="generations_json_references",
        )
        for img in prepared_images:
            mime_type = img.mime_type
            encoded = base64.b64encode(img.data).decode("ascii")
            reference_images.append(f"data:{mime_type};base64,{encoded}")
        return reference_images

    def _detect_image_mime(self, data: bytes) -> str:
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

    def _prepare_reference_images(
        self, request: GenerationRequest, *, task_id: str | None = None
    ) -> list[ImageData]:
        """将 OpenAI/NewAPI 参考图规范化为接口稳定接受的 PNG/JPEG。"""
        images = request.images[:4]
        if len(request.images) > 4:
            self._trace_file(
                "reference_images_truncated",
                task_id,
                original_count=len(request.images),
                kept_count=len(images),
            )
            logger.warning(
                f"{self._get_log_prefix(task_id)} 参考图超过 4 张，已截断为前 4 张"
            )

        prepared: list[ImageData] = []
        for index, img in enumerate(images):
            real_mime = self._detect_image_mime(img.data)
            if real_mime in {"image/png", "image/jpeg"}:
                prepared.append(ImageData(data=img.data, mime_type=real_mime))
                continue

            try:
                pil_image = Image.open(BytesIO(img.data))
                pil_image.load()
                if pil_image.mode in ("RGBA", "LA", "P"):
                    background = Image.new("RGB", pil_image.size, (255, 255, 255))
                    rgba_image = pil_image.convert("RGBA")
                    background.paste(rgba_image, mask=rgba_image.split()[3])
                    pil_image = background
                elif pil_image.mode != "RGB":
                    pil_image = pil_image.convert("RGB")

                output = BytesIO()
                pil_image.save(output, format="JPEG", quality=95)
                prepared.append(
                    ImageData(data=output.getvalue(), mime_type="image/jpeg")
                )
                self._trace_file(
                    "reference_image_converted",
                    task_id,
                    index=index,
                    source_mime=real_mime,
                    target_mime="image/jpeg",
                    bytes=len(output.getvalue()),
                )
                logger.info(
                    f"{self._get_log_prefix(task_id)} 参考图已转换为 JPEG "
                    f"(index={index}, source_mime={real_mime})"
                )
            except Exception as exc:  # noqa: BLE001
                self._trace_file(
                    "reference_image_skipped",
                    task_id,
                    index=index,
                    source_mime=real_mime,
                    exception=repr(exc),
                )
                logger.warning(
                    f"{self._get_log_prefix(task_id)} 参考图无法解析，已跳过 "
                    f"(index={index}, mime={real_mime}): {exc}"
                )
        return prepared

    def _image_filename(self, index: int, mime_type: str) -> str:
        extension = "jpg" if mime_type == "image/jpeg" else "png"
        return f"image_{index}.{extension}"

    def _is_newapi_async_enabled(self) -> bool:
        return bool(self.config.extra.get("newapi_async", True))

    def _proxy_fallback_direct_enabled(self) -> bool:
        return bool(self.config.extra.get("proxy_fallback_direct", True))

    def _connect_timeout_seconds(self) -> float:
        raw_value = self.config.extra.get("connect_timeout", 30)
        try:
            value = float(raw_value)
        except (TypeError, ValueError):
            value = 30.0
        if value <= 0:
            value = 30.0
        return value

    def _is_socks_proxy_enabled(self) -> bool:
        checker = getattr(self, "_is_socks_proxy", None)
        if callable(checker):
            return bool(checker())
        parsed = urlparse(str(self.proxy or ""))
        return parsed.scheme.lower().startswith("socks")

    def _is_public_openai_base(self) -> bool:
        if not self.base_url:
            return True
        hostname = (urlparse(self.base_url).hostname or "").lower()
        return hostname in {"api.openai.com", "openai.com", "www.openai.com"}

    def _should_use_newapi_async(self) -> bool:
        return not self._is_public_openai_base() and self._is_newapi_async_enabled()

    def _should_force_image_to_image(self, request: GenerationRequest) -> bool:
        return bool(request.images)

    def _reference_image_trace_items(
        self, images: list[ImageData]
    ) -> list[dict[str, Any]]:
        return [
            {
                "index": index,
                "mime_type": img.mime_type,
                "bytes": len(img.data),
                "filename": self._image_filename(index, img.mime_type),
            }
            for index, img in enumerate(images)
        ]

    def _trace_reference_images_ignored(
        self,
        response: dict[str, Any],
        *,
        task_id: str | None,
        submit_mode: str,
        reference_image_count: int,
    ) -> None:
        if reference_image_count <= 0:
            return

        usage = response.get("usage")
        if not isinstance(usage, dict):
            return

        input_details = usage.get("input_tokens_details")
        if not isinstance(input_details, dict):
            return

        image_tokens = input_details.get("image_tokens")
        try:
            ignored = int(image_tokens) == 0
        except (TypeError, ValueError):
            ignored = False
        if not ignored:
            return

        self._trace_file(
            "reference_images_ignored",
            task_id,
            submit_mode=submit_mode,
            reference_image_count=reference_image_count,
            image_tokens=image_tokens,
        )
        logger.warning(
            f"{self._get_log_prefix(task_id)} API 响应显示参考图可能未被上游识别 "
            f"(mode={submit_mode}, reference_image_count={reference_image_count}, "
            f"image_tokens={image_tokens})"
        )

    # ------------------------------------------------------------------
    # 错误格式化：包含 524 等 CDN 超时的识别
    # ------------------------------------------------------------------

    def _parse_error_message(self, payload: Any) -> str | None:
        """从 API 错误响应中提取详细错误消息（error.message）。"""
        if isinstance(payload, str):
            payload = payload.strip()
            if payload.startswith("{"):
                try:
                    payload = json.loads(payload)
                except json.JSONDecodeError:
                    return None
            else:
                return None
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                msg = error.get("message")
                if isinstance(msg, str) and msg.strip():
                    return msg.strip()
        return None

    def _format_api_error(self, status: int, error_text: str, duration: float) -> str:
        message = self._parse_error_message(error_text)
        detail = f": {message}" if message else ""

        if status in {504, 524}:
            return (
                f"timeout_{status}: 请求在 {duration:.1f}s 后被网关/CDN 断开 "
                f"(status={status})"
            )
        if status == 429:
            error_code = self._parse_error_code(error_text) or "rate_limited"
            return f"{error_code}: 请求被限流{detail} (status=429)"
        if status == 402:
            return f"insufficient_balance: 余额不足{detail} (status=402)"
        if status in {401, 403}:
            error_code = self._parse_error_code(error_text) or f"auth_error_{status}"
            return f"{error_code}: 鉴权失败{detail} (status={status})"
        if status == 400:
            error_code = self._parse_error_code(error_text) or "bad_request"
            return f"{error_code}: 请求参数错误{detail} (status=400)"
        if status in {500, 502, 503}:
            error_code = self._parse_error_code(error_text) or f"server_error_{status}"
            return f"{error_code}: 上游服务错误{detail} (status={status})"
        return f"api_error_{status}: API 返回错误{detail} (status={status})"

    def _build_common_payload(
        self, request: GenerationRequest, response_format: str | None
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.model or "dall-e-3",
            "prompt": request.prompt,
            "n": 1,
            "size": self._resolve_size(request),
        }
        if response_format:
            payload["response_format"] = response_format
        return payload

    def _build_generation_payload(
        self, request: GenerationRequest, response_format: str | None
    ) -> dict[str, Any]:
        return self._build_common_payload(request, response_format)

    def _build_async_generation_payload(
        self,
        request: GenerationRequest,
        response_format: str | None,
        *,
        reference_images: list[str] | None = None,
    ) -> dict[str, Any]:
        payload = self._build_common_payload(request, response_format)
        payload["wait"] = False
        if reference_images:
            payload["reference_images"] = reference_images
        return payload

    def _build_payload(self, request: GenerationRequest) -> dict[str, Any]:
        response_format = self._initial_response_format()
        return self._build_generation_payload(request, response_format)

    def _initial_response_format(self) -> str | None:
        if self._is_gpt_image_model():
            return "url" if self._prefer_url_response() else None
        return "b64_json"

    def _build_edit_form(
        self,
        request: GenerationRequest,
        response_format: str | None,
        *,
        async_wait: bool,
        prepared_images: list[ImageData] | None = None,
    ) -> aiohttp.FormData:
        form = aiohttp.FormData()
        form.add_field("model", self.model or "gpt-image-1")
        form.add_field("prompt", request.prompt)
        form.add_field("n", "1")
        form.add_field("size", self._resolve_size(request))
        if response_format:
            form.add_field("response_format", response_format)
        # 注意：/v1/images/edits (multipart) 不支持 wait=false
        # 异步模式仅在 /v1/images/generations (JSON) 中生效
        # 因此这里不再添加 wait 字段
        images = (
            prepared_images
            if prepared_images is not None
            else self._prepare_reference_images(request, task_id=request.task_id)
        )
        for index, img in enumerate(images):
            form.add_field(
                "image",
                img.data,
                content_type=img.mime_type,
                filename=self._image_filename(index, img.mime_type),
            )
        return form

    def _resolve_size(self, request: GenerationRequest) -> str:
        if self._supports_exact_size_passthrough():
            return self._build_exact_size(request.aspect_ratio, request.resolution)
        return self._build_compatible_size(
            request.aspect_ratio, self._is_gpt_image_model()
        )

    # ------------------------------------------------------------------
    # 动态超时：根据分辨率和是否图生图自动调整
    # ------------------------------------------------------------------

    def _get_timeout(
        self,
        *,
        resolution: str | None = None,
        has_images: bool = False,
    ) -> aiohttp.ClientTimeout:
        connect_timeout = self._connect_timeout_seconds()

        base = self.timeout  # 用户配置的超时，默认 180s
        res = (resolution or "1K").upper()

        # 分辨率倍率
        resolution_multiplier = {"1K": 1.0, "2K": 1.5, "4K": 2.5}.get(res, 1.0)

        # 图生图额外时间（参考图处理需要更多时间）
        image_bonus = 60 if has_images else 0

        total = max(base * resolution_multiplier + image_bonus, 180)
        total = min(total, 600)  # 上限 600s 与上游一致

        return aiohttp.ClientTimeout(
            total=total,
            connect=connect_timeout,
            sock_connect=connect_timeout,
        )

    def _get_timeout_for_request(self, request: GenerationRequest) -> aiohttp.ClientTimeout:
        """根据请求参数计算合适的超时时间。"""
        return self._get_timeout(
            resolution=request.resolution,
            has_images=bool(request.images),
        )

    def _get_poll_timeout(self, request: GenerationRequest | None = None) -> aiohttp.ClientTimeout:
        connect_timeout = self._connect_timeout_seconds()
        res = ((request.resolution if request else None) or "1K").upper()
        base = self.timeout
        multiplier = {"1K": 1.0, "2K": 1.5, "4K": 2.5}.get(res, 1.0)
        total = max(base * multiplier, 180)
        total = min(total, 600)
        return aiohttp.ClientTimeout(
            total=total,
            connect=connect_timeout,
            sock_connect=connect_timeout,
        )

    def _get_download_timeout(self) -> aiohttp.ClientTimeout:
        connect_timeout = self._connect_timeout_seconds()
        return aiohttp.ClientTimeout(
            total=getattr(self, "download_timeout", 30),
            connect=connect_timeout,
            sock_connect=connect_timeout,
        )

    def _supports_exact_size_passthrough(self) -> bool:
        return not self._is_public_openai_base()

    # ------------------------------------------------------------------
    # 1K 分辨率修复：不再静默升级非方形 1K 到 2K
    # ------------------------------------------------------------------

    @staticmethod
    def _align_to_16(value: int) -> int:
        """向下对齐到最近的 16 的倍数（至少 16）。"""
        return max(16, (value // 16) * 16)

    def _build_exact_size(
        self, aspect_ratio: str | None, resolution: str | None
    ) -> str:
        normalized_resolution = (resolution or "1K").upper()
        width_ratio, height_ratio = self._parse_aspect_ratio(aspect_ratio)
        max_side = {"1K": 1024, "2K": 2048, "4K": self.MAX_EXACT_EDGE}.get(
            normalized_resolution,
            1024,
        )
        min_edge = 1024  # gpt-image-2 最低像素限制要求最短边 >= 1024

        # 像素预算上限（防止总像素超出 API 限制）
        # 1K: 1024×1792 ≈ 1.84M  2K: 2048×3584 ≈ 7.34M  4K: 控制在 ~8.3M
        pixel_budget = {"1K": 1_835_008, "2K": 7_340_032, "4K": 8_294_400}.get(
            normalized_resolution,
            1_835_008,
        )

        if width_ratio >= height_ratio:
            width = max_side
            height = round(max_side * height_ratio / width_ratio)
        else:
            width = round(max_side * width_ratio / height_ratio)
            height = max_side

        # 确保最短边不低于 min_edge（避免低于模型最低像素限制）
        short_edge = min(width, height)
        if short_edge < min_edge:
            scale = min_edge / short_edge
            width = round(width * scale)
            height = round(height * scale)

        # 像素预算限制：如果总像素超出预算，等比缩小
        total_pixels = width * height
        if total_pixels > pixel_budget:
            scale = (pixel_budget / total_pixels) ** 0.5
            width = round(width * scale)
            height = round(height * scale)

        # 对齐到 16 的倍数（API 要求 width 和 height 必须能被 16 整除）
        width = self._align_to_16(width)
        height = self._align_to_16(height)

        return f"{width}x{height}"

    def _build_compatible_size(
        self, aspect_ratio: str | None, gpt_model: bool
    ) -> str:
        if not aspect_ratio or aspect_ratio == "自动":
            return "1024x1024"

        if gpt_model:
            mapping = {
                "1:1": "1024x1024",
                "3:2": "1536x1024",
                "16:9": "1536x1024",
                "4:3": "1536x1024",
                "5:4": "1536x1024",
                "21:9": "1536x1024",
                "2:3": "1024x1536",
                "3:4": "1024x1536",
                "9:16": "1024x1536",
                "4:5": "1024x1536",
            }
        else:
            mapping = {
                "1:1": "1024x1024",
                "3:2": "1792x1024",
                "16:9": "1792x1024",
                "4:3": "1792x1024",
                "5:4": "1792x1024",
                "21:9": "1792x1024",
                "2:3": "1024x1792",
                "3:4": "1024x1792",
                "9:16": "1024x1792",
                "4:5": "1024x1792",
            }

        return mapping.get(aspect_ratio, "1024x1024")

    def _parse_aspect_ratio(self, aspect_ratio: str | None) -> tuple[int, int]:
        if not aspect_ratio or aspect_ratio == "自动":
            return 1, 1
        try:
            width_str, height_str = aspect_ratio.split(":", 1)
            width = int(width_str)
            height = int(height_str)
            if width > 0 and height > 0:
                return width, height
        except (ValueError, TypeError):
            pass
        return 1, 1

    # ------------------------------------------------------------------
    # HTTP 请求底层
    # ------------------------------------------------------------------

    def _is_proxy_connect_error(self, exc: BaseException) -> bool:
        if isinstance(
            exc,
            (
                aiohttp.ClientConnectorError,
                aiohttp.ClientConnectorCertificateError,
                aiohttp.ClientConnectorSSLError,
                aiohttp.ClientHttpProxyError,
                aiohttp.ServerFingerprintMismatch,
                asyncio.TimeoutError,
                ConnectionAbortedError,
                ConnectionError,
                TimeoutError,
            ),
        ):
            return True
        message = str(exc).lower()
        return any(
            marker in message
            for marker in (
                "ssl handshake",
                "cannot connect to host",
                "connect call failed",
                "connection refused",
                "connection reset",
                "connection aborted",
                "proxy connection",
                "tls",
            )
        )

    def _request_proxy_kwargs_for_mode(self, *, use_proxy: bool) -> dict[str, Any]:
        if not use_proxy:
            return {}
        return self._request_proxy_kwargs()

    async def _get_request_session_for_mode(self, *, use_proxy: bool):
        if use_proxy or not self._is_socks_proxy_enabled():
            return self._get_session(), False
        session = aiohttp.ClientSession()
        return session, True

    async def _request_text_once(
        self,
        method: str,
        url: str,
        *,
        task_id: str | None,
        timeout: aiohttp.ClientTimeout,
        use_proxy: bool,
        **kwargs: Any,
    ) -> tuple[int, str, float, bool, dict[str, str]]:
        """发送请求并返回 (status, text, duration, used_proxy, response_headers)。"""
        session, close_after = await self._get_request_session_for_mode(
            use_proxy=use_proxy
        )
        request_fn = getattr(session, method.lower(), None)
        if request_fn is None:
            if close_after:
                await session.close()
            raise RuntimeError(f"Unsupported HTTP method: {method}")

        start_time = time.time()
        response_started = False
        try:
            async with request_fn(
                url,
                timeout=timeout,
                **self._request_proxy_kwargs_for_mode(use_proxy=use_proxy),
                **kwargs,
            ) as resp:
                response_started = True
                text = await resp.text()
                duration = time.time() - start_time
                # 提取关键响应头
                resp_headers = self._extract_key_headers(resp.headers)
                return resp.status, text, duration, use_proxy, resp_headers
        except Exception as exc:  # noqa: BLE001
            duration = time.time() - start_time
            if use_proxy and not response_started and self._is_proxy_connect_error(exc):
                raise _OpenAIConnectStageError(
                    exc, duration=duration, used_proxy=True
                ) from exc
            raise
        finally:
            if close_after:
                await session.close()

    def _extract_key_headers(self, headers: Any) -> dict[str, str]:
        """提取对诊断有价值的响应头。"""
        key_names = {
            "cf-ray", "x-request-id", "x-ratelimit-remaining",
            "x-ratelimit-reset", "content-type", "retry-after",
            "x-task-id", "x-error-code",
        }
        result = {}
        if headers:
            for name in key_names:
                value = headers.get(name)
                if value:
                    result[name] = str(value)
        return result

    async def _request_bytes_once(
        self,
        method: str,
        url: str,
        *,
        task_id: str | None,
        timeout: aiohttp.ClientTimeout,
        use_proxy: bool,
        **kwargs: Any,
    ) -> tuple[int, bytes, str | None, float, bool]:
        session, close_after = await self._get_request_session_for_mode(
            use_proxy=use_proxy
        )
        request_fn = getattr(session, method.lower(), None)
        if request_fn is None:
            if close_after:
                await session.close()
            raise RuntimeError(f"Unsupported HTTP method: {method}")

        start_time = time.time()
        response_started = False
        try:
            async with request_fn(
                url,
                timeout=timeout,
                **self._request_proxy_kwargs_for_mode(use_proxy=use_proxy),
                **kwargs,
            ) as resp:
                response_started = True
                duration = time.time() - start_time
                if resp.status == 200:
                    return resp.status, await resp.read(), None, duration, use_proxy
                return resp.status, b"", await resp.text(), duration, use_proxy
        except Exception as exc:  # noqa: BLE001
            duration = time.time() - start_time
            if use_proxy and not response_started and self._is_proxy_connect_error(exc):
                raise _OpenAIConnectStageError(
                    exc, duration=duration, used_proxy=True
                ) from exc
            raise
        finally:
            if close_after:
                await session.close()

    def _trace_proxy_connect_failed(
        self,
        *,
        task_id: str | None,
        method: str,
        url: str,
        exc: _OpenAIConnectStageError,
        fallback_direct: bool,
    ) -> None:
        self._trace_file(
            "proxy_connect_failed",
            task_id,
            method=method,
            url=url,
            elapsed_seconds=round(exc.duration, 3),
            exception=repr(exc.original),
            exception_type=type(exc.original).__name__,
            fallback_direct=fallback_direct,
        )
        logger.warning(
            f"{self._get_log_prefix(task_id)} 代理连接失败 "
            f"(method={method}, url={url}, elapsed={exc.duration:.2f}s, "
            f"fallback_direct={fallback_direct}): {exc.original}"
        )

    def _format_proxy_direct_failure(
        self, proxy_exc: _OpenAIConnectStageError, direct_exc: Exception
    ) -> str:
        return (
            "代理连接失败且直连也失败: "
            f"proxy={proxy_exc.original}; direct={direct_exc}"
        )

    async def _request_bytes_with_proxy_fallback(
        self,
        method: str,
        url: str,
        *,
        task_id: str | None,
        timeout: aiohttp.ClientTimeout,
        **kwargs: Any,
    ) -> tuple[int, bytes, str | None, float, bool]:
        try:
            return await self._request_bytes_once(
                method, url, task_id=task_id, timeout=timeout,
                use_proxy=bool(self.proxy), **kwargs,
            )
        except _OpenAIConnectStageError as proxy_exc:
            fallback_direct = self._proxy_fallback_direct_enabled()
            self._trace_proxy_connect_failed(
                task_id=task_id, method=method, url=url,
                exc=proxy_exc, fallback_direct=fallback_direct,
            )
            if not fallback_direct:
                return 0, b"", str(proxy_exc.original), proxy_exc.duration, True
            try:
                result = await self._request_bytes_once(
                    method, url, task_id=task_id, timeout=timeout,
                    use_proxy=False, **kwargs,
                )
                self._trace_file(
                    "proxy_fallback_direct_success", task_id,
                    method=method, url=url, status=result[0],
                    elapsed_seconds=round(result[3], 3),
                )
                return result
            except Exception as direct_exc:  # noqa: BLE001
                self._trace_file(
                    "proxy_fallback_direct_failed", task_id,
                    method=method, url=url,
                    exception=repr(direct_exc),
                )
                raise RuntimeError(
                    self._format_proxy_direct_failure(proxy_exc, direct_exc)
                ) from direct_exc

    async def _request_json(
        self,
        method: str,
        url: str,
        *,
        task_id: str | None = None,
        timeout: aiohttp.ClientTimeout | None = None,
        request: GenerationRequest | None = None,
        **kwargs: Any,
    ) -> tuple[int, str, float]:
        prefix = self._get_log_prefix(task_id)
        request_timeout = timeout or self._get_timeout_for_request(request) if request else timeout or self._get_timeout()

        try:
            status, text, duration, used_proxy, resp_headers = await self._request_text_once(
                method, url, task_id=task_id, timeout=request_timeout,
                use_proxy=bool(self.proxy), **kwargs,
            )
        except _OpenAIConnectStageError as proxy_exc:
            fallback_direct = self._proxy_fallback_direct_enabled()
            self._trace_proxy_connect_failed(
                task_id=task_id, method=method, url=url,
                exc=proxy_exc, fallback_direct=fallback_direct,
            )
            if not fallback_direct:
                return 0, str(proxy_exc.original), proxy_exc.duration
            try:
                status, text, duration, used_proxy, resp_headers = await self._request_text_once(
                    method, url, task_id=task_id, timeout=request_timeout,
                    use_proxy=False, **kwargs,
                )
                self._trace_file(
                    "proxy_fallback_direct_success", task_id,
                    method=method, url=url, status=status,
                    elapsed_seconds=round(duration, 3),
                )
            except Exception as direct_exc:  # noqa: BLE001
                self._trace_file(
                    "proxy_fallback_direct_failed", task_id,
                    method=method, url=url, exception=repr(direct_exc),
                )
                raise RuntimeError(
                    self._format_proxy_direct_failure(proxy_exc, direct_exc)
                ) from direct_exc

        self._trace_file(
            "http_response", task_id,
            method=method, url=url, status=status,
            elapsed_seconds=round(duration, 3),
            body=text,
            response_headers=resp_headers,
            proxy_enabled=used_proxy,
            connect_timeout=self._connect_timeout_seconds(),
        )
        if status >= 400:
            # 非 200 状态码始终输出到 AstrBot 主日志，便于排查
            logger.warning(
                f"{prefix} {method} {url} -> status={status}, "
                f"elapsed={duration:.2f}s, proxy={'on' if used_proxy else 'off'}"
            )
        elif self._trace_mode():
            logger.debug(
                f"{prefix} {method} {url} -> status={status}, "
                f"elapsed={duration:.2f}s, proxy={'on' if used_proxy else 'off'}"
            )
        return status, text, duration

    def _parse_error_code(self, payload: Any) -> str | None:
        if isinstance(payload, dict):
            error = payload.get("error")
            if isinstance(error, dict):
                code = error.get("code") or error.get("type")
                if code:
                    return str(code)
            if isinstance(error, str) and error.strip():
                return error.strip()
            code = payload.get("code") or payload.get("type")
            if code:
                return str(code)
        if isinstance(payload, str) and payload.strip():
            stripped = payload.strip()
            if stripped.startswith("{"):
                try:
                    return self._parse_error_code(json.loads(stripped))
                except json.JSONDecodeError:
                    pass
            return stripped
        return None

    def _should_fallback_to_b64(self, error: str | None) -> bool:
        if not error:
            return False
        code = error.split(":", 1)[0].strip().lower()
        return code in {"download_failed", "invalid_response", "invalid_request_error"}

    # ------------------------------------------------------------------
    # 提交请求
    # ------------------------------------------------------------------

    async def _submit_request(
        self,
        request: GenerationRequest,
        *,
        response_format: str | None,
        use_edit: bool,
    ) -> tuple[dict[str, Any] | None, str | None]:
        base = self.base_url.rstrip("/") if self.base_url else "https://api.openai.com"
        headers = {"Authorization": f"Bearer {self._get_current_api_key()}"}
        prefix = self._get_log_prefix(request.task_id)
        async_wait = self._should_use_newapi_async()
        has_reference_images = self._should_force_image_to_image(request)

        # 图生图始终走 multipart edits
        if use_edit or has_reference_images:
            url = f"{base}/v1/images/edits"
            submit_mode = "edits_multipart"
            use_edit = True
        else:
            url = f"{base}/v1/images/generations"
            submit_mode = "generations_json"

        prepared_edit_images: list[ImageData] = []
        if use_edit:
            prepared_edit_images = self._prepare_reference_images(
                request, task_id=request.task_id
            )
            if not prepared_edit_images:
                return None, "invalid_reference_image: 参考图无法解析或格式不受支持"

        self._trace_file(
            "submit_request", request.task_id,
            method="POST", url=url,
            proxy_enabled=bool(self.proxy),
            connect_timeout=self._connect_timeout_seconds(),
            async_wait=async_wait and not use_edit,
            **self._request_summary(
                request, response_format,
                async_wait=async_wait and not use_edit,
                use_edit=use_edit,
                submit_mode=submit_mode,
                reference_image_count=len(prepared_edit_images),
            ),
        )

        if use_edit:
            # 图生图: multipart edits（不支持异步 wait=false）
            form = self._build_edit_form(
                request, response_format,
                async_wait=False, prepared_images=prepared_edit_images,
            )
            status, text, duration = await self._request_json(
                "POST", url, task_id=request.task_id,
                data=form, headers=headers, request=request,
            )
        else:
            # 文生图: JSON generations
            payload = self._build_generation_payload(request, response_format)
            if async_wait:
                payload["wait"] = False
            headers["Content-Type"] = "application/json"
            status, text, duration = await self._request_json(
                "POST", url, task_id=request.task_id,
                json=payload, headers=headers, request=request,
            )

        if has_reference_images:
            logger.info(
                f"{prefix} 图生图路径 (mode={submit_mode}, "
                f"ref_count={len(prepared_edit_images)})"
            )

        if status not in (200, 202):
            if status == 0:
                return None, f"network_connect_failed: {text}"
            # 将原始错误响应记录到 AstrBot 主日志，便于排查
            error_preview = text[:800] if isinstance(text, str) else str(text)[:800]
            logger.error(
                f"{prefix} API 请求失败 (status={status}, mode={submit_mode}, "
                f"elapsed={duration:.2f}s): {error_preview}"
            )
            return None, self._format_api_error(status, text, duration)

        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None, "invalid_response: API 返回的不是有效 JSON"

        self._trace_file(
            "submit_success", request.task_id,
            status=status, elapsed_seconds=round(duration, 3),
            upstream_task_id=data.get("task_id"),
            submit_mode=submit_mode,
        )
        return data, None

    async def _poll_newapi_task(
        self, task_id: str, request: GenerationRequest
    ) -> tuple[dict[str, Any] | None, str | None]:
        base = self.base_url.rstrip("/") if self.base_url else "https://api.openai.com"
        url = f"{base}/v1/images/tasks/{task_id}"
        prefix = self._get_log_prefix(request.task_id)
        start_time = time.time()
        headers = {"Authorization": f"Bearer {self._get_current_api_key()}"}
        poll_timeout = self._get_poll_timeout(request)
        max_poll_seconds = min(poll_timeout.total or 600, 600)

        while True:
            status, text, _ = await self._request_json(
                "GET", url, task_id=request.task_id,
                timeout=self._get_poll_timeout(request), headers=headers,
            )

            if status != 200:
                return None, self._format_api_error(status, text, 0.0)

            try:
                data = json.loads(text)
            except json.JSONDecodeError:
                return None, "invalid_response: 轮询返回的不是有效 JSON"

            task_status = str(data.get("status", "")).lower()
            elapsed = time.time() - start_time
            self._trace_file(
                "poll_status", request.task_id,
                upstream_task_id=task_id, status=task_status or "unknown",
                elapsed_seconds=round(elapsed, 3),
            )

            if task_status == "success":
                return data, None

            if task_status in {"failed", "error"}:
                error_code = self._parse_error_code(data) or "invalid_response"
                error_message = data.get("error")
                if isinstance(error_message, dict):
                    message = error_message.get("message") or error_code
                elif isinstance(error_message, str) and error_message.strip():
                    message = error_message
                else:
                    message = error_code
                return None, f"{error_code}: {message}"

            if elapsed >= max_poll_seconds:
                return None, "poll_timeout: 图像生成超时"

            await asyncio.sleep(self.POLL_INTERVAL_SECONDS)

    async def _download_image_from_url(
        self, url: str, task_id: str | None = None
    ) -> bytes | None:
        prefix = self._get_log_prefix(task_id)
        download_url = self._normalize_image_url(url)
        self._trace_file("download_start", task_id, url=download_url)

        try:
            status, image_bytes, body, duration, used_proxy = (
                await self._request_bytes_with_proxy_fallback(
                    "GET", download_url, task_id=task_id,
                    timeout=self._get_download_timeout(),
                )
            )
        except Exception as exc:  # noqa: BLE001
            self._trace_file(
                "download_exception", task_id, url=download_url,
                exception=repr(exc),
            )
            logger.error(f"{prefix} 图片下载异常: {exc}")
            return None

        if status == 200:
            self._trace_file(
                "download_success", task_id, url=download_url,
                elapsed_seconds=round(duration, 3), bytes=len(image_bytes),
            )
            return image_bytes

        self._trace_file(
            "download_http_error", task_id, url=download_url,
            status=status, elapsed_seconds=round(duration, 3),
        )
        logger.error(f"{prefix} 图片下载失败 ({status}): {download_url}")
        return None

    async def _extract_images(
        self, response: dict[str, Any], task_id: str | None = None
    ) -> tuple[list[bytes] | None, str | None]:
        if not isinstance(response, dict):
            return None, "invalid_response: 响应格式异常"

        if "error" in response and not response.get("data"):
            error_code = self._parse_error_code(response) or "invalid_response"
            error_value = response.get("error")
            if isinstance(error_value, dict):
                message = error_value.get("message") or error_code
            elif isinstance(error_value, str) and error_value.strip():
                message = error_value
            else:
                message = error_code
            return None, f"{error_code}: {message}"

        data = response.get("data")
        if not isinstance(data, list):
            return None, "invalid_response: 响应中未找到 data 字段"

        images: list[bytes] = []
        download_failed = False
        decode_failed = False

        for index, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            if "b64_json" in item and item["b64_json"]:
                try:
                    decoded = base64.b64decode(item["b64_json"])
                    images.append(decoded)
                    continue
                except Exception:  # noqa: BLE001
                    decode_failed = True

            url = item.get("url")
            if url:
                url_text = str(url)
                content = self._decode_data_url(url_text)
                if content is None:
                    content = await self._download_image_from_url(url_text, task_id)
                if content is not None:
                    images.append(content)
                else:
                    download_failed = True

        if images:
            return images, None
        if download_failed:
            return None, "download_failed: 图片下载失败"
        if decode_failed:
            return None, "invalid_response: b64_json 解码失败"
        return None, "invalid_response: 未找到有效的图片数据"

    # ------------------------------------------------------------------
    # 核心生成逻辑
    # ------------------------------------------------------------------

    async def _generate_with_response_format(
        self,
        request: GenerationRequest,
        response_format: str | None,
    ) -> tuple[list[bytes] | None, str | None]:
        has_reference_images = self._should_force_image_to_image(request)
        use_edit = has_reference_images

        # 文生图 + NewAPI 异步：使用 wait=false + 轮询
        # 图生图：始终同步（edits 不支持 wait=false）
        if self._should_use_newapi_async() and not use_edit:
            response, error = await self._submit_request(
                request, response_format=response_format,
                use_edit=use_edit,
            )
            if error:
                return None, error

            task_id = str((response or {}).get("task_id") or "")
            if task_id:
                response, error = await self._poll_newapi_task(task_id, request)
                if error:
                    return None, error
            return await self._extract_images(response or {}, request.task_id)

        # 同步模式（图生图始终走这条路径）
        response, error = await self._submit_request(
            request, response_format=response_format,
            use_edit=use_edit,
        )
        if error:
            return None, error
        return await self._extract_images(response or {}, request.task_id)

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """单次生成请求。

        图生图：强制走最基础的 multipart edits 同步模式，
                response_format 固定为 b64_json，忽略所有高级选项。
        文生图：使用 NewAPI 异步模式（wait=false + 轮询），支持 URL 回退。
        """
        # 首次调用时输出 trace 状态信息
        if not hasattr(self, "_trace_status_logged"):
            self._trace_status_logged = True
            trace_on = self._trace_mode()
            trace_path = self._trace_log_path()
            logger.info(
                f"[ImageGen] OpenAI adapter trace_mode={'ON' if trace_on else 'OFF'}, "
                f"trace_log_path={trace_path or '(none)'}"
            )
            if trace_on:
                # 主动触发 logger 初始化，确保目录和文件创建
                self._get_trace_file_logger()

        has_reference_images = self._should_force_image_to_image(request)

        if has_reference_images:
            # 图生图：强制最基础模式
            # - response_format 固定 b64_json（最稳定，所有端点都支持）
            # - 不走 URL 回退逻辑
            # - _generate_with_response_format 内部会自动走 edits 同步路径
            self._trace_file(
                "image_to_image_basic_mode", request.task_id,
                response_format="b64_json",
                reason="图生图强制使用最基础 multipart edits 模式",
            )
            return await self._generate_with_response_format(request, "b64_json")

        # 文生图：保持原有逻辑
        desired_format = self._initial_response_format()

        images, error = await self._generate_with_response_format(
            request, desired_format,
        )
        if images is not None:
            return images, None

        # URL 回退到 b64_json（仅文生图）
        if desired_format == "url" and self._should_fallback_to_b64(error):
            self._trace_file(
                "fallback_to_b64", request.task_id,
                reason=error, retry_response_format="b64_json",
            )
            logger.info(
                f"{self._get_log_prefix(request.task_id)} "
                f"URL 路线失败，回退到 b64_json"
            )
            return await self._generate_with_response_format(request, "b64_json")

        return None, error or "invalid_response: 未能获取图片数据"
