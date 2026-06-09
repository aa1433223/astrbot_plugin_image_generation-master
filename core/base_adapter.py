from __future__ import annotations

import abc
import asyncio
import base64
import copy
from typing import Any
from urllib.parse import urljoin, urlparse

import aiohttp

from astrbot.api import logger

from .constants import DEFAULT_DOWNLOAD_TIMEOUT
from .types import (
    AdapterConfig,
    ErrorCategory,
    GenerationRequest,
    GenerationResult,
    ImageCapability,
)
from .utils import mask_sensitive

try:
    from aiohttp_socks import ProxyConnector
except ImportError:  # pragma: no cover - optional dependency
    ProxyConnector = None


# ---------------------------------------------------------------------------
# 不应重试的错误类别（立即失败）
# ---------------------------------------------------------------------------
_NON_RETRYABLE_CATEGORIES = frozenset(
    {ErrorCategory.BAD_REQUEST, ErrorCategory.BALANCE}
)

# 可通过轮换 Key 恢复的错误类别
_KEY_ROTATION_CATEGORIES = frozenset(
    {ErrorCategory.AUTH_ERROR, ErrorCategory.RATE_LIMIT}
)

# 可通过降级参数恢复的错误类别
_DEGRADABLE_CATEGORIES = frozenset(
    {ErrorCategory.TIMEOUT, ErrorCategory.SERVER_ERROR}
)

# 降分辨率顺序
_RESOLUTION_DOWNGRADE = {"4K": "2K", "2K": "1K"}


class BaseImageAdapter(abc.ABC):
    """图像生成适配器基类。"""

    def __init__(self, config: AdapterConfig):
        self.config = config
        self.api_keys = config.api_keys or []
        self.current_key_index = 0
        self.base_url = (config.base_url or "").rstrip("/")
        self.model = config.model
        self.proxy = config.proxy
        self.timeout = config.timeout
        self.download_timeout = DEFAULT_DOWNLOAD_TIMEOUT
        self.max_retry_attempts = max(1, config.max_retry_attempts)
        self.safety_settings = config.safety_settings
        self._session: aiohttp.ClientSession | None = None

    @abc.abstractmethod
    def get_capabilities(self) -> ImageCapability:
        """获取适配器支持的功能。"""

    def _get_configured_capabilities(self) -> ImageCapability:
        """根据配置项构建适配器能力。"""
        capability_map: dict[str, ImageCapability] = {
            "text_to_image": ImageCapability.TEXT_TO_IMAGE,
            "image_to_image": ImageCapability.IMAGE_TO_IMAGE,
            "aspect_ratio": ImageCapability.ASPECT_RATIO,
            "resolution": ImageCapability.RESOLUTION,
        }

        result = ImageCapability.NONE
        for key, capability_flag in capability_map.items():
            if self.config.capability_options.get(key, False):
                result |= capability_flag
        return result

    async def close(self) -> None:
        """关闭底层的 HTTP 会话。"""

        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    def _get_session(self) -> aiohttp.ClientSession:
        """获取或创建 HTTP 会话。"""
        if self._is_socks_proxy():
            if ProxyConnector is None:
                raise RuntimeError("检测到 SOCKS 代理，但未安装 aiohttp-socks")
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession(
                    connector=ProxyConnector.from_url(self.proxy)
                )
            return self._session

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _is_socks_proxy(self) -> bool:
        """当前代理是否为 SOCKS 代理。"""
        if not self.proxy:
            return False
        return urlparse(self.proxy).scheme.lower().startswith("socks")

    def _request_proxy_kwargs(self) -> dict[str, Any]:
        """返回 aiohttp 单次请求使用的代理参数。

        SOCKS 代理通过 ClientSession 的 connector 处理，不能再传 proxy 参数。
        """
        if not self.proxy or self._is_socks_proxy():
            return {}
        return {"proxy": self.proxy}

    def _get_current_api_key(self) -> str:
        """获取当前使用的 API Key。"""
        if not self.api_keys:
            return ""
        return self.api_keys[self.current_key_index % len(self.api_keys)]

    def _get_request_api_key(self, request: GenerationRequest | None = None) -> str:
        """获取本次请求使用的 API Key，允许调用方按用户覆盖。"""
        if request and request.api_key_override:
            return request.api_key_override
        return self._get_current_api_key()

    def _get_masked_api_key(self) -> str:
        """获取脱敏后的当前 API Key，用于日志输出。"""
        return mask_sensitive(self._get_current_api_key())

    def _get_log_prefix(self, task_id: str | None = None) -> str:
        """获取统一的日志前缀。"""
        adapter_name = self.__class__.__name__.replace("Adapter", "")
        prefix = f"[ImageGen] [{adapter_name}]"
        if task_id:
            prefix += f" [{task_id}]"
        return prefix

    def _get_timeout(self) -> aiohttp.ClientTimeout:
        """获取统一的请求超时配置。"""
        return aiohttp.ClientTimeout(total=self.timeout)

    def _get_download_timeout(self) -> aiohttp.ClientTimeout:
        """获取统一的下载超时配置。"""
        return aiohttp.ClientTimeout(total=self.download_timeout)

    def _normalize_image_url(self, image_url: str, base_url: str | None = None) -> str:
        """规范化 API 返回的图片地址，支持相对路径。"""
        if image_url.startswith(("http://", "https://", "data:")):
            return image_url
        base = (base_url or self.base_url or "").rstrip("/")
        if not base:
            return image_url
        return urljoin(f"{base}/", image_url.lstrip("/"))

    def _decode_data_url(self, image_url: str) -> bytes | None:
        """解码 data:image/...;base64,... 形式的图片。"""
        if not image_url.startswith("data:image/") or ";base64," not in image_url:
            return None
        try:
            _, _, data_part = image_url.partition(";base64,")
            return base64.b64decode(data_part)
        except Exception:  # noqa: BLE001
            return None

    async def _download_image_from_url(
        self,
        url: str,
        task_id: str | None = None,
        *,
        base_url: str | None = None,
    ) -> bytes | None:
        """从 API 返回的 URL 下载图片，自动复用适配器代理与下载超时。"""
        prefix = self._get_log_prefix(task_id)
        download_url = self._normalize_image_url(url, base_url)
        try:
            async with self._get_session().get(
                download_url,
                timeout=self._get_download_timeout(),
                **self._request_proxy_kwargs(),
            ) as resp:
                if resp.status == 200:
                    return await resp.read()
                logger.error(f"{prefix} 下载图片失败 ({resp.status}): {download_url}")
        except Exception as exc:  # noqa: BLE001
            logger.error(f"{prefix} 下载图片异常: {exc}")
        return None

    async def _extract_openai_style_images(
        self,
        response: dict[str, Any],
        task_id: str | None = None,
        *,
        base_url: str | None = None,
    ) -> tuple[list[bytes] | None, str | None]:
        """提取 OpenAI 风格 data[].b64_json / data[].url 图片结果。"""
        prefix = self._get_log_prefix(task_id)

        if not isinstance(response, dict):
            return None, "响应格式异常"

        data = response.get("data")
        if not isinstance(data, list):
            return None, f"响应中未找到 data 字段: {response}"

        images: list[bytes] = []
        saw_decode_failure = False
        saw_download_failure = False

        for item in data:
            if not isinstance(item, dict):
                continue

            b64_json = item.get("b64_json")
            if b64_json:
                try:
                    images.append(base64.b64decode(b64_json))
                    continue
                except Exception as exc:  # noqa: BLE001
                    saw_decode_failure = True
                    logger.warning(f"{prefix} b64_json 解码失败: {exc}")

            image_url = item.get("url")
            if image_url:
                image_url = str(image_url)
                content = self._decode_data_url(image_url)
                if content is None:
                    content = await self._download_image_from_url(
                        image_url,
                        task_id,
                        base_url=base_url,
                    )
                if content is not None:
                    images.append(content)
                else:
                    saw_download_failure = True

        if images:
            return images, None
        if saw_download_failure:
            return None, "图片下载失败"
        if saw_decode_failure:
            return None, "b64_json 解码失败"
        return None, "未找到有效的图片数据"

    def _rotate_api_key(self) -> None:
        """轮换 API Key。"""
        if len(self.api_keys) > 1:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            logger.info(
                f"{self._get_log_prefix()} 轮换 API Key -> 索引 {self.current_key_index}"
            )

    def update_model(self, model: str) -> None:
        """更新使用的模型。"""
        self.model = model

    # ------------------------------------------------------------------
    # 错误分类
    # ------------------------------------------------------------------

    def _classify_error(self, error: str | None, status: int = 0) -> ErrorCategory:
        """根据错误信息和 HTTP 状态码分类错误。"""
        if not error and status == 0:
            return ErrorCategory.UNKNOWN

        error_lower = (error or "").lower()

        # 超时类
        if status in {504, 524, 408} or any(
            kw in error_lower
            for kw in ("timeout", "超时", "timed out", "poll_timeout")
        ):
            return ErrorCategory.TIMEOUT

        # 限流类
        if status == 429 or "rate_limit" in error_lower or "rate_limited" in error_lower:
            return ErrorCategory.RATE_LIMIT

        # 余额不足
        if status == 402 or "insufficient_balance" in error_lower or "余额" in error_lower:
            return ErrorCategory.BALANCE

        # 鉴权
        if status in {401, 403} or any(
            kw in error_lower
            for kw in ("auth_error", "auth_required", "model_not_allowed", "鉴权")
        ):
            return ErrorCategory.AUTH_ERROR

        # 参数错误
        if status == 400 or any(
            kw in error_lower
            for kw in ("bad_request", "invalid_request", "invalid_value", "invalid_size", "参数错误")
        ):
            return ErrorCategory.BAD_REQUEST

        # 下载失败
        if "download_failed" in error_lower:
            return ErrorCategory.DOWNLOAD

        # 网络连接
        if any(
            kw in error_lower
            for kw in ("connect", "network", "dns", "proxy_connect", "代理连接")
        ):
            return ErrorCategory.NETWORK

        # 服务端错误
        if status in {500, 502, 503}:
            return ErrorCategory.SERVER_ERROR

        return ErrorCategory.UNKNOWN

    # ------------------------------------------------------------------
    # 智能重试 + 降级回退
    # ------------------------------------------------------------------

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """带智能回退的图像生成模板方法。

        回退策略：
        1. 原始参数尝试（含 Key 轮换重试）
        2. 超时/服务端错误时自动降分辨率重试
        3. 降级到最低分辨率仍失败则返回错误

        子类应重写 ``_generate_once()`` 方法来实现具体的生成逻辑。
        如需在生成前进行预处理验证，可重写 ``_pre_generate()`` 方法。
        """
        if not self.api_keys and not request.api_key_override:
            return GenerationResult(images=None, error="未配置 API Key")

        # 预处理检查（子类可重写）
        pre_result = self._pre_generate(request)
        if pre_result is not None:
            return pre_result

        prefix = self._get_log_prefix(request.task_id)
        current_request = request
        last_error = "生成失败"
        degraded = False

        # 外层：参数降级循环
        while True:
            # 内层：同参数下的 Key 轮换重试
            for attempt in range(self.max_retry_attempts):
                if attempt:
                    logger.info(
                        f"{prefix} 重试 ({attempt + 1}/{self.max_retry_attempts})"
                    )

                images, err = await self._generate_once(current_request)
                if images is not None:
                    if degraded:
                        logger.info(
                            f"{prefix} 降级后生成成功 "
                            f"(resolution={current_request.resolution})"
                        )
                    return GenerationResult(images=images, error=None)

                last_error = err or "生成失败"
                error_cat = self._classify_error(last_error)

                # 不可重试的错误：立即失败
                if error_cat in _NON_RETRYABLE_CATEGORIES:
                    logger.warning(
                        f"{prefix} 不可重试的错误 ({error_cat.value}): {last_error}"
                    )
                    return GenerationResult(images=None, error=last_error)

                # 鉴权/限流：轮换 Key 可能有效
                if error_cat in _KEY_ROTATION_CATEGORIES:
                    if current_request.api_key_override:
                        logger.warning(
                            f"{prefix} 使用个人 Key 时发生 {error_cat.value}，停止重试"
                        )
                        return GenerationResult(images=None, error=last_error)
                    if len(self.api_keys) > 1:
                        self._rotate_api_key()
                        continue
                    # 只有一个 Key，不重试
                    logger.warning(
                        f"{prefix} {error_cat.value} 且仅有 1 个 Key，停止重试"
                    )
                    return GenerationResult(images=None, error=last_error)

                # 超时/服务端错误：先轮换 Key 重试，后续考虑降级
                if error_cat in _DEGRADABLE_CATEGORIES and len(self.api_keys) > 1:
                    self._rotate_api_key()
                    continue

                # 下载失败：轮换无意义，跳出内层直接降级
                if error_cat == ErrorCategory.DOWNLOAD:
                    logger.info(
                        f"{prefix} 下载失败，跳过 Key 轮换"
                    )
                    break

                # 网络/未知错误：短暂等待后轮换 Key 重试
                if attempt < self.max_retry_attempts - 1:
                    self._rotate_api_key()
                    wait_seconds = min(2 ** (attempt + 1), 10)
                    logger.info(
                        f"{prefix} {error_cat.value} 错误，等待 {wait_seconds}s 后重试"
                    )
                    await asyncio.sleep(wait_seconds)

            # 内层重试用尽，尝试降级参数
            error_cat = self._classify_error(last_error)
            if error_cat not in _DEGRADABLE_CATEGORIES:
                break  # 非超时/服务端错误，不适合降级

            current_resolution = (current_request.resolution or "1K").upper()
            lower_resolution = _RESOLUTION_DOWNGRADE.get(current_resolution)

            if not lower_resolution:
                break  # 已经是最低分辨率

            logger.info(
                f"{prefix} {error_cat.value} 后自动降分辨率: "
                f"{current_resolution} → {lower_resolution}"
            )

            # 创建降级请求（深拷贝避免修改原始请求）
            degraded_request = copy.copy(current_request)
            degraded_request.resolution = lower_resolution
            current_request = degraded_request
            degraded = True
            # 重置 Key 索引，从头开始
            self.current_key_index = 0

        return GenerationResult(images=None, error=f"重试失败: {last_error}")

    def _pre_generate(self, request: GenerationRequest) -> GenerationResult | None:
        """生成前的预处理检查。

        子类可重写此方法进行参数验证。
        返回 None 表示通过检查，返回 GenerationResult 表示提前返回错误。
        """
        return None

    @abc.abstractmethod
    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """执行单次生成请求。

        子类必须实现此方法。
        返回 (images, error) 元组，成功时 images 非空，失败时 error 非空。
        """
