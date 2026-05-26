from __future__ import annotations

import base64
import time
from typing import Any

from astrbot.api import logger

from ..core.base_adapter import BaseImageAdapter
from ..core.constants import (
    GITEE_AI_DEFAULT_BASE_URL,
    RESOLUTION_1K_MAP,
    RESOLUTION_2K_MAP,
)
from ..core.types import GenerationRequest, GenerationResult, ImageCapability


class ZImageAdapter(BaseImageAdapter):
    """Gitee AI 图像生成适配器 (z-image-turbo)。"""

    DEFAULT_BASE_URL = GITEE_AI_DEFAULT_BASE_URL

    def get_capabilities(self) -> ImageCapability:
        """获取适配器支持的功能。"""
        return self._get_configured_capabilities()

    # generate() 方法由基类提供，使用模板方法模式

    def _pre_generate(self, request: GenerationRequest) -> GenerationResult | None:
        """Z-Image 不支持参考图，在生成前进行检查。"""
        if request.images:
            return GenerationResult(
                images=None, error="Z-Image 适配器目前仅支持文生图，请勿上传图片。"
            )

        prefix = self._get_log_prefix(request.task_id)
        logger.info(
            f"{prefix} 开始生成: prompt='{request.prompt[:50]}...', model='{self.model or 'z-image-turbo'}'"
        )
        return None

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """执行单次生图请求。"""
        start_time = time.time()
        payload = self._build_payload(request)
        session = self._get_session()
        prefix = self._get_log_prefix(request.task_id)

        base = self.base_url or self.DEFAULT_BASE_URL
        url = f"{base.rstrip('/')}/v1/images/generations"

        logger.debug(f"{prefix} 请求 URL: {url}, Payload 字段: {list(payload.keys())}")

        headers = {
            "Authorization": f"Bearer {self._get_current_api_key()}",
            "Content-Type": "application/json",
            "X-Failover-Enabled": "true",
        }

        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                **self._request_proxy_kwargs(),
                timeout=self._get_timeout(),
            ) as resp:
                duration = time.time() - start_time
                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(
                        f"{prefix} API 错误 ({resp.status}, 耗时: {duration:.2f}s): {error_text}"
                    )
                    return None, f"API 错误 ({resp.status})"

                data = await resp.json()
                logger.info(f"{prefix} 生成成功 (耗时: {duration:.2f}s)")
                return await self._extract_images(data, request.task_id)
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{prefix} 请求异常 (耗时: {duration:.2f}s): {e}")
            return None, str(e)

    def _build_payload(self, request: GenerationRequest) -> dict:
        """构建请求载荷。"""
        prefix = self._get_log_prefix(request.task_id)

        size = "1024x1024"
        aspect_ratio = request.aspect_ratio or "1:1"
        if aspect_ratio == "自动":
            aspect_ratio = "1:1"

        if request.resolution in ("2K", "4K"):
            # 4K 暂时沿用 2K 的逻辑，因为 API 未提供 4K 映射
            size = RESOLUTION_2K_MAP.get(aspect_ratio, "2048x2048")
        else:
            size = RESOLUTION_1K_MAP.get(aspect_ratio, "1024x1024")

        logger.debug(
            f"{prefix} 参数: size={size}, aspect_ratio={aspect_ratio}, resolution={request.resolution or '1K'}"
        )

        payload: dict[str, Any] = {
            "model": self.model or "z-image-turbo",
            "prompt": request.prompt,
            "size": size,
            "num_inference_steps": 9,
        }

        return payload

    async def _extract_images(
        self, data: dict, task_id: str | None = None
    ) -> tuple[list[bytes] | None, str | None]:
        """从 API 响应中提取图像数据。"""
        prefix = self._get_log_prefix(task_id)
        # Gitee 的响应格式通常遵循 OpenAI 规范
        images, error = await self._extract_openai_style_images(data, task_id)

        if not images:
            return None, error or "未生成任何图像"

        logger.info(f"{prefix} 成功提取 {len(images)} 张图像")
        return images, None
