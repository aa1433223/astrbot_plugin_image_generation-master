from __future__ import annotations

import base64
import re
import time
from typing import Any

import aiohttp

from astrbot.api import logger

from ..core.base_adapter import BaseImageAdapter
from ..core.constants import GEMINI_DEFAULT_BASE_URL
from ..core.types import GenerationRequest, ImageCapability


class GeminiOpenAIAdapter(BaseImageAdapter):
    """通过 OpenAI 兼容的聊天补全接口进行 Gemini 图像生成。"""

    DEFAULT_BASE_URL = GEMINI_DEFAULT_BASE_URL

    def get_capabilities(self) -> ImageCapability:
        """获取适配器支持的功能。"""
        return self._get_configured_capabilities()

    # generate() 方法由基类提供，使用模板方法模式

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """执行单次生图请求。"""
        payload = self._build_payload(request)
        session = self._get_session()
        response = await self._make_request(session, payload, request.task_id)
        if response is None:
            return None, "API 请求失败"

        images = await self._extract_images(response, request.task_id)
        if images:
            return images, None

        # 尝试提取文本错误信息
        if "choices" in response and response["choices"]:
            content = response["choices"][0].get("message", {}).get("content")
            if isinstance(content, str) and content.strip():
                return None, f"未生成图片，API 返回文本: {content[:100]}"
        return None, "响应中未找到图片 data"

    def _build_payload(self, request: GenerationRequest) -> dict:
        """构建请求载荷。"""
        message_content: list[dict] = [
            {"type": "text", "text": f"Generate an image: {request.prompt}"}
        ]

        for image in request.images:
            b64_data = base64.b64encode(image.data).decode("utf-8")
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image.mime_type};base64,{b64_data}"},
                }
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": message_content}],
            "modalities": ["image", "text"],
            "stream": False,
        }

        image_config: dict[str, Any] = {}
        generation_config: dict[str, Any] = {}

        if request.aspect_ratio and not request.images:
            image_config["aspectRatio"] = request.aspect_ratio
        if request.resolution:
            image_config["imageSize"] = request.resolution
        if image_config:
            generation_config["imageConfig"] = image_config
        if generation_config:
            payload["generationConfig"] = generation_config

        return payload

    async def _make_request(
        self,
        session: aiohttp.ClientSession,
        payload: dict,
        task_id: str | None,
    ) -> dict | None:
        """发送 API 请求。"""
        start_time = time.time()
        url = f"{self.base_url or self.DEFAULT_BASE_URL}/v1/chat/completions"
        api_key = self._get_current_api_key()
        masked_key = self._get_masked_api_key()
        prefix = self._get_log_prefix(task_id)
        logger.debug(f"{prefix} 请求 -> {url}, key={masked_key}")

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

        try:
            async with session.post(
                url,
                json=payload,
                headers=headers,
                timeout=self._get_timeout(),
                **self._request_proxy_kwargs(),
            ) as response:
                duration = time.time() - start_time
                logger.debug(
                    f"{prefix} 状态 -> {response.status} (耗时: {duration:.2f}s)"
                )
                if response.status != 200:
                    error_text = await response.text()
                    preview = (
                        error_text[:200] + "..."
                        if len(error_text) > 200
                        else error_text
                    )
                    logger.error(
                        f"{prefix} 错误 {response.status} (耗时: {duration:.2f}s): {preview}"
                    )
                    return None
                return await response.json()
        except Exception as e:
            duration = time.time() - start_time
            logger.error(f"{prefix} 请求异常 (耗时: {duration:.2f}s): {e}")
            return None

    async def _extract_images(
        self, response_data: dict[str, Any], task_id: str | None = None
    ) -> list[bytes] | None:
        """从响应数据中提取图像。"""
        images: list[bytes] = []
        prefix = self._get_log_prefix(task_id)

        # DALL-E 风格
        if isinstance(response_data.get("data"), list):
            openai_images, _ = await self._extract_openai_style_images(
                response_data,
                task_id,
            )
            if openai_images:
                images.extend(openai_images)

        # 聊天补全风格
        if choices := response_data.get("choices"):
            message = (
                choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            )
            content = message.get("content")

            if isinstance(content, str):
                markdown_matches = re.findall(r"!\[.*?\]\((.*?)\)", content)
                for url in markdown_matches:
                    decoded = self._decode_image_url(url, task_id)
                    if decoded:
                        images.append(decoded)
                    elif data := await self._download_image_from_url(url, task_id):
                        images.append(data)

                content_without_md = re.sub(r"!\[.*?\]\(.*?\)", "", content)
                pattern = re.compile(
                    r"data\s*:\s*image/([a-zA-Z0-9.+-]+)\s*;\s*base64\s*,\s*([-A-Za-z0-9+/=_\s]+)",
                    flags=re.IGNORECASE,
                )
                for _, b64_str in pattern.findall(content_without_md):
                    try:
                        images.append(base64.b64decode(b64_str))
                    except Exception as e:
                        logger.warning(f"{prefix} Base64 解码失败 (inline): {e}")

            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "image_url":
                        image_url = part.get("image_url", {}).get("url")
                        if not image_url:
                            continue
                        decoded = self._decode_image_url(image_url, task_id)
                        if decoded:
                            images.append(decoded)
                        elif data := await self._download_image_from_url(
                            image_url, task_id
                        ):
                            images.append(data)

            if message.get("images"):
                for img_item in message["images"]:
                    url = None
                    if isinstance(img_item, dict):
                        url = img_item.get("url") or img_item.get("image_url", {}).get(
                            "url"
                        )
                    elif isinstance(img_item, str):
                        url = img_item
                    if not url:
                        continue
                    decoded = self._decode_image_url(url, task_id)
                    if decoded:
                        images.append(decoded)
                    elif data := await self._download_image_from_url(url, task_id):
                        images.append(data)

        return images or None

    def _decode_image_url(self, url: str, task_id: str | None = None) -> bytes | None:
        """解码 Data URL 形式的图像。"""
        decoded = self._decode_data_url(url)
        if decoded is None and url.startswith("data:image/"):
            prefix = self._get_log_prefix(task_id)
            logger.error(f"{prefix} Base64 解码失败")
        return decoded
