from __future__ import annotations

import asyncio
from collections.abc import Iterable
from io import BytesIO

from PIL import Image

from astrbot.api import logger

from .constants import (
    MASK_MIN_LENGTH,
    MASK_PLACEHOLDER,
    MASK_VISIBLE_CHARS,
    SUPPORTED_ASPECT_RATIOS,
    SUPPORTED_RESOLUTIONS,
)
from .types import ImageData

SUPPORTED_IMAGE_FORMATS = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif",
}

# 使用 constants.py 中的定义，转换为 set 以保持向后兼容
ALLOWED_ASPECT_RATIOS = set(SUPPORTED_ASPECT_RATIOS)
ALLOWED_RESOLUTIONS = set(SUPPORTED_RESOLUTIONS)


def detect_mime_type(data: bytes) -> str:
    """根据魔数（Magic Numbers）尽力检测 MIME 类型。"""

    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if len(data) > 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"heic", b"heix", b"heim", b"heis"):
            return "image/heic"
        if brand in (b"mif1", b"msf1", b"heif"):
            return "image/heif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def _sync_convert_image_format(image_data: bytes, mime_type: str) -> ImageData:
    """同步将不支持的图像转换为 JPEG。"""

    try:
        img = Image.open(BytesIO(image_data))

        if img.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            elif img.mode == "LA":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[3])
            img = background

        output = BytesIO()
        img.save(output, format="JPEG", quality=95)
        logger.debug("[ImageGen] 已将图像转换为 JPEG")
        return ImageData(data=output.getvalue(), mime_type="image/jpeg")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[ImageGen] 图像转换失败: {exc}")
        return ImageData(data=image_data, mime_type=mime_type)


async def convert_image_format(image_data: bytes, mime_type: str) -> ImageData:
    """如果 MIME 类型不支持，则转换图像。"""

    real_mime = detect_mime_type(image_data)
    if real_mime in SUPPORTED_IMAGE_FORMATS:
        return ImageData(data=image_data, mime_type=real_mime)
    logger.info(f"[ImageGen] 正在转换图像格式: {mime_type} -> image/jpeg")
    return await asyncio.to_thread(_sync_convert_image_format, image_data, mime_type)


async def convert_images_batch(images: Iterable[ImageData]) -> list[ImageData]:
    """并行批量转换图像。"""

    tasks = [convert_image_format(img.data, img.mime_type) for img in images]
    return await asyncio.gather(*tasks)


def validate_aspect_ratio(value: str | None) -> str | None:
    """验证宽高比是否在允许的集合中。"""

    if value is None:
        return None
    return value if value in ALLOWED_ASPECT_RATIOS else None


def validate_resolution(value: str | None) -> str | None:
    """验证分辨率是否在允许的集合中。"""

    if value is None:
        return None
    return value if value in ALLOWED_RESOLUTIONS else None


def mask_sensitive(
    value: str,
    visible_chars: int = MASK_VISIBLE_CHARS,
    min_length: int = MASK_MIN_LENGTH,
    placeholder: str = MASK_PLACEHOLDER,
) -> str:
    """对敏感信息进行脱敏处理。

    Args:
        value: 需要脱敏的字符串
        visible_chars: 两端显示的字符数
        min_length: 需要脱敏的最小长度
        placeholder: 中间的占位符

    Returns:
        脱敏后的字符串
    """
    if len(value) <= min_length:
        return placeholder
    return f"{value[:visible_chars]}{placeholder}{value[-visible_chars:]}"
