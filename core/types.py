from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any


class ErrorCategory(enum.Enum):
    """错误分类，用于智能回退决策。"""

    TIMEOUT = "timeout"          # 超时类：524/504/408/asyncio.TimeoutError
    RATE_LIMIT = "rate_limit"    # 限流类：429/503(rate_limited)
    SERVER_ERROR = "server_error"  # 服务端错误：500/502/503(非限流)
    AUTH_ERROR = "auth_error"    # 鉴权类：401/403
    BALANCE = "balance"          # 余额不足：402
    BAD_REQUEST = "bad_request"  # 参数错误：400
    NETWORK = "network"          # 网络连接：connect_failed/DNS
    DOWNLOAD = "download"        # 图片下载失败
    UNKNOWN = "unknown"


class AdapterType(str, enum.Enum):
    """支持的图像生成适配器类型。"""

    GEMINI = "gemini"
    GEMINI_OPENAI = "gemini_openai"
    OPENAI = "openai"
    Z_IMAGE = "z_image_gitee"
    JIMENG2API = "jimeng2api"
    GROK = "grok"


class ImageCapability(enum.Flag):
    """图像生成适配器支持的功能。"""

    NONE = 0
    TEXT_TO_IMAGE = enum.auto()  # 文生图
    IMAGE_TO_IMAGE = enum.auto()  # 图生图
    RESOLUTION = enum.auto()  # 指定分辨率
    ASPECT_RATIO = enum.auto()  # 指定宽高比


@dataclass
class AdapterMetadata:
    """关于适配器能力的元数据。"""

    name: str
    capabilities: ImageCapability = ImageCapability.TEXT_TO_IMAGE


@dataclass
class AdapterConfig:
    """构造适配器所需的配置。"""

    type: AdapterType = AdapterType.GEMINI
    name: str = ""  # 供应商展示名称
    base_url: str | None = None
    api_keys: list[str] = field(default_factory=list)
    model: str = ""
    available_models: list[str] = field(default_factory=list)
    proxy: str | None = None
    timeout: int = 180
    max_retry_attempts: int = 3
    safety_settings: str | None = None
    capability_options: dict[str, bool] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)  # 适配器特有配置


@dataclass
class ImageData:
    """带有 MIME 类型的图像二进制数据。"""

    data: bytes
    mime_type: str


@dataclass
class GenerationRequest:
    """用户生图请求。"""

    prompt: str
    images: list[ImageData] = field(default_factory=list)
    aspect_ratio: str | None = None
    resolution: str | None = None
    task_id: str | None = None


@dataclass
class GenerationResult:
    """生图尝试的结果。"""

    images: list[bytes] | None = None
    error: str | None = None
