"""
Core module for image generation plugin
图像生成插件的核心模块
"""

from .base_adapter import BaseImageAdapter
from .config_manager import (
    CacheSettings,
    ConfigManager,
    GenerationSettings,
    ImageAuditSettings,
    PluginConfig,
    PromptAuditSettings,
    SafetyAuditSettings,
    UsageSettings,
)
from .constants import (
    DEFAULT_ASPECT_RATIO,
    DEFAULT_MAX_RETRY_ATTEMPTS,
    DEFAULT_RESOLUTION,
    DEFAULT_TIMEOUT,
    GEMINI_DEFAULT_BASE_URL,
    GEMINI_SAFETY_CATEGORIES,
    GITEE_AI_DEFAULT_BASE_URL,
    JIMENG_DEFAULT_BASE_URL,
    LOG_PREFIX,
    OPENAI_DEFAULT_BASE_URL,
    RESOLUTION_1K_MAP,
    RESOLUTION_2K_MAP,
    SUPPORTED_ASPECT_RATIOS,
    SUPPORTED_RESOLUTIONS,
)
from .image_processor import ImageProcessor
from .task_manager import TaskManager
from .types import (
    AdapterConfig,
    AdapterMetadata,
    AdapterType,
    GenerationRequest,
    GenerationResult,
    ImageCapability,
    ImageData,
)
from .usage_manager import UsageManager
from .utils import (
    convert_image_format,
    convert_images_batch,
    detect_mime_type,
    mask_sensitive,
    validate_aspect_ratio,
    validate_resolution,
)


def __getattr__(name: str):
    """按需导入会反向依赖 adapter 包的对象，避免适配器单独导入时循环引用。"""
    if name == "ImageGenerator":
        from .generator import ImageGenerator

        return ImageGenerator
    if name in {"ImageGenerationTool", "adjust_tool_parameters"}:
        from .llm_tool import ImageGenerationTool, adjust_tool_parameters

        return {
            "ImageGenerationTool": ImageGenerationTool,
            "adjust_tool_parameters": adjust_tool_parameters,
        }[name]
    if name == "SafetyAuditor":
        from .safety_auditor import SafetyAuditor

        return SafetyAuditor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    # 基类和核心组件
    "BaseImageAdapter",
    "ImageGenerator",
    "TaskManager",
    # 新增管理器
    "ConfigManager",
    "UsageManager",
    "ImageProcessor",
    # 配置数据类
    "PluginConfig",
    "UsageSettings",
    "CacheSettings",
    "GenerationSettings",
    "PromptAuditSettings",
    "ImageAuditSettings",
    "SafetyAuditSettings",
    # LLM 工具
    "ImageGenerationTool",
    "adjust_tool_parameters",
    "SafetyAuditor",
    # 数据类型
    "AdapterConfig",
    "AdapterMetadata",
    "AdapterType",
    "GenerationRequest",
    "GenerationResult",
    "ImageCapability",
    "ImageData",
    # 工具函数
    "convert_image_format",
    "convert_images_batch",
    "detect_mime_type",
    "mask_sensitive",
    "validate_aspect_ratio",
    "validate_resolution",
    # 常量
    "LOG_PREFIX",
    "DEFAULT_TIMEOUT",
    "DEFAULT_MAX_RETRY_ATTEMPTS",
    "DEFAULT_ASPECT_RATIO",
    "DEFAULT_RESOLUTION",
    "GEMINI_DEFAULT_BASE_URL",
    "GEMINI_SAFETY_CATEGORIES",
    "OPENAI_DEFAULT_BASE_URL",
    "GITEE_AI_DEFAULT_BASE_URL",
    "JIMENG_DEFAULT_BASE_URL",
    "RESOLUTION_1K_MAP",
    "RESOLUTION_2K_MAP",
    "SUPPORTED_ASPECT_RATIOS",
    "SUPPORTED_RESOLUTIONS",
]
