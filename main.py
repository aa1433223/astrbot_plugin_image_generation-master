"""
AstrBot 图像生成插件主模块

"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Coroutine
from dataclasses import dataclass
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.star.star_tools import StarTools

from .core.config_manager import ConfigManager
from .core.generator import ImageGenerator
from .core.image_processor import ImageProcessor
from .core.llm_tool import ImageGenerationTool, adjust_tool_parameters
from .core.safety_auditor import SafetyAuditor
from .core.task_manager import TaskManager
from .core.types import ErrorCategory, GenerationRequest, ImageCapability, ImageData
from .core.usage_manager import UsageManager
from .core.utils import mask_sensitive, validate_aspect_ratio, validate_resolution


# ---------------------------------------------------------------------------
# 用户友好错误消息映射
# ---------------------------------------------------------------------------
_USER_FRIENDLY_ERRORS: dict[ErrorCategory, str] = {
    ErrorCategory.TIMEOUT: "生成超时，请稍后再试",
    ErrorCategory.RATE_LIMIT: "请求过于频繁，请稍后再试",
    ErrorCategory.SERVER_ERROR: "上游服务暂时不可用，请稍后再试",
    ErrorCategory.AUTH_ERROR: "服务鉴权失败，请联系管理员检查配置",
    ErrorCategory.BALANCE: "API 余额不足，请联系管理员充值",
    ErrorCategory.BAD_REQUEST: "请求参数错误，请调整提示词或参数后重试",
    ErrorCategory.NETWORK: "网络连接失败，请稍后再试",
    ErrorCategory.DOWNLOAD: "图片下载失败，请稍后再试",
    ErrorCategory.UNKNOWN: "生成失败，请稍后再试",
}


def _user_friendly_error(raw_error: str) -> str:
    """将技术性错误信息转换为用户友好的简短提示。"""
    error_lower = (raw_error or "").lower()

    if any(kw in error_lower for kw in ("timeout", "超时", "timed out", "poll_timeout")):
        return _USER_FRIENDLY_ERRORS[ErrorCategory.TIMEOUT]
    if "rate_limit" in error_lower or "rate_limited" in error_lower:
        return _USER_FRIENDLY_ERRORS[ErrorCategory.RATE_LIMIT]
    if "insufficient_balance" in error_lower or "余额" in error_lower:
        return _USER_FRIENDLY_ERRORS[ErrorCategory.BALANCE]
    if any(kw in error_lower for kw in ("auth_error", "auth_required", "鉴权", "model_not_allowed")):
        return _USER_FRIENDLY_ERRORS[ErrorCategory.AUTH_ERROR]
    if "bad_request" in error_lower or "invalid_request" in error_lower or "参数错误" in error_lower:
        return _USER_FRIENDLY_ERRORS[ErrorCategory.BAD_REQUEST]
    if "download_failed" in error_lower or "下载失败" in error_lower:
        return _USER_FRIENDLY_ERRORS[ErrorCategory.DOWNLOAD]
    if any(kw in error_lower for kw in ("connect", "network", "dns", "代理连接")):
        return _USER_FRIENDLY_ERRORS[ErrorCategory.NETWORK]
    if any(kw in error_lower for kw in ("server_error", "服务错误", "502", "503", "500")):
        return _USER_FRIENDLY_ERRORS[ErrorCategory.SERVER_ERROR]

    return _USER_FRIENDLY_ERRORS[ErrorCategory.UNKNOWN]


@dataclass(slots=True)
class InlineGenerationOptions:
    prompt: str
    aspect_ratio: str
    resolution: str
    aspect_ratio_explicit: bool = False
    resolution_explicit: bool = False
    error: str | None = None


@dataclass(slots=True)
class ParsedGenerationCommand:
    prompt: str
    aspect_ratio: str
    resolution: str
    preset_name: str | None = None
    aspect_ratio_explicit: bool = False
    resolution_explicit: bool = False
    error: str | None = None


@dataclass(slots=True)
class ParsedPresetAddCommand:
    name: str
    content: str
    structured: bool = False
    error: str | None = None


@dataclass(slots=True)
class PresetTokenMatch:
    name: str
    start: int
    end: int
    exact: bool = False


def _normalize_inline_aspect_ratio(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().replace("：", ":").replace("；", ":").replace(";", ":")
    return validate_aspect_ratio(normalized)


def _normalize_inline_resolution(value: str | None) -> str | None:
    if not value:
        return None
    normalized = value.strip().upper()
    return validate_resolution(normalized)


def _parse_inline_generation_options(
    text: str,
    default_aspect_ratio: str,
    default_resolution: str,
) -> InlineGenerationOptions:
    """Parse optional leading /生图 parameters without touching prompt body."""

    raw_text = text or ""
    token_matches = list(re.finditer(r"\S+", raw_text))
    tokens = [match.group(0) for match in token_matches]
    aspect_ratio = default_aspect_ratio
    resolution = default_resolution
    aspect_ratio_explicit = False
    resolution_explicit = False
    consumed = 0

    while consumed < len(tokens):
        token = tokens[consumed]
        lowered = token.lower()

        if lowered in {"--ar", "--aspect-ratio", "--aspect_ratio"}:
            if consumed + 1 >= len(tokens):
                return InlineGenerationOptions(
                    prompt=raw_text[token_matches[consumed].start() :].strip(),
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    error=f"参数 {token} 缺少宽高比值",
                )
            value = _normalize_inline_aspect_ratio(tokens[consumed + 1])
            if not value:
                return InlineGenerationOptions(
                    prompt=raw_text[
                        token_matches[consumed + 2].start() :
                    ].lstrip()
                    if consumed + 2 < len(token_matches)
                    else "",
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    error=f"不支持的宽高比: {tokens[consumed + 1]}",
                )
            aspect_ratio = value
            aspect_ratio_explicit = True
            consumed += 2
            continue

        if lowered.startswith("--ar=") or lowered.startswith("--aspect-ratio=") or lowered.startswith("--aspect_ratio="):
            _, _, raw_value = token.partition("=")
            value = _normalize_inline_aspect_ratio(raw_value)
            if not value:
                return InlineGenerationOptions(
                    prompt=raw_text[
                        token_matches[consumed + 1].start() :
                    ].lstrip()
                    if consumed + 1 < len(token_matches)
                    else "",
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    error=f"不支持的宽高比: {raw_value}",
                )
            aspect_ratio = value
            aspect_ratio_explicit = True
            consumed += 1
            continue

        if lowered in {"--res", "--resolution"}:
            if consumed + 1 >= len(tokens):
                return InlineGenerationOptions(
                    prompt=raw_text[token_matches[consumed].start() :].strip(),
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    error=f"参数 {token} 缺少分辨率值",
                )
            value = _normalize_inline_resolution(tokens[consumed + 1])
            if not value:
                return InlineGenerationOptions(
                    prompt=raw_text[
                        token_matches[consumed + 2].start() :
                    ].lstrip()
                    if consumed + 2 < len(token_matches)
                    else "",
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    error=f"不支持的分辨率: {tokens[consumed + 1]}",
                )
            resolution = value
            resolution_explicit = True
            consumed += 2
            continue

        if lowered.startswith("--res=") or lowered.startswith("--resolution="):
            _, _, raw_value = token.partition("=")
            value = _normalize_inline_resolution(raw_value)
            if not value:
                return InlineGenerationOptions(
                    prompt=raw_text[
                        token_matches[consumed + 1].start() :
                    ].lstrip()
                    if consumed + 1 < len(token_matches)
                    else "",
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    error=f"不支持的分辨率: {raw_value}",
                )
            resolution = value
            resolution_explicit = True
            consumed += 1
            continue

        value = _normalize_inline_aspect_ratio(token)
        if value:
            aspect_ratio = value
            aspect_ratio_explicit = True
            consumed += 1
            continue

        value = _normalize_inline_resolution(token)
        if value:
            resolution = value
            resolution_explicit = True
            consumed += 1
            continue

        break

    return InlineGenerationOptions(
        prompt=raw_text[token_matches[consumed].start() :].strip()
        if consumed < len(token_matches)
        else "",
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        aspect_ratio_explicit=aspect_ratio_explicit,
        resolution_explicit=resolution_explicit,
    )


def _find_preset_name(token: str, presets: dict[str, Any]) -> str | None:
    match = _find_preset_token_span([token], presets)
    return match.name if match else None


def _preset_name_tokens(name: str) -> list[str]:
    return re.findall(r"\S+", str(name or ""))


def _find_preset_token_span(
    tokens: list[str],
    presets: dict[str, Any],
) -> PresetTokenMatch | None:
    """Find a preset name in command tokens, preferring the longest span."""

    if not tokens or not presets:
        return None

    best: PresetTokenMatch | None = None
    lowered_tokens = [token.lower() for token in tokens]

    for preset_name in presets:
        preset_tokens = _preset_name_tokens(str(preset_name))
        if not preset_tokens or len(preset_tokens) > len(tokens):
            continue

        preset_lowered = [token.lower() for token in preset_tokens]
        span_len = len(preset_tokens)
        for start in range(0, len(tokens) - span_len + 1):
            end = start + span_len
            exact = tokens[start:end] == preset_tokens
            matched = exact or lowered_tokens[start:end] == preset_lowered
            if not matched:
                continue

            candidate = PresetTokenMatch(
                name=str(preset_name),
                start=start,
                end=end,
                exact=exact,
            )
            if best is None:
                best = candidate
                continue
            best_len = best.end - best.start
            if span_len > best_len:
                best = candidate
            elif span_len == best_len:
                if candidate.exact and not best.exact:
                    best = candidate
                elif candidate.exact == best.exact and candidate.start < best.start:
                    best = candidate

    return best


def _parse_generation_command_text(
    text: str,
    presets: dict[str, Any],
    default_aspect_ratio: str,
    default_resolution: str,
) -> ParsedGenerationCommand:
    """Parse /生图 text while allowing preset and options in flexible order."""

    raw_text = text or ""
    token_matches = list(re.finditer(r"\S+", raw_text))
    tokens = [match.group(0) for match in token_matches]

    preset_match = _find_preset_token_span(tokens, presets)
    if not preset_match:
        parsed_options = _parse_inline_generation_options(
            raw_text,
            default_aspect_ratio,
            default_resolution,
        )
        return ParsedGenerationCommand(
            prompt=parsed_options.prompt,
            aspect_ratio=parsed_options.aspect_ratio,
            resolution=parsed_options.resolution,
            aspect_ratio_explicit=parsed_options.aspect_ratio_explicit,
            resolution_explicit=parsed_options.resolution_explicit,
            error=parsed_options.error,
        )

    aspect_ratio = default_aspect_ratio
    resolution = default_resolution
    aspect_ratio_explicit = False
    resolution_explicit = False
    preset_name: str | None = None
    prompt_parts: list[str] = []
    index = 0

    while index < len(tokens):
        if preset_name is None and preset_match and index == preset_match.start:
            preset_name = preset_match.name
            index = preset_match.end
            continue

        token = tokens[index]
        lowered = token.lower()

        if lowered in {"--ar", "--aspect-ratio", "--aspect_ratio"}:
            if index + 1 >= len(tokens):
                return ParsedGenerationCommand(
                    prompt=" ".join(prompt_parts),
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    preset_name=preset_name,
                    aspect_ratio_explicit=aspect_ratio_explicit,
                    resolution_explicit=resolution_explicit,
                    error=f"参数 {token} 缺少宽高比值",
                )
            raw_value = tokens[index + 1]
            value = _normalize_inline_aspect_ratio(raw_value)
            if not value:
                return ParsedGenerationCommand(
                    prompt=" ".join(prompt_parts + tokens[index + 2 :]),
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    preset_name=preset_name,
                    aspect_ratio_explicit=aspect_ratio_explicit,
                    resolution_explicit=resolution_explicit,
                    error=f"不支持的宽高比: {raw_value}",
                )
            aspect_ratio = value
            aspect_ratio_explicit = True
            index += 2
            continue

        if (
            lowered.startswith("--ar=")
            or lowered.startswith("--aspect-ratio=")
            or lowered.startswith("--aspect_ratio=")
        ):
            _, _, raw_value = token.partition("=")
            value = _normalize_inline_aspect_ratio(raw_value)
            if not value:
                return ParsedGenerationCommand(
                    prompt=" ".join(prompt_parts + tokens[index + 1 :]),
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    preset_name=preset_name,
                    aspect_ratio_explicit=aspect_ratio_explicit,
                    resolution_explicit=resolution_explicit,
                    error=f"不支持的宽高比: {raw_value}",
                )
            aspect_ratio = value
            aspect_ratio_explicit = True
            index += 1
            continue

        if lowered in {"--res", "--resolution"}:
            if index + 1 >= len(tokens):
                return ParsedGenerationCommand(
                    prompt=" ".join(prompt_parts),
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    preset_name=preset_name,
                    aspect_ratio_explicit=aspect_ratio_explicit,
                    resolution_explicit=resolution_explicit,
                    error=f"参数 {token} 缺少分辨率值",
                )
            raw_value = tokens[index + 1]
            value = _normalize_inline_resolution(raw_value)
            if not value:
                return ParsedGenerationCommand(
                    prompt=" ".join(prompt_parts + tokens[index + 2 :]),
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    preset_name=preset_name,
                    aspect_ratio_explicit=aspect_ratio_explicit,
                    resolution_explicit=resolution_explicit,
                    error=f"不支持的分辨率: {raw_value}",
                )
            resolution = value
            resolution_explicit = True
            index += 2
            continue

        if lowered.startswith("--res=") or lowered.startswith("--resolution="):
            _, _, raw_value = token.partition("=")
            value = _normalize_inline_resolution(raw_value)
            if not value:
                return ParsedGenerationCommand(
                    prompt=" ".join(prompt_parts + tokens[index + 1 :]),
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    preset_name=preset_name,
                    aspect_ratio_explicit=aspect_ratio_explicit,
                    resolution_explicit=resolution_explicit,
                    error=f"不支持的分辨率: {raw_value}",
                )
            resolution = value
            resolution_explicit = True
            index += 1
            continue

        value = _normalize_inline_aspect_ratio(token)
        if value:
            aspect_ratio = value
            aspect_ratio_explicit = True
            index += 1
            continue

        value = _normalize_inline_resolution(token)
        if value:
            resolution = value
            resolution_explicit = True
            index += 1
            continue

        prompt_parts.append(token)
        index += 1

    return ParsedGenerationCommand(
        prompt=" ".join(prompt_parts).strip(),
        aspect_ratio=aspect_ratio,
        resolution=resolution,
        preset_name=preset_name,
        aspect_ratio_explicit=aspect_ratio_explicit,
        resolution_explicit=resolution_explicit,
    )


def _parse_preset_payload(
    content: str,
    default_aspect_ratio: str,
    default_resolution: str,
) -> tuple[str, str, str]:
    prompt = content
    aspect_ratio = default_aspect_ratio
    resolution = default_resolution
    try:
        if isinstance(content, str) and content.strip().startswith("{"):
            preset_data = json.loads(content)
            if isinstance(preset_data, dict):
                prompt = str(preset_data.get("prompt", ""))
                aspect_ratio = str(preset_data.get("aspect_ratio", aspect_ratio))
                resolution = str(preset_data.get("resolution", resolution))
    except json.JSONDecodeError:
        prompt = content
    return prompt, aspect_ratio, resolution


def _format_preset_content_for_display(content: Any) -> str:
    text = str(content)
    try:
        if text.strip().startswith("{"):
            data = json.loads(text)
            if isinstance(data, dict):
                prompt = str(data.get("prompt", ""))
                aspect_ratio = data.get("aspect_ratio")
                resolution = data.get("resolution")
                tags = "".join(
                    f"[{value}]"
                    for value in (aspect_ratio, resolution)
                    if value
                )
                preview = prompt[:20] + "..." if len(prompt) > 20 else prompt
                return f"{tags}: {preview}" if tags else preview
    except json.JSONDecodeError:
        pass
    return text[:20] + "..." if len(text) > 20 else text


def _parse_preset_add_command(
    payload: str,
    default_aspect_ratio: str,
    default_resolution: str,
) -> ParsedPresetAddCommand:
    raw_payload = (payload or "").strip()
    if not raw_payload:
        return ParsedPresetAddCommand("", "", error="格式错误: /预设 添加 名称:内容")

    first_token = raw_payload.split(maxsplit=1)[0]
    if ":" in first_token:
        name, content = raw_payload.split(":", 1)
        if name.strip() and content.strip():
            return ParsedPresetAddCommand(name.strip(), content.strip())
        return ParsedPresetAddCommand("", "", error="格式错误: /预设 添加 名称:内容")

    parts = raw_payload.split(maxsplit=1)
    if len(parts) < 2:
        return ParsedPresetAddCommand("", "", error="格式错误: /预设 添加 名称:内容")

    name, rest = parts[0].strip(), parts[1].strip()
    parsed_options = _parse_inline_generation_options(
        rest,
        default_aspect_ratio,
        default_resolution,
    )
    if parsed_options.error:
        return ParsedPresetAddCommand(name, "", error=parsed_options.error)
    if not parsed_options.prompt:
        return ParsedPresetAddCommand("", "", error="格式错误: /预设 添加 名称 9:16 4K 内容")

    content = json.dumps(
        {
            "prompt": parsed_options.prompt,
            "aspect_ratio": parsed_options.aspect_ratio,
            "resolution": parsed_options.resolution,
        },
        ensure_ascii=False,
    )
    return ParsedPresetAddCommand(name, content, structured=True)


class ImageGenerationPlugin(Star):
    """图像生成插件主类"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.context = context

        # 数据目录配置
        self.data_dir = StarTools.get_data_dir()
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 初始化配置管理器
        self.config_manager = ConfigManager(config, data_dir=self.data_dir)

        # 初始化使用数据管理器
        self.usage_manager = UsageManager(
            str(self.data_dir), self.config_manager.usage_settings
        )

        # 初始化图片处理器
        self.image_processor = ImageProcessor(
            str(self.cache_dir),
            self.config_manager.usage_settings.max_image_size_mb,
            self.config_manager.cache_settings.max_cache_count,
        )

        # 初始化任务管理器
        self.task_manager = TaskManager()

        # 初始化安全审核器
        self.safety_auditor = SafetyAuditor(self.context, self.config_manager)

        # 初始化生成器
        self.generator: ImageGenerator | None = None
        self.semaphore: asyncio.Semaphore | None = None
        self._jimeng_token_adapter: Any | None = None
        self._handled_command_events: dict[str, float] = {}

    # ---------------------- 生命周期 ----------------------

    async def initialize(self):
        """插件加载时调用"""
        logger.info(
            f"[ImageGen] 数据目录: {self.data_dir}, 缓存目录: {self.cache_dir}"
        )

        if self.config_manager.adapter_config:
            self.generator = ImageGenerator(self.config_manager.adapter_config)
            self.semaphore = asyncio.Semaphore(self.config_manager.max_concurrent_tasks)
        else:
            logger.error("[ImageGen] 适配器配置加载失败，插件未初始化")

        logger.info(
            "[ImageGen] 开关状态: "
            f"enable_llm_tool={self.config_manager.enable_llm_tool}, "
            f"safety_audit.enabled={self.config_manager.safety_audit_settings.enabled}"
        )

        # 注册 LLM 工具
        if self.config_manager.enable_llm_tool and self.generator:
            self._register_or_refresh_llm_tool()

        # 配置定时任务
        self._setup_tasks()

        # 执行启动任务（在后台异步执行）
        self.task_manager.create_task(self.task_manager.run_startup_tasks())

        logger.info(
            f"[ImageGen] 插件加载完成，模型: {self.config_manager.adapter_config.model if self.config_manager.adapter_config else '未知'}"
        )

    async def terminate(self):
        """插件卸载时调用"""
        try:
            await self.task_manager.cancel_all()
            if self.generator:
                await self.generator.close()
            if self._jimeng_token_adapter:
                await self._jimeng_token_adapter.close()
                self._jimeng_token_adapter = None
            logger.info("[ImageGen] 插件已卸载")
        except Exception as exc:
            logger.error(f"[ImageGen] 卸载清理出错: {exc}")

    # ---------------------- 内部工具 ----------------------

    def _setup_tasks(self) -> None:
        """配置并启动定时任务。"""
        # 1. 缓存清理任务
        self.task_manager.start_loop_task(
            name="cache_cleanup",
            coro_func=self.image_processor.cleanup_cache,
            interval_seconds=self.config_manager.cache_settings.cleanup_interval_hours
            * 3600,
            run_immediately=True,
        )

        # 2. Jimeng2API 自动领积分任务
        self._setup_jimeng_token_task()

    def _setup_jimeng_token_task(self) -> None:
        """配置即梦自动领积分任务。

        该任务会：
        1. 在插件启动时执行一次（通过启动任务）
        2. 每天日期变更时自动执行（通过每日任务）

        注意：只要配置中包含即梦渠道，就会启用该任务，
        无论当前使用的是哪个渠道。
        """
        from .adapter.jimeng2api_adapter import Jimeng2APIAdapter
        from .core.types import AdapterType

        # 检查配置中是否包含即梦渠道（而非检查当前适配器）
        jimeng_config = self.config_manager.get_provider_config(AdapterType.JIMENG2API)
        if not jimeng_config:
            return

        # 创建专门用于任务的即梦适配器实例
        self._jimeng_token_adapter = Jimeng2APIAdapter(jimeng_config)

        # 1. 注册为启动任务，插件启动时执行一次
        self.task_manager.register_startup_task(
            name="jimeng_token_receive",
            coro_func=self._jimeng_token_adapter.receive_token,
        )

        # 2. 注册为每日任务，日期变更时执行
        self.task_manager.start_daily_task(
            name="jimeng_token_receive",
            coro_func=self._jimeng_token_adapter.receive_token,
            check_interval_seconds=300,  # 每5分钟检查一次日期变更
            run_immediately=False,  # 启动任务已处理，无需重复执行
        )
        logger.info("[ImageGen] 已配置即梦2API自动领积分任务（启动时+每日）")

    def _claim_command_event(self, event: AstrMessageEvent, intent: str) -> bool:
        """同一条指令只处理一次，避免兜底监听和标准 command 双触发。"""
        try:
            if getattr(event, "_imagegen_command_claimed", False):
                return False
            setattr(event, "_imagegen_command_claimed", True)
        except Exception:
            pass

        now = time.time()
        expired = [
            key
            for key, ts in self._handled_command_events.items()
            if now - ts > 30
        ]
        for key in expired:
            self._handled_command_events.pop(key, None)

        message_obj = getattr(event, "message_obj", None)
        message_id = (
            getattr(message_obj, "message_id", None)
            or getattr(message_obj, "id", None)
            or getattr(event, "message_id", None)
        )
        event_key = (
            f"{intent}:{event.unified_msg_origin}:{message_id}"
            if message_id is not None
            else f"{intent}:{id(event)}"
        )
        if event_key in self._handled_command_events:
            return False

        self._handled_command_events[event_key] = now
        return True

    def _adjust_tool_parameters(self, tool: ImageGenerationTool) -> None:
        """根据适配器能力动态调整工具参数。"""
        if not self.generator or not self.generator.adapter:
            return
        capabilities = self.generator.adapter.get_capabilities()
        adjust_tool_parameters(tool, capabilities)

    def _register_or_refresh_llm_tool(self) -> None:
        """注册或刷新生图 LLM 工具。"""
        if not self.config_manager.enable_llm_tool or not self.generator:
            return

        tool = ImageGenerationTool(plugin=self)
        self._adjust_tool_parameters(tool)
        self.context.add_llm_tools(tool)
        logger.info("[ImageGen] 已注册图像生成工具")

    def create_background_task(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
        """创建后台任务并添加到管理器中。"""
        return self.task_manager.create_task(coro)

    def get_llm_provider(self):
        """获取独立 LLM Provider（如已配置）。

        参考 hextech 插件的 context.get_provider_by_id() 模式，
        优先使用配置的 llm_provider_id，回退到系统首个可用 Provider。

        Returns:
            LLM Provider 实例，如果无法获取则返回 None。
        """
        provider_id = self.config_manager.llm_provider_id
        if provider_id:
            provider = self.context.get_provider_by_id(provider_id)
            if provider:
                return provider
            logger.warning(
                f"[ImageGen] 配置的 LLM Provider '{provider_id}' 不可用，"
                "尝试回退到首个可用 Provider"
            )
        # 回退：使用系统首个可用 Provider
        if hasattr(self.context, "get_all_providers"):
            providers = self.context.get_all_providers()
            if providers:
                return providers[0]
        return None

    # ---------------------- 核心生图逻辑 ----------------------

    async def _generate_and_send_image_async(
        self,
        prompt: str,
        unified_msg_origin: str,
        images_data: list[tuple[bytes, str]] | None = None,
        aspect_ratio: str = "1:1",
        resolution: str = "1K",
        task_id: str | None = None,
        reserved_usage: bool = False,
    ) -> None:
        """异步生成图片并发送。"""
        reservation_transferred = False
        try:
            if not self.generator or not self.generator.adapter:
                return

            capabilities = self.generator.adapter.get_capabilities()

            # 检查并清理不支持的参数
            if not (capabilities & ImageCapability.IMAGE_TO_IMAGE) and images_data:
                logger.warning(
                    f"[ImageGen] 当前适配器不支持参考图，已忽略 {len(images_data)} 张图片"
                )
                images_data = None

            if not (capabilities & ImageCapability.ASPECT_RATIO) and aspect_ratio != "自动":
                logger.info(
                    f"[ImageGen] 当前适配器不支持指定比例，已忽略参数: {aspect_ratio}"
                )
                aspect_ratio = "自动"

            if not (capabilities & ImageCapability.RESOLUTION) and resolution != "1K":
                logger.info(
                    f"[ImageGen] 当前适配器不支持指定分辨率，已忽略参数: {resolution}"
                )
                resolution = "1K"

            if not task_id:
                task_id = hashlib.md5(
                    f"{time.time()}{unified_msg_origin}".encode()
                ).hexdigest()[:8]

            final_ar = validate_aspect_ratio(aspect_ratio) or None
            if final_ar == "自动":
                final_ar = None
            final_res = validate_resolution(resolution)

            images: list[ImageData] = []
            if images_data:
                for data, mime in images_data:
                    images.append(ImageData(data=data, mime_type=mime))

            # 使用信号量控制并发
            if self.semaphore is None:
                reservation_transferred = True
                await self._do_generate_and_send(
                    prompt, unified_msg_origin, images, final_ar, final_res, task_id,
                    reserved_usage=reserved_usage,
                )
                return

            async with self.semaphore:
                reservation_transferred = True
                await self._do_generate_and_send(
                    prompt, unified_msg_origin, images, final_ar, final_res, task_id,
                    reserved_usage=reserved_usage,
                )
        finally:
            if reserved_usage and not reservation_transferred:
                self.usage_manager.release_usage_reservation(unified_msg_origin)

    async def _do_generate_and_send(
        self,
        prompt: str,
        unified_msg_origin: str,
        images: list[ImageData],
        aspect_ratio: str | None,
        resolution: str | None,
        task_id: str,
        *,
        reserved_usage: bool = False,
    ) -> None:
        """执行生成逻辑并发送结果。"""
        start_time = time.time()
        usage_committed = False
        try:
            if not self.generator:
                logger.warning("[ImageGen] 生成器未初始化，跳过生成请求")
                return
            result = await self.generator.generate(
                GenerationRequest(
                    prompt=prompt,
                    images=images,
                    aspect_ratio=aspect_ratio,
                    resolution=resolution,
                    task_id=task_id,
                )
            )
            end_time = time.time()
            duration = end_time - start_time

            if result.error:
                # 详细错误仅记录到日志，用户只看到简短友好提示
                logger.error(
                    f"[ImageGen] 任务 {task_id} 生成失败，耗时: {duration:.2f}s, 错误: {result.error}"
                )
                friendly_msg = _user_friendly_error(result.error)
                await self.context.send_message(
                    unified_msg_origin,
                    MessageChain().message(f"❌ {friendly_msg}"),
                )
                return

            logger.info(
                f"[ImageGen] 任务 {task_id} 生成成功，耗时: {duration:.2f}s, 图片数量: {len(result.images) if result.images else 0}"
            )

            if not result.images:
                return

            generated_file_paths: list[str] = []
            for img_bytes in result.images:
                file_path = self.image_processor.save_generated_image(task_id, img_bytes)
                if file_path:
                    generated_file_paths.append(file_path)

            if not generated_file_paths:
                logger.warning(f"[ImageGen] 任务 {task_id} 未能保存任何生成图片")
                return

            # 生图后图片审核
            image_allowed, image_reason = await self.safety_auditor.audit_generated_images(
                prompt=prompt,
                image_paths=generated_file_paths,
                unified_msg_origin=unified_msg_origin,
            )
            if not image_allowed:
                logger.warning(f"[ImageGen] 任务 {task_id} 图片审核未通过: {image_reason}")
                await self.context.send_message(
                    unified_msg_origin,
                    MessageChain().message(f"❌ 图片内容审核未通过: {image_reason}"),
                )
                return

            # 记录使用次数
            self.usage_manager.record_usage(
                unified_msg_origin,
                reserved=reserved_usage,
            )
            usage_committed = True

            chain = MessageChain()
            for file_path in generated_file_paths:
                chain.file_image(file_path)

            info_parts = []
            if self.config_manager.show_generation_info:
                info_parts.append(
                    f"✨ 生成成功！\n📊 耗时: {duration:.2f}s\n🖼️ 数量: {len(generated_file_paths)}张"
                )

            if self.config_manager.show_model_info and self.config_manager.adapter_config:
                info_parts.append(
                    f"🤖 模型: {self.config_manager.adapter_config.name}/{self.config_manager.adapter_config.model}"
                )

            if self.usage_manager.is_daily_limit_enabled():
                count = self.usage_manager.get_usage_count(unified_msg_origin)
                info_parts.append(
                    f"📅 今日用量: {count}/{self.usage_manager.get_daily_limit()}"
                )

            if info_parts:
                chain.message("\n" + "\n".join(info_parts))

            await self.context.send_message(unified_msg_origin, chain)
        finally:
            if reserved_usage and not usage_committed:
                self.usage_manager.release_usage_reservation(unified_msg_origin)

    # ---------------------- 指令处理 ----------------------

    @staticmethod
    def _detect_intent(raw_text: str) -> tuple[str | None, str]:
        """识别未进入标准 command 过滤器的斜杠指令。"""
        text = (raw_text or "").strip()
        if not text.startswith("/"):
            return None, ""

        for intent, commands in (
            ("model", ("生图模型",)),
            ("preset", ("预设",)),
            ("generate", ("生图", "画图", "生成图")),
        ):
            for command in commands:
                prefix = f"/{command}"
                if text == prefix:
                    return intent, ""
                if text.startswith(prefix) and text[len(prefix)].isspace():
                    return intent, text[len(prefix) :].strip()
        return None, ""

    @filter.event_message_type(filter.EventMessageType.ALL)
    async def listen_plain_messages(self, event: AstrMessageEvent):
        """兜底监听原始 / 指令，绕过部分环境的 wake/command 限制。"""
        raw_text = (event.message_str or "").strip()
        intent, arg = self._detect_intent(raw_text)
        if not intent:
            return

        if not self._claim_command_event(event, intent):
            return
        event.stop_event()

        logger.info(f"[ImageGen] passive listener matched intent: {intent}")
        if intent == "generate":
            async for result in self._handle_generate_image(
                event,
                prompt_text=arg,
            ):
                yield result
        elif intent == "model":
            async for result in self._handle_model_command(
                event,
                model_index=arg,
            ):
                yield result
        elif intent == "preset":
            async for result in self._handle_preset_command(
                event,
                cmd_text=arg,
            ):
                yield result

    @filter.command("生图", alias=["画图", "生成图"])
    async def generate_image_command(self, event: AstrMessageEvent):
        """处理生图指令。"""
        if not self._claim_command_event(event, "generate"):
            return
        async for result in self._handle_generate_image(event):
            yield result

    async def _handle_generate_image(
        self,
        event: AstrMessageEvent,
        prompt_text: str | None = None,
    ):
        """处理生图逻辑，供标准 command 和被动监听共用。"""
        user_id = event.unified_msg_origin

        reservation_result = self.usage_manager.reserve_usage(user_id)
        if reservation_result is not True:
            if isinstance(reservation_result, str) and reservation_result:
                yield event.plain_result(reservation_result)
            return

        masked_uid = mask_sensitive(user_id)

        user_input = (event.message_str or "").strip()
        logger.info(f"[ImageGen] 收到生图指令 - 用户: {masked_uid}, 输入: {user_input}")

        if prompt_text is None:
            cmd_parts = user_input.split(maxsplit=1)
            if not cmd_parts:
                self.usage_manager.release_usage_reservation(user_id)
                return
            prompt = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
        else:
            prompt = prompt_text.strip()
        aspect_ratio = self.config_manager.default_aspect_ratio
        resolution = self.config_manager.default_resolution

        parsed_command = _parse_generation_command_text(
            prompt,
            self.config_manager.presets,
            aspect_ratio,
            resolution,
        )
        if parsed_command.error:
            self.usage_manager.release_usage_reservation(user_id)
            yield event.plain_result(f"❌ {parsed_command.error}")
            return

        matched_preset = parsed_command.preset_name
        extra_content = parsed_command.prompt
        aspect_ratio = parsed_command.aspect_ratio
        resolution = parsed_command.resolution
        preset_ar_explicit = False
        preset_res_explicit = False

        if matched_preset:
            logger.info(f"[ImageGen] 命中预设: {matched_preset}")
            preset_content = self.config_manager.presets[matched_preset]
            preset_prompt, preset_ar, preset_res = _parse_preset_payload(
                str(preset_content),
                self.config_manager.default_aspect_ratio,
                self.config_manager.default_resolution,
            )
            prompt = preset_prompt

            # 检测预设是否显式定义了 aspect_ratio / resolution
            try:
                raw_str = str(preset_content).strip()
                if raw_str.startswith("{"):
                    preset_data = json.loads(raw_str)
                    if isinstance(preset_data, dict):
                        preset_ar_explicit = "aspect_ratio" in preset_data
                        preset_res_explicit = "resolution" in preset_data
            except json.JSONDecodeError:
                pass

            if not parsed_command.aspect_ratio_explicit and preset_ar:
                aspect_ratio = preset_ar
            if not parsed_command.resolution_explicit and preset_res:
                resolution = preset_res
            if extra_content:
                prompt = f"{prompt} {extra_content}"
        else:
            prompt = extra_content

        if not prompt:
            self.usage_manager.release_usage_reservation(user_id)
            yield event.plain_result("❌ 请提供图片生成的提示词或预设名称！")
            return

        prompt_audit_start = time.time()
        prompt_allowed, prompt_reason = await self.safety_auditor.audit_prompt(
            prompt, event.unified_msg_origin
        )
        prompt_audit_duration = time.time() - prompt_audit_start
        logger.info(
            f"[ImageGen] 提示词审核阶段耗时: {prompt_audit_duration:.2f}s"
        )
        if self.config_manager.show_diagnostic_timing:
            yield event.plain_result(
                f"[诊断] 提示词审核耗时: {prompt_audit_duration:.2f}s"
            )
        if not prompt_allowed:
            self.usage_manager.release_usage_reservation(user_id)
            yield event.plain_result(f"❌ 提示词审核未通过: {prompt_reason}")
            return

        # 获取参考图
        images_data = []
        reference_cache_paths: list[str] = []
        fetch_images_start = time.time()
        if (
            self.generator
            and self.generator.adapter
            and (
                self.generator.adapter.get_capabilities()
                & ImageCapability.IMAGE_TO_IMAGE
            )
        ):
            fetched_images = await self.image_processor.fetch_images_from_event_with_status(
                event
            )
            images_data = fetched_images.images
            reference_cache_paths = fetched_images.cache_paths
            fetch_images_duration = time.time() - fetch_images_start
            logger.info(
                f"[ImageGen] 参考图提取阶段耗时: {fetch_images_duration:.2f}s"
            )
            if self.config_manager.show_diagnostic_timing:
                yield event.plain_result(
                    f"[诊断] 参考图提取耗时: {fetch_images_duration:.2f}s"
                )
            if fetched_images.has_candidates and not images_data:
                logger.warning(
                    f"[ImageGen] 参考图候选 {fetched_images.candidate_count} 个，"
                    "但全部缓存失败，已取消本次图生图任务"
                )
                self.usage_manager.release_usage_reservation(user_id)
                yield event.plain_result(
                    "❌ 参考图下载失败，已取消生图任务。请重新发送图片后再试。"
                )
                return

        # 有参考图时，如果用户和预设都没显式指定比例/分辨率，使用图生图的独立默认值
        # 优先级：用户显式指定 > 预设显式定义 > 图生图默认值 > 文生图默认值
        if images_data:
            if not parsed_command.aspect_ratio_explicit and not preset_ar_explicit:
                aspect_ratio = self.config_manager.i2i_default_aspect_ratio
            if not parsed_command.resolution_explicit and not preset_res_explicit:
                resolution = self.config_manager.i2i_default_resolution

        msg = "已开始生图任务"
        msg += f"[{aspect_ratio}][{resolution}]"
        if images_data:
            msg += f"[{len(images_data)}张参考图]"
            if reference_cache_paths:
                msg += f"[缓存成功{len(reference_cache_paths)}张]"
        if matched_preset:
            msg += f"[预设: {matched_preset}]"
        yield event.plain_result(msg)

        task_id = hashlib.md5(f"{time.time()}{user_id}".encode()).hexdigest()[:8]
        logger.info(
            f"[ImageGen] 启动生图任务: task_id={task_id}, "
            f"aspect_ratio={aspect_ratio}, resolution={resolution}, "
            f"reference_cache_paths={len(reference_cache_paths)}, "
            f"cache_dir={self.cache_dir}"
        )

        self.create_background_task(
            self._generate_and_send_image_async(
                prompt=prompt,
                images_data=images_data or None,
                unified_msg_origin=event.unified_msg_origin,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                task_id=task_id,
                reserved_usage=True,
            )
        )

    @filter.command("生图模型")
    async def model_command(self, event: AstrMessageEvent, model_index: str = ""):
        """切换生图模型。"""
        if not self._claim_command_event(event, "model"):
            return
        async for result in self._handle_model_command(event, model_index=model_index):
            yield result

    async def _handle_model_command(
        self,
        event: AstrMessageEvent,
        model_index: str = "",
    ):
        """切换生图模型，供标准 command 和被动监听共用。"""
        if not self.config_manager.adapter_config:
            yield event.plain_result("❌ 适配器未初始化")
            return

        model_index = (model_index or "").strip()
        models = self.config_manager.adapter_config.available_models or []

        if not model_index:
            lines = ["📋 可用模型列表:"]
            current_model_full = f"{self.config_manager.adapter_config.name}/{self.config_manager.adapter_config.model}"
            for idx, model in enumerate(models, 1):
                marker = " ✓" if model == current_model_full else ""
                lines.append(f"{idx}. {model}{marker}")
            lines.append(f"\n当前使用: {current_model_full}")
            yield event.plain_result("\n".join(lines))
            return

        try:
            index = int(model_index) - 1
            if 0 <= index < len(models):
                raw_model = models[index]  # "供应商名称/模型名称"

                # 更新配置并重新加载
                self.config_manager.save_model_setting(raw_model)
                self.config_manager.reload()

                if self.generator:
                    await self.generator.update_adapter(
                        self.config_manager.adapter_config
                    )
                if self.config_manager.enable_llm_tool:
                    self._register_or_refresh_llm_tool()

                yield event.plain_result(f"✅ 模型已切换: {raw_model}")
            else:
                yield event.plain_result("❌ 无效的序号")
        except ValueError:
            yield event.plain_result("❌ 请输入有效的数字序号")

    @filter.command("预设")
    async def preset_command(self, event: AstrMessageEvent):
        """管理生图预设。"""
        if not self._claim_command_event(event, "preset"):
            return
        async for result in self._handle_preset_command(event):
            yield result

    async def _handle_preset_command(
        self,
        event: AstrMessageEvent,
        cmd_text: str | None = None,
    ):
        """管理生图预设，供标准 command 和被动监听共用。"""
        user_id = event.unified_msg_origin
        masked_uid = mask_sensitive(user_id)
        message_str = (event.message_str or "").strip()
        logger.info(
            f"[ImageGen] 收到预设指令 - 用户: {masked_uid}, 内容: {message_str}"
        )

        if cmd_text is None:
            parts = message_str.split(maxsplit=1)
            cmd_text = parts[1].strip() if len(parts) > 1 else ""
        else:
            cmd_text = cmd_text.strip()

        if not cmd_text:
            if not self.config_manager.presets:
                yield event.plain_result("📋 当前没有预设")
                return
            preset_list = ["📋 预设列表:"]
            for idx, (name, prompt) in enumerate(
                self.config_manager.presets.items(), 1
            ):
                display = _format_preset_content_for_display(prompt)
                preset_list.append(f"{idx}. {name}: {display}")
            yield event.plain_result("\n".join(preset_list))
            return

        if cmd_text.startswith("添加 "):
            parsed_add = _parse_preset_add_command(
                cmd_text[3:],
                self.config_manager.default_aspect_ratio,
                self.config_manager.default_resolution,
            )
            if parsed_add.error:
                yield event.plain_result(f"❌ {parsed_add.error}")
                return
            self.config_manager.save_preset(parsed_add.name, parsed_add.content)
            suffix = " [结构化]" if parsed_add.structured else ""
            yield event.plain_result(f"✅ 预设已添加: {parsed_add.name}{suffix}")
        elif cmd_text.startswith("删除 "):
            name = cmd_text[3:].strip()
            if self.config_manager.delete_preset(name):
                yield event.plain_result(f"✅ 预设已删除: {name}")
            else:
                yield event.plain_result(f"❌ 预设不存在: {name}")
