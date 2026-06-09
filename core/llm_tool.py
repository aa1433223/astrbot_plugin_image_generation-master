"""
LLM 可调用的图像生成工具模块
"""

from __future__ import annotations

import asyncio
import hashlib
import time
from typing import TYPE_CHECKING, Any

from pydantic import Field
from pydantic.dataclasses import dataclass as pydantic_dataclass

from astrbot.api import logger
from astrbot.core.agent.run_context import ContextWrapper
from astrbot.core.agent.tool import FunctionTool, ToolExecResult
from astrbot.core.astr_agent_context import AstrAgentContext

from .types import ImageCapability

if TYPE_CHECKING:
    pass


@pydantic_dataclass
class ImageGenerationTool(FunctionTool[AstrAgentContext]):
    """LLM 可调用的图像生成工具。"""

    name: str = "generate_image"
    description: str = "使用生图模型生成或修改图片"
    parameters: dict = Field(
        default_factory=lambda: {
            "type": "object",
            "properties": {
                "prompt": {
                    "type": "string",
                    "description": "生图时使用的提示词(要将用户的意图原样传达给模型)。如果用户提到了画图但没有具体描述，请根据上下文推断或提示用户描述。",
                },
                "aspect_ratio": {
                    "type": "string",
                    "description": "图片宽高比。如果不确定，请使用'自动'。",
                    "enum": [
                        "自动",
                        "1:1",
                        "2:3",
                        "3:2",
                        "3:4",
                        "4:3",
                        "4:5",
                        "5:4",
                        "9:16",
                        "16:9",
                        "21:9",
                    ],
                    "default": "自动",
                },
                "resolution": {
                    "type": "string",
                    "description": "图片质量/分辨率。默认使用 '1K'。",
                    "enum": ["1K", "2K", "4K"],
                    "default": "1K",
                },
                "avatar_references": {
                    "type": "array",
                    "description": "当需要使用某人的头像时使用。'self'表示机器人，'sender'表示发送者，也可以直接使用ID做参数。",
                    "items": {"type": "string"},
                },
            },
            "required": ["prompt"],
        }
    )

    # 使用 Any 避免 Pydantic 循环引用问题
    # 实际类型为 ImageGenerationPlugin，在 TYPE_CHECKING 中定义
    plugin: Any = None

    async def call(
        self, context: ContextWrapper[AstrAgentContext], **kwargs: Any
    ) -> ToolExecResult:
        """执行工具调用。"""
        # 获取提示词
        prompt = kwargs.get("prompt", "").strip()
        if not prompt:
            return "❌ 请提供图片生成的提示词"

        plugin = self.plugin
        if not plugin:
            return "❌ 插件未正确初始化 (Plugin instance missing)"

        # 获取事件上下文
        event = None
        if hasattr(context, "context") and isinstance(
            context.context, AstrAgentContext
        ):
            event = context.context.event
        elif isinstance(context, dict):
            event = context.get("event")

        if not event:
            logger.warning(
                f"[ImageGen] 工具调用上下文缺少事件。上下文类型: {type(context)}"
            )
            return "❌ 无法获取当前消息上下文"

        api_key_override, key_error = plugin.resolve_user_api_key_for_event(event)
        if key_error:
            logger.warning(
                f"[ImageGen] 工具调用个人 Key 校验失败: {key_error} "
                f"(用户: {event.unified_msg_origin})"
            )
            return f"❌ {key_error}"

        if (
            not plugin.config_manager.adapter_config
            or (
                not plugin.config_manager.adapter_config.api_keys
                and not api_key_override
            )
        ):
            logger.warning(
                f"[ImageGen] 工具调用失败: 未配置 API Key (用户: {event.unified_msg_origin})"
            )
            return "❌ 未配置 API Key，无法生成图片"

        reservation_result = plugin.usage_manager.reserve_usage(event.unified_msg_origin)
        if reservation_result is not True:
            if isinstance(reservation_result, str) and reservation_result:
                logger.warning(
                    f"[ImageGen] 工具调用触发限制: {reservation_result} "
                    f"(用户: {event.unified_msg_origin})"
                )
            return reservation_result

        prompt_allowed, prompt_reason = await plugin.safety_auditor.audit_prompt(
            prompt, event.unified_msg_origin
        )
        if not prompt_allowed:
            plugin.usage_manager.release_usage_reservation(event.unified_msg_origin)
            return f"❌ 提示词审核未通过: {prompt_reason}"

        # 工具调用同样支持获取上下文参考图（消息/引用/头像）
        images_data = []
        reference_cache_paths: list[str] = []
        avatar_candidate_count = 0
        capabilities = (
            plugin.generator.adapter.get_capabilities()
            if plugin.generator and plugin.generator.adapter
            else ImageCapability.NONE
        )

        try:
            if capabilities & ImageCapability.IMAGE_TO_IMAGE:
                fetched_images = await plugin.image_processor.fetch_images_from_event_with_status(
                    event
                )
                images_data = fetched_images.images
                reference_cache_paths.extend(fetched_images.cache_paths)
                if fetched_images.has_candidates and not images_data:
                    logger.warning(
                        f"[ImageGen] 工具调用检测到参考图候选 {fetched_images.candidate_count} 个，"
                        "但全部缓存失败，已取消图生图任务"
                    )
                    plugin.usage_manager.release_usage_reservation(
                        event.unified_msg_origin
                    )
                    return "❌ 参考图下载失败，已取消生图任务。请重新发送图片后再试。"

                # 处理头像引用参数
                avatar_refs = kwargs.get("avatar_references", [])
                if avatar_refs and isinstance(avatar_refs, list):
                    for ref in avatar_refs:
                        if not isinstance(ref, str):
                            continue
                        ref = ref.strip().lower()
                        user_id = None
                        if ref == "self":
                            user_id = str(event.get_self_id())
                        elif ref == "sender":
                            user_id = str(
                                event.get_sender_id() or event.unified_msg_origin
                            )
                        else:
                            # 简单的 QQ 号校验（可选）
                            if ref.isdigit():
                                user_id = ref

                        if user_id:
                            avatar_candidate_count += 1
                            avatar = await plugin.image_processor.get_avatar_cached(
                                user_id
                            )
                            if avatar:
                                images_data.append((avatar.data, avatar.mime_type))
                                reference_cache_paths.append(avatar.cache_path)
                                logger.info(
                                    f"[ImageGen] 已添加 {user_id} 的头像作为参考图"
                                )
                            else:
                                logger.warning(
                                    f"[ImageGen] 头像参考图缓存失败，已跳过: user_id={user_id}"
                                )
                    if avatar_candidate_count and not images_data:
                        logger.warning(
                            f"[ImageGen] 工具调用头像参考图候选 {avatar_candidate_count} 个，"
                            "但全部缓存失败，已取消图生图任务"
                        )
                        plugin.usage_manager.release_usage_reservation(
                            event.unified_msg_origin
                        )
                        return "❌ 参考图下载失败，已取消生图任务。请重新发送图片后再试。"
        except Exception as e:
            logger.error(f"[ImageGen] 处理参考图失败: {e}", exc_info=True)
            plugin.usage_manager.release_usage_reservation(event.unified_msg_origin)
            return "❌ 参考图处理失败，已取消生图任务。请重新发送图片后再试。"

        # 生成任务 ID
        task_id = hashlib.md5(
            f"{time.time()}{event.unified_msg_origin}".encode()
        ).hexdigest()[:8]
        logger.info(
            f"[ImageGen] 工具调用启动生图任务: task_id={task_id}, "
            f"reference_cache_paths={len(reference_cache_paths)}, "
            f"cache_dir={plugin.image_processor.cache_dir}"
        )

        # 创建后台任务进行生图
        # 添加延迟，让 LLM 有足够时间先发送回复消息，避免图片先于文字到达
        async def _delayed_generate() -> None:
            generation_started = False
            try:
                await asyncio.sleep(3)
                generation_started = True
                await plugin._generate_and_send_image_async(
                    prompt=prompt,
                    images_data=images_data or None,
                    unified_msg_origin=event.unified_msg_origin,
                    aspect_ratio=kwargs.get("aspect_ratio")
                    or plugin.config_manager.default_aspect_ratio,
                    resolution=kwargs.get("resolution")
                    or plugin.config_manager.default_resolution,
                    task_id=task_id,
                    reserved_usage=True,
                    api_key_override=api_key_override,
                )
            finally:
                if not generation_started:
                    plugin.usage_manager.release_usage_reservation(
                        event.unified_msg_origin
                    )

        plugin.create_background_task(_delayed_generate())

        mode = "图生图" if images_data else "文生图"
        return f"✅ 已启动{mode}任务 (任务ID: {task_id})"


def adjust_tool_parameters(
    tool: ImageGenerationTool, capabilities: ImageCapability
) -> None:
    """根据适配器能力动态调整工具参数。"""
    props = tool.parameters["properties"]

    if not (capabilities & ImageCapability.ASPECT_RATIO):
        if "aspect_ratio" in props:
            del props["aspect_ratio"]
            logger.debug("[ImageGen] 适配器不支持宽高比，已从工具参数中移除")

    if not (capabilities & ImageCapability.RESOLUTION):
        if "resolution" in props:
            del props["resolution"]
            logger.debug("[ImageGen] 适配器不支持分辨率，已从工具参数中移除")

    if not (capabilities & ImageCapability.IMAGE_TO_IMAGE):
        if "avatar_references" in props:
            del props["avatar_references"]
            logger.debug("[ImageGen] 适配器不支持参考图，已从工具参数中移除头像引用")
