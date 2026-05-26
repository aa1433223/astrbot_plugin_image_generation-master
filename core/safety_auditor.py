"""Safety audit module for prompt and generated images."""

from __future__ import annotations

import json
import re

from astrbot.api import logger
from astrbot.api.star import Context

from .config_manager import ConfigManager


class SafetyAuditor:
    """Audits prompts and generated images."""

    PROMPT_PLACEHOLDER = "{prompt}"

    def __init__(self, context: Context, config_manager: ConfigManager):
        self._context = context
        self._config_manager = config_manager

    async def audit_prompt(
        self, prompt: str, unified_msg_origin: str
    ) -> tuple[bool, str]:
        safety_settings = self._config_manager.safety_audit_settings
        if not safety_settings.enabled:
            return True, ""

        if self._is_umo_whitelisted(unified_msg_origin):
            return True, ""

        settings = safety_settings.prompt_audit

        hit = self._match_blocked_word(prompt, settings.blocked_words)
        if hit:
            return False, f"命中屏蔽词: {hit}"

        if not settings.enable_ai_audit:
            return True, ""

        review_prompt = self._build_review_prompt(
            settings.ai_prompt,
            prompt,
            append_prompt_if_missing_placeholder=True,
        )
        return await self._audit_with_model(
            unified_msg_origin=unified_msg_origin,
            review_prompt=review_prompt,
            provider_id=settings.ai_provider_id,
            image_urls=None,
        )

    async def audit_generated_images(
        self,
        prompt: str,
        image_paths: list[str],
        unified_msg_origin: str,
    ) -> tuple[bool, str]:
        safety_settings = self._config_manager.safety_audit_settings
        if not safety_settings.enabled:
            return True, ""

        if self._is_umo_whitelisted(unified_msg_origin):
            return True, ""

        settings = safety_settings.image_audit
        if not settings.enable_ai_audit:
            return True, ""

        review_prompt = self._build_review_prompt(
            settings.ai_prompt,
            prompt,
            append_prompt_if_missing_placeholder=False,
        )
        return await self._audit_with_model(
            unified_msg_origin=unified_msg_origin,
            review_prompt=review_prompt,
            provider_id=settings.ai_provider_id,
            image_urls=image_paths,
        )

    def _is_umo_whitelisted(self, unified_msg_origin: str) -> bool:
        umo = unified_msg_origin.strip()
        if not umo:
            return False
        return umo in self._config_manager.safety_audit_settings.umo_whitelist

    def _build_review_prompt(
        self,
        template: str,
        prompt: str,
        *,
        append_prompt_if_missing_placeholder: bool,
    ) -> str:
        review_prompt = template.strip()
        prompt = prompt.strip()

        if not review_prompt:
            review_prompt = (
                "请根据输入内容完成安全审核。"
                '仅输出 JSON：{"allow": true/false, "reason": "简短原因"}。'
            )

        if self.PROMPT_PLACEHOLDER in review_prompt:
            return review_prompt.replace(self.PROMPT_PLACEHOLDER, prompt)

        if not append_prompt_if_missing_placeholder or not prompt:
            return review_prompt

        # 兼容旧的提示词审核配置：即使没有占位符，也会附加当前提示词给审核模型。
        return f"{review_prompt}\n\n用户提示词：\n{prompt}"

    async def _audit_with_model(
        self,
        *,
        unified_msg_origin: str,
        review_prompt: str,
        provider_id: str,
        image_urls: list[str] | None,
    ) -> tuple[bool, str]:
        provider = None
        if provider_id:
            provider = self._context.get_provider_by_id(provider_id)
            if not provider:
                logger.warning(
                    f"[ImageGen] 未找到审核 Provider ID: {provider_id}，将回退到当前会话模型"
                )

        if provider is None:
            provider = self._context.get_using_provider(unified_msg_origin)

        if not provider:
            msg = "安全审核异常：未找到可用审核模型"
            logger.warning(f"[ImageGen] {msg}")
            return False, msg

        try:
            response = await provider.text_chat(
                prompt=review_prompt,
                image_urls=image_urls or [],
                persist=False,
            )
            completion_text = (response.completion_text or "").strip()
            decision, reason = self._parse_audit_response(completion_text)
            return decision, reason
        except Exception as exc:
            msg = f"安全审核异常：模型调用失败 - {str(exc)[:180]}"
            logger.warning(f"[ImageGen] {msg}")
            return False, msg

    def _match_blocked_word(self, prompt: str, blocked_words: list[str]) -> str:
        content = prompt.lower()
        for word in blocked_words:
            if word and word.lower() in content:
                return word
        return ""

    def _parse_audit_response(self, text: str) -> tuple[bool, str]:
        if not text:
            return False, "安全审核异常：模型返回为空"

        payload = self._extract_json(text)
        if payload is not None:
            allow = self._to_bool(payload.get("allow"))
            reason = str(payload.get("reason") or "").strip()
            if allow is not None:
                return allow, reason or ("审核通过" if allow else "审核未通过")

        lowered = text.lower()
        reject_tokens = ("reject", "deny", "forbid", "不通过", "违规", "拒绝", "不允许")
        allow_tokens = ("allow", "pass", "safe", "通过", "安全", "允许")

        if any(token in lowered for token in reject_tokens):
            return False, text[:120]
        if any(token in lowered for token in allow_tokens):
            return True, text[:120]

        return False, f"安全审核异常：无法判定审核结果，原始返回: {text[:120]}"

    def _extract_json(self, text: str) -> dict[str, object] | None:
        text = text.strip()
        try:
            obj = json.loads(text)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            pass

        match = re.search(r"\{[\s\S]*\}", text)
        if not match:
            return None
        try:
            obj = json.loads(match.group(0))
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
        return None

    def _to_bool(self, value: object) -> bool | None:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.strip().lower()
            if lowered in {"true", "1", "yes", "allow", "pass", "通过", "允许"}:
                return True
            if lowered in {"false", "0", "no", "reject", "deny", "拒绝", "不通过"}:
                return False
        if isinstance(value, (int, float)):
            return bool(value)
        return None
