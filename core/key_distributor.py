"""Integration helpers for astrbot_plugin_newapi_key_distributor."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from astrbot.api import logger

from .config_manager import KeyDistributorSettings


@dataclass(slots=True)
class UserKeyResult:
    api_key: str | None = None
    key_id: str = ""
    message: str = ""


class KeyDistributorResolver:
    """Reads per-user NewAPI keys from the key distributor plugin data file."""

    def __init__(self, settings: KeyDistributorSettings):
        self.settings = settings

    @property
    def enabled(self) -> bool:
        return self.settings.enabled

    @property
    def require_key(self) -> bool:
        return self.settings.require_key

    def _data_path(self) -> Path | None:
        value = (self.settings.data_path or "").strip()
        if not value:
            return None
        return Path(value)

    def _load_data(self) -> dict[str, Any] | None:
        path = self._data_path()
        if path is None:
            return None
        try:
            if not path.exists():
                return None
            data = json.loads(path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else None
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"[ImageGen] 读取 Key 分发数据失败: {path} - {exc}")
            return None

    def resolve(self, qq_id: str) -> UserKeyResult:
        if not self.enabled:
            return UserKeyResult()
        qq_id = str(qq_id or "").strip()
        if not qq_id:
            return UserKeyResult(message="无法识别你的 QQ 号，不能使用个人 Key 生图。")

        data = self._load_data()
        if data is None:
            return UserKeyResult(
                message="未找到 Key 分发数据，请联系管理员检查 key_distributor.data_path。"
            )

        keys = data.get("keys", {})
        if not isinstance(keys, dict):
            return UserKeyResult(message="Key 分发数据格式异常，请联系管理员。")

        items = [
            item
            for item in keys.values()
            if isinstance(item, dict)
            and str(item.get("qq_id") or "") == qq_id
            and item.get("status") == "active"
        ]
        if not items:
            return UserKeyResult(message="你还没有可用的 NewAPI Key，请先申请或联系管理员生成。")

        items.sort(key=lambda item: int(item.get("created_at") or 0), reverse=True)
        for item in items:
            key = str(item.get("key_plain") or "").strip()
            if key:
                return UserKeyResult(api_key=key, key_id=str(item.get("id") or ""))

        return UserKeyResult(
            message="你的 Key 未保存完整内容，请联系管理员开启 store_plain_keys 后重新生成。"
        )
