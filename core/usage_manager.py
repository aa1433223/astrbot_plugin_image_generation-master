"""
用户使用数据管理模块
"""

from __future__ import annotations

import datetime
import json
import os
import time
from typing import TYPE_CHECKING

from astrbot.api import logger

from .constants import USAGE_DATA_RETENTION_DAYS

if TYPE_CHECKING:
    from .config_manager import UsageSettings


class UsageManager:
    """用户使用数据管理器。"""

    def __init__(self, data_dir: str, settings: UsageSettings):
        self._data_dir = data_dir
        self._settings = settings
        self._usage_file = os.path.join(data_dir, "usage.json")
        self._usage_data: dict[str, dict[str, int]] = {}  # {date: {user_id: count}}
        self._user_request_timestamps: dict[str, float] = {}  # 用于频率限制
        self._load_usage_data()

    def update_settings(self, settings: UsageSettings) -> None:
        """更新设置。"""
        self._settings = settings

    def _load_usage_data(self) -> None:
        """加载用户使用数据。"""
        if os.path.exists(self._usage_file):
            try:
                with open(self._usage_file, encoding="utf-8") as f:
                    self._usage_data = json.load(f)

                # 清理旧数据，只保留最近 N 天（由 USAGE_DATA_RETENTION_DAYS 控制）
                today = datetime.date.today()
                keys_to_delete = []
                for date_str in self._usage_data:
                    try:
                        date_obj = datetime.date.fromisoformat(date_str)
                        if (today - date_obj).days > USAGE_DATA_RETENTION_DAYS:
                            keys_to_delete.append(date_str)
                    except ValueError:
                        keys_to_delete.append(date_str)

                if keys_to_delete:
                    for key in keys_to_delete:
                        del self._usage_data[key]
                    self._save_usage_data()
            except Exception as exc:
                logger.error(f"[ImageGen] 加载使用数据失败: {exc}")
                self._usage_data = {}

    def _save_usage_data(self) -> None:
        """保存用户使用数据。"""
        try:
            os.makedirs(os.path.dirname(self._usage_file), exist_ok=True)
            with open(self._usage_file, "w", encoding="utf-8") as f:
                json.dump(self._usage_data, f, ensure_ascii=False, indent=2)
        except Exception as exc:
            logger.error(f"[ImageGen] 保存使用数据失败: {exc}")

    def is_session_blocked(self, user_id: str) -> bool:
        """Check whether the current session UMO is blocked."""
        uid = user_id.strip()
        if not uid:
            return False
        return uid in self._settings.umo_blacklist

    def check_rate_limit(self, user_id: str) -> bool | str:
        """检查用户请求频率限制和每日限制。

        返回:
            - True: 检查通过
            - str: 错误消息
        """
        # 1. 检查频率限制
        if self.is_session_blocked(user_id):
            return self._settings.blacklist_block_message

        if self._settings.rate_limit_seconds > 0:
            now = time.time()
            last_ts = self._user_request_timestamps.get(user_id, 0)
            if now - last_ts < self._settings.rate_limit_seconds:
                remaining = int(self._settings.rate_limit_seconds - (now - last_ts))
                return f"❌ 请求过于频繁，请在 {remaining} 秒后再试"
            self._user_request_timestamps[user_id] = now

        # 2. 检查每日限制
        if self._settings.enable_daily_limit:
            today = datetime.date.today().isoformat()
            if today not in self._usage_data:
                self._usage_data[today] = {}

            count = self._usage_data[today].get(user_id, 0)
            if count >= self._settings.daily_limit_count:
                return f"❌ 您今日的生图额度已用完 ({self._settings.daily_limit_count}次)，请明天再试"

        return True

    def record_usage(self, user_id: str) -> None:
        """记录用户使用次数。"""
        if self._settings.enable_daily_limit:
            today = datetime.date.today().isoformat()
            if today not in self._usage_data:
                self._usage_data[today] = {}
            self._usage_data[today][user_id] = (
                self._usage_data[today].get(user_id, 0) + 1
            )
            self._save_usage_data()

    def get_usage_count(self, user_id: str) -> int:
        """获取用户今日使用次数。"""
        today = datetime.date.today().isoformat()
        return self._usage_data.get(today, {}).get(user_id, 0)

    def get_daily_limit(self) -> int:
        """获取每日限制次数。"""
        return self._settings.daily_limit_count

    def is_daily_limit_enabled(self) -> bool:
        """是否启用每日限制。"""
        return self._settings.enable_daily_limit
