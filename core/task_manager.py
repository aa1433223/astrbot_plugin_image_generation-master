from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable, Coroutine
from datetime import datetime
from typing import Any

from astrbot.api import logger


class TaskManager:
    """统一的任务管理器，管理插件的后台任务和定时任务。"""

    def __init__(self):
        self.background_tasks: set[asyncio.Task] = set()
        self._loop_tasks: dict[str, asyncio.Task] = {}
        self._daily_tasks: dict[str, asyncio.Task] = {}
        self._last_run_dates: dict[str, str] = {}  # 记录每日任务上次执行的日期
        self._startup_tasks: list[Callable[[], Coroutine[Any, Any, Any]]] = []
        self._startup_completed: bool = False

    def create_task(
        self, coro: Coroutine[Any, Any, Any], name: str | None = None
    ) -> asyncio.Task:
        """创建一个普通的后台任务。"""
        task = asyncio.create_task(coro)
        if name:
            task.set_name(name)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        return task

    def start_loop_task(
        self,
        name: str,
        coro_func: Callable[[], Coroutine[Any, Any, Any]],
        interval_seconds: float,
        run_immediately: bool = True,
    ) -> None:
        """启动一个周期性的定时任务。

        Args:
            name: 任务名称，用于唯一标识和日志记录。
            coro_func: 返回协程的函数（任务的主逻辑）。
            interval_seconds: 执行间隔（秒）。
            run_immediately: 是否在启动时立即执行一次。
        """
        if name in self._loop_tasks:
            self.stop_loop_task(name)

        async def _loop():
            if run_immediately:
                try:
                    await coro_func()
                except Exception as e:
                    logger.error(
                        f"[ImageGen] [TaskManager] 定时任务 {name} 初始执行失败: {e}",
                        exc_info=True,
                    )

            while True:
                try:
                    await asyncio.sleep(interval_seconds)
                    await coro_func()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(
                        f"[ImageGen] [TaskManager] 定时任务 {name} 执行出错: {e}",
                        exc_info=True,
                    )

        task = asyncio.create_task(_loop(), name=f"loop_{name}")
        self._loop_tasks[name] = task
        self.background_tasks.add(task)
        task.add_done_callback(functools.partial(self._on_loop_task_done, name))
        logger.info(
            f"[ImageGen] [TaskManager] 定时任务 {name} 已启动 (间隔: {interval_seconds}s)"
        )

    def stop_loop_task(self, name: str) -> None:
        """停止指定的定时任务。"""
        if task := self._loop_tasks.pop(name, None):
            if not task.done():
                task.cancel()
            logger.info(f"[ImageGen] [TaskManager] 定时任务 {name} 已停止")

    def _on_loop_task_done(self, name: str, task: asyncio.Task) -> None:
        """定时任务结束时的回调。"""
        self.background_tasks.discard(task)
        self._loop_tasks.pop(name, None)

    def register_startup_task(
        self,
        name: str,
        coro_func: Callable[[], Coroutine[Any, Any, Any]],
    ) -> None:
        """注册一个启动时执行的任务。

        Args:
            name: 任务名称，用于日志记录。
            coro_func: 返回协程的函数（任务的主逻辑）。
        """
        self._startup_tasks.append((name, coro_func))
        logger.info(f"[ImageGen] [TaskManager] 已注册启动任务: {name}")

    async def run_startup_tasks(self) -> None:
        """执行所有注册的启动任务。

        此方法应在插件初始化完成后调用一次。
        """
        if self._startup_completed:
            logger.warning("[ImageGen] [TaskManager] 启动任务已执行过，跳过重复执行")
            return

        if not self._startup_tasks:
            logger.info("[ImageGen] [TaskManager] 没有注册的启动任务")
            self._startup_completed = True
            return

        logger.info(
            f"[ImageGen] [TaskManager] 开始执行 {len(self._startup_tasks)} 个启动任务"
        )

        for name, coro_func in self._startup_tasks:
            try:
                logger.info(f"[ImageGen] [TaskManager] 执行启动任务: {name}")
                await coro_func()
                logger.info(f"[ImageGen] [TaskManager] 启动任务 {name} 执行完成")
            except Exception as e:
                logger.error(
                    f"[ImageGen] [TaskManager] 启动任务 {name} 执行失败: {e}",
                    exc_info=True,
                )

        self._startup_completed = True
        logger.info("[ImageGen] [TaskManager] 所有启动任务执行完毕")

    def start_daily_task(
        self,
        name: str,
        coro_func: Callable[[], Coroutine[Any, Any, Any]],
        check_interval_seconds: float = 60.0,
        run_immediately: bool = False,
    ) -> None:
        """启动一个每日任务，在日期变更时执行。

        Args:
            name: 任务名称，用于唯一标识和日志记录。
            coro_func: 返回协程的函数（任务的主逻辑）。
            check_interval_seconds: 检查日期变更的间隔（秒），默认 60 秒。
            run_immediately: 是否在启动时立即执行一次（无论日期）。
        """
        if name in self._daily_tasks:
            self.stop_daily_task(name)

        async def _daily_loop():
            # 初始化上次执行日期
            if run_immediately:
                try:
                    await coro_func()
                    self._last_run_dates[name] = datetime.now().strftime("%Y-%m-%d")
                    logger.info(
                        f"[ImageGen] [TaskManager] 每日任务 {name} 初始执行完成"
                    )
                except Exception as e:
                    logger.error(
                        f"[ImageGen] [TaskManager] 每日任务 {name} 初始执行失败: {e}",
                        exc_info=True,
                    )
            else:
                # 记录当前日期，避免启动当天重复执行
                self._last_run_dates[name] = datetime.now().strftime("%Y-%m-%d")

            while True:
                try:
                    await asyncio.sleep(check_interval_seconds)
                    current_date = datetime.now().strftime("%Y-%m-%d")
                    last_run_date = self._last_run_dates.get(name)

                    if current_date != last_run_date:
                        logger.info(
                            f"[ImageGen] [TaskManager] 检测到日期变更 ({last_run_date} -> {current_date})，执行每日任务 {name}"
                        )
                        try:
                            await coro_func()
                            self._last_run_dates[name] = current_date
                            logger.info(
                                f"[ImageGen] [TaskManager] 每日任务 {name} 执行完成"
                            )
                        except Exception as e:
                            logger.error(
                                f"[ImageGen] [TaskManager] 每日任务 {name} 执行出错: {e}",
                                exc_info=True,
                            )
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(
                        f"[ImageGen] [TaskManager] 每日任务 {name} 循环出错: {e}",
                        exc_info=True,
                    )

        task = asyncio.create_task(_daily_loop(), name=f"daily_{name}")
        self._daily_tasks[name] = task
        self.background_tasks.add(task)
        task.add_done_callback(functools.partial(self._on_daily_task_done, name))
        logger.info(
            f"[ImageGen] [TaskManager] 每日任务 {name} 已启动 (检查间隔: {check_interval_seconds}s)"
        )

    def stop_daily_task(self, name: str) -> None:
        """停止指定的每日任务。"""
        if task := self._daily_tasks.pop(name, None):
            if not task.done():
                task.cancel()
            self._last_run_dates.pop(name, None)
            logger.info(f"[ImageGen] [TaskManager] 每日任务 {name} 已停止")

    def _on_daily_task_done(self, name: str, task: asyncio.Task) -> None:
        """每日任务结束时的回调。"""
        self.background_tasks.discard(task)
        self._daily_tasks.pop(name, None)
        self._last_run_dates.pop(name, None)

    async def cancel_all(self):
        """取消所有正在运行的任务。"""
        for task in list(self.background_tasks):
            if not task.done():
                task.cancel()

        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        self.background_tasks.clear()
        self._loop_tasks.clear()
        self._daily_tasks.clear()
        self._last_run_dates.clear()
        logger.info("[ImageGen] [TaskManager] 所有后台任务已取消")
