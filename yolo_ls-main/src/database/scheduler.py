"""数据库定时清理任务（APScheduler 优先，threading 回退）。"""
from __future__ import annotations

import logging
import threading
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

try:
    from apscheduler.schedulers.background import BackgroundScheduler
    from apscheduler.triggers.interval import IntervalTrigger
    _HAS_APSCHEDULER = True
except ImportError:
    BackgroundScheduler = None  # type: ignore
    IntervalTrigger = None  # type: ignore
    _HAS_APSCHEDULER = False


class DatabaseCleanupScheduler:
    """
    周期性调用 Database.clear_old_records()。
    默认每 24 小时执行一次，可在配置 database.cleanup_interval_hours 中修改。
    """

    def __init__(
        self,
        cleanup_fn: Callable[[], None],
        interval_hours: float = 24.0,
    ):
        self._cleanup_fn = cleanup_fn
        self._interval_hours = max(0.5, float(interval_hours))
        self._scheduler: Any = None
        self._timer: Optional[threading.Timer] = None
        self._stopped = False

    def _run_cleanup(self) -> None:
        if self._stopped:
            return
        try:
            self._cleanup_fn()
            logger.info("Scheduled database cleanup completed")
        except Exception:
            logger.exception("Scheduled database cleanup failed")

    def _schedule_timer_chain(self) -> None:
        if self._stopped:
            return
        interval_sec = self._interval_hours * 3600.0
        self._timer = threading.Timer(interval_sec, self._timer_callback)
        self._timer.daemon = True
        self._timer.start()

    def _timer_callback(self) -> None:
        self._run_cleanup()
        self._schedule_timer_chain()

    def start(self) -> Dict[str, Any]:
        """启动后台清理；返回使用的后端信息。"""
        self._stopped = False
        if _HAS_APSCHEDULER and BackgroundScheduler is not None:
            self._scheduler = BackgroundScheduler(daemon=True)
            self._scheduler.add_job(
                self._run_cleanup,
                trigger=IntervalTrigger(hours=self._interval_hours),
                id='db_cleanup',
                replace_existing=True,
            )
            self._scheduler.start()
            logger.info(
                "Database cleanup scheduler started (APScheduler, every %.1f h)",
                self._interval_hours,
            )
            return {'backend': 'apscheduler', 'interval_hours': self._interval_hours}

        self._schedule_timer_chain()
        logger.info(
            "Database cleanup scheduler started (threading.Timer, every %.1f h)",
            self._interval_hours,
        )
        return {'backend': 'threading', 'interval_hours': self._interval_hours}

    def stop(self) -> None:
        self._stopped = True
        if self._scheduler is not None:
            try:
                self._scheduler.shutdown(wait=False)
            except Exception:
                logger.debug("Scheduler shutdown", exc_info=True)
            self._scheduler = None
        if self._timer is not None:
            self._timer.cancel()
            self._timer = None


def start_db_cleanup_from_config(database, config: Dict[str, Any]) -> Optional[DatabaseCleanupScheduler]:
    """根据配置创建并启动清理调度器。"""
    db_cfg = config.get('database', {})
    if not db_cfg.get('cleanup_enabled', True):
        return None

    interval_hours = float(db_cfg.get('cleanup_interval_hours', 24))
    retention_days = int(db_cfg.get('retention_days', 30))
    max_rows = int(db_cfg.get('max_rows_per_table', 50_000))
    max_size_mb = int(db_cfg.get('max_size_mb', 512))

    def _cleanup() -> None:
        database.clear_old_records(
            days=retention_days,
            max_rows_per_table=max_rows,
            max_db_size_mb=max_size_mb,
        )

    scheduler = DatabaseCleanupScheduler(_cleanup, interval_hours=interval_hours)
    scheduler.start()
    return scheduler
