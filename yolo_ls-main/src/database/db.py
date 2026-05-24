"""数据库模块"""
import logging
import sqlite3
import threading
from pathlib import Path
from queue import Queue, Empty
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from contextlib import contextmanager

from src.utils.constants import (
    DEFAULT_DB_RETENTION_DAYS,
    DEFAULT_DB_MAX_ROWS_PER_TABLE,
    DEFAULT_DB_MAX_SIZE_MB,
)

logger = logging.getLogger(__name__)


class _SQLiteConnectionPool:
    """轻量连接池（无需 SQLAlchemy）。"""

    def __init__(self, db_path: Path, pool_size: int = 5):
        self.db_path = db_path
        self.pool_size = max(1, pool_size)
        self._queue: Queue = Queue(maxsize=self.pool_size)
        self._lock = threading.Lock()
        for _ in range(self.pool_size):
            self._queue.put(self._create_connection())

    def _create_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def acquire(self, timeout: float = 30.0) -> sqlite3.Connection:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            logger.warning("Connection pool exhausted, creating extra connection")
            return self._create_connection()

    def release(self, conn: sqlite3.Connection) -> None:
        try:
            self._queue.put_nowait(conn)
        except Exception:
            try:
                conn.close()
            except sqlite3.Error:
                pass


class Database:
    """SQLite 数据库管理"""

    def __init__(self, db_path: str = "data/traffic.db", pool_size: int = 5):
        """
        初始化数据库

        Args:
            db_path: 数据库文件路径
            pool_size: 连接池大小（>1 启用池化）
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._pool: Optional[_SQLiteConnectionPool] = None
        if pool_size > 1:
            self._pool = _SQLiteConnectionPool(self.db_path, pool_size=pool_size)

        self._actual_path = self.db_path  # 记录实际使用的路径
        try:
            self._init_tables()
        except sqlite3.OperationalError as exc:
            # 常见于 data 目录只读（如 Docker 以 root 创建后在宿主运行）
            if "readonly" not in str(exc).lower() and "permission" not in str(exc).lower():
                raise
            fallback = Path('/tmp/traffic.db')
            fallback.parent.mkdir(parents=True, exist_ok=True)
            self.db_path = fallback
            self._actual_path = fallback
            logger.warning(
                "Database at %s is read-only; falling back to %s. "
                "Data WILL BE LOST on container restart! "
                "Fix permissions on the original path or mount a writable volume.",
                str(db_path), str(fallback),
            )
            self._init_tables()

    def _open_connection(self) -> sqlite3.Connection:
        """复用单连接，避免每查询新建连接。"""
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
            )
            self._conn.row_factory = sqlite3.Row
        return self._conn

    @contextmanager
    def _get_connection(self):
        """线程安全的数据库访问（连接池或单连接）"""
        if self._pool is not None:
            conn = self._pool.acquire()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                logger.exception("Database operation failed")
                raise
            finally:
                self._pool.release(conn)
            return

        with self._lock:
            conn = self._open_connection()
            try:
                yield conn
                conn.commit()
            except Exception:
                conn.rollback()
                logger.exception("Database operation failed")
                raise

    def close(self):
        """关闭持久连接（应用退出时调用）"""
        with self._lock:
            if self._conn is not None:
                self._conn.close()
                self._conn = None
        if self._pool is not None:
            while not self._pool._queue.empty():
                try:
                    conn = self._pool._queue.get_nowait()
                    conn.close()
                except Empty:
                    break
            self._pool = None

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

    def _init_tables(self):
        """初始化数据表"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 车辆记录表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS vehicles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_id INTEGER,
                    plate_number TEXT,
                    vehicle_type TEXT,
                    color TEXT,
                    first_seen TIMESTAMP,
                    last_seen TIMESTAMP,
                    avg_speed REAL,
                    direction TEXT
                )
            ''')

            # 违规记录表（支持自适应违规检测）
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS violations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    record_id TEXT UNIQUE,
                    track_id INTEGER,
                    violation_type TEXT,
                    timestamp TIMESTAMP,
                    location_x INTEGER,
                    location_y INTEGER,
                    speed REAL,
                    plate_number TEXT,
                    snapshot_path TEXT,
                    is_exempted INTEGER DEFAULT 0,
                    exemption_reason TEXT,
                    exemption_details TEXT,
                    nearby_emergency_vehicles TEXT
                )
            ''')

            # 交通流量统计表
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS traffic_flow (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TIMESTAMP,
                    vehicle_count INTEGER,
                    avg_speed REAL,
                    direction TEXT
                )
            ''')

            # 创建索引
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_vehicles_plate
                ON vehicles(plate_number)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_violations_type
                ON violations(violation_type)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_violations_time
                ON violations(timestamp)
            ''')
            cursor.execute('''
                CREATE INDEX IF NOT EXISTS idx_violations_plate
                ON violations(plate_number)
            ''')

    def add_vehicle(self, track_id: int, plate_number: Optional[str],
                    vehicle_type: str, color: str, speed: float,
                    direction: str) -> int:
        """添加车辆记录"""
        now = datetime.now()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO vehicles
                (track_id, plate_number, vehicle_type, color, first_seen,
                 last_seen, avg_speed, direction)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (track_id, plate_number, vehicle_type, color, now, now,
                  speed, direction))
            return cursor.lastrowid

    def update_vehicle(self, track_id: int, speed: float, direction: str,
                      plate_number: Optional[str] = None,
                      vehicle_type: Optional[str] = None,
                      color: Optional[str] = None):
        """更新车辆记录"""
        now = datetime.now()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                UPDATE vehicles
                SET last_seen = ?,
                    avg_speed = ?,
                    direction = ?,
                    plate_number = COALESCE(?, plate_number),
                    vehicle_type = COALESCE(?, vehicle_type),
                    color = COALESCE(?, color)
                WHERE track_id = ?
            ''', (now, speed, direction, plate_number, vehicle_type, color, track_id))

    def add_violation(self, track_id: int, violation_type: str,
                      location: tuple, speed: Optional[float] = None,
                      plate_number: Optional[str] = None,
                      snapshot_path: Optional[str] = None,
                      record_id: Optional[str] = None,
                      is_exempted: bool = False,
                      exemption_reason: Optional[str] = None,
                      exemption_details: Optional[str] = None,
                      nearby_emergency_vehicles: Optional[List[str]] = None) -> int:
        """
        添加违规记录（支持自适应违规检测）

        Args:
            track_id: 车辆跟踪ID
            violation_type: 违规类型
            location: 位置坐标
            speed: 速度
            plate_number: 车牌号
            snapshot_path: 截图路径
            record_id: 记录ID（时间戳格式）
            is_exempted: 是否免责
            exemption_reason: 免责原因
            exemption_details: 免责详情
            nearby_emergency_vehicles: 附近特种车辆列表
        """
        now = datetime.now()
        if record_id is None:
            record_id = now.strftime("%Y%m%d_%H%M%S_%f")

        evs_str = ",".join(nearby_emergency_vehicles) if nearby_emergency_vehicles else None

        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO violations
                (record_id, track_id, violation_type, timestamp, location_x, location_y,
                 speed, plate_number, snapshot_path, is_exempted, exemption_reason,
                 exemption_details, nearby_emergency_vehicles)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (record_id, track_id, violation_type, now, location[0], location[1],
                  speed, plate_number, snapshot_path, 1 if is_exempted else 0,
                  exemption_reason, exemption_details, evs_str))
            return cursor.lastrowid

    def add_traffic_flow(self, vehicle_count: int, avg_speed: float,
                         direction: str):
        """添加交通流量记录"""
        now = datetime.now()
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('''
                INSERT INTO traffic_flow
                (timestamp, vehicle_count, avg_speed, direction)
                VALUES (?, ?, ?, ?)
            ''', (now, vehicle_count, avg_speed, direction))

    def get_table(self, table: str, limit: int = 200) -> List[Dict[str, Any]]:
        """
        通用表查询（仅用于 GUI 调试查看）
        """
        allowed = {"vehicles", "violations", "traffic_flow"}
        if table not in allowed:
            raise ValueError(f"Unsupported table: {table}")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT * FROM {table} ORDER BY id DESC LIMIT ?", (limit,))
            return [dict(row) for row in cursor.fetchall()]

    def get_vehicles(self,
                     plate_number: Optional[str] = None,
                     limit: int = 200) -> List[Dict[str, Any]]:
        """查询车辆记录（用于 GUI 展示）"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = '''
                SELECT id, track_id, plate_number, vehicle_type, color,
                       first_seen, last_seen, avg_speed, direction
                FROM vehicles
                WHERE 1=1
            '''
            params: List[Any] = []
            if plate_number:
                query += ' AND plate_number LIKE ?'
                params.append(f'%{plate_number}%')
            query += ' ORDER BY last_seen DESC LIMIT ?'
            params.append(limit)
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def delete_vehicles_by_ids(self, ids: List[int]) -> int:
        """根据主键ID删除车辆记录，返回删除数量"""
        if not ids:
            return 0
        with self._get_connection() as conn:
            cursor = conn.cursor()
            placeholders = ','.join('?' for _ in ids)
            cursor.execute(f'DELETE FROM vehicles WHERE id IN ({placeholders})', ids)
            deleted = cursor.rowcount
            # 如果全部删空，则重置自增ID
            self._reset_autoincrement_if_empty(cursor, "vehicles")
            return deleted

    def get_violations(self, violation_type: Optional[str] = None,
                       start_time: Optional[datetime] = None,
                       end_time: Optional[datetime] = None,
                       include_exempted: bool = True,
                       only_exempted: bool = False,
                       limit: int = 100) -> List[Dict[str, Any]]:
        """
        查询违规记录

        Args:
            violation_type: 违规类型筛选
            start_time: 开始时间
            end_time: 结束时间
            include_exempted: 是否包含免责记录
            only_exempted: 是否只查询免责记录
            limit: 返回数量限制
        """
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = 'SELECT * FROM violations WHERE 1=1'
            params = []

            if violation_type:
                query += ' AND violation_type = ?'
                params.append(violation_type)
            if start_time:
                query += ' AND timestamp >= ?'
                params.append(start_time)
            if end_time:
                query += ' AND timestamp <= ?'
                params.append(end_time)
            if only_exempted:
                query += ' AND is_exempted = 1'
            elif not include_exempted:
                query += ' AND is_exempted = 0'

            query += ' ORDER BY timestamp DESC LIMIT ?'
            params.append(limit)

            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]

    def get_traffic_stats(self, start_time: Optional[datetime] = None,
                          end_time: Optional[datetime] = None) -> Dict[str, Any]:
        """获取交通统计"""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            query = '''
                SELECT
                    COUNT(*) as total_vehicles,
                    AVG(avg_speed) as avg_speed,
                    COUNT(DISTINCT direction) as direction_count
                FROM traffic_flow WHERE 1=1
            '''
            params = []

            if start_time:
                query += ' AND timestamp >= ?'
                params.append(start_time)
            if end_time:
                query += ' AND timestamp <= ?'
                params.append(end_time)

            cursor.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else {}

    def get_violation_stats(self) -> Dict[str, Any]:
        """获取违规统计（包含免责统计）"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # 按类型统计
            cursor.execute('''
                SELECT violation_type, COUNT(*) as count
                FROM violations WHERE is_exempted = 0
                GROUP BY violation_type
            ''')
            by_type = {row['violation_type']: row['count'] for row in cursor.fetchall()}

            # 免责统计
            cursor.execute('''
                SELECT exemption_reason, COUNT(*) as count
                FROM violations WHERE is_exempted = 1
                GROUP BY exemption_reason
            ''')
            exempted_by_reason = {row['exemption_reason']: row['count'] for row in cursor.fetchall()}

            # 总计
            cursor.execute('SELECT COUNT(*) as total FROM violations')
            total = cursor.fetchone()['total']

            cursor.execute('SELECT COUNT(*) as exempted FROM violations WHERE is_exempted = 1')
            exempted = cursor.fetchone()['exempted']

            return {
                'by_type': by_type,
                'exempted_by_reason': exempted_by_reason,
                'total': total,
                'exempted': exempted,
                'actual_violations': total - exempted
            }

    def search_by_plate(self, plate_number: str) -> List[Dict[str, Any]]:
        """按车牌号搜索（车辆表 + 违规表）"""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            cursor.execute('''
                SELECT
                    'vehicle' as source,
                    track_id,
                    plate_number,
                    vehicle_type,
                    color,
                    first_seen as timestamp,
                    last_seen,
                    avg_speed as speed,
                    direction,
                    NULL as violation_type,
                    NULL as snapshot_path
                FROM vehicles
                WHERE plate_number LIKE ?
            ''', (f'%{plate_number}%',))
            vehicle_results = [dict(row) for row in cursor.fetchall()]

            cursor.execute('''
                SELECT
                    'violation' as source,
                    track_id,
                    plate_number,
                    NULL as vehicle_type,
                    NULL as color,
                    timestamp,
                    NULL as last_seen,
                    speed,
                    NULL as direction,
                    violation_type,
                    snapshot_path
                FROM violations
                WHERE plate_number LIKE ?
            ''', (f'%{plate_number}%',))
            violation_results = [dict(row) for row in cursor.fetchall()]

            results = vehicle_results + violation_results
            results.sort(key=lambda item: item.get('timestamp') or '', reverse=True)
            return results

    def clear_old_records(self,
                          days: int = DEFAULT_DB_RETENTION_DAYS,
                          max_rows_per_table: int = DEFAULT_DB_MAX_ROWS_PER_TABLE,
                          max_db_size_mb: int = DEFAULT_DB_MAX_SIZE_MB):
        """
        清理旧记录：按天数删除，并按行数/库文件大小做二次裁剪。
        """
        cutoff = datetime.now() - timedelta(days=days)
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('DELETE FROM vehicles WHERE last_seen < ?', (cutoff,))
            cursor.execute('DELETE FROM violations WHERE timestamp < ?', (cutoff,))
            cursor.execute('DELETE FROM traffic_flow WHERE timestamp < ?', (cutoff,))

            for table, order_col in (
                ('vehicles', 'id'),
                ('violations', 'id'),
                ('traffic_flow', 'id'),
            ):
                self._trim_table_to_max_rows(cursor, table, order_col, max_rows_per_table)

            self._reset_autoincrement_if_empty(cursor, "vehicles")
            self._reset_autoincrement_if_empty(cursor, "violations")
            self._reset_autoincrement_if_empty(cursor, "traffic_flow")

        if max_db_size_mb > 0 and self.db_path.exists():
            size_mb = self.db_path.stat().st_size / (1024 * 1024)
            if size_mb > max_db_size_mb:
                logger.warning(
                    "DB size %.1f MB > %d MB, running VACUUM and extra trim",
                    size_mb, max_db_size_mb,
                )
                self._vacuum_and_trim(max_rows_per_table // 2)

    def _trim_table_to_max_rows(self, cursor: sqlite3.Cursor,
                                table: str, order_col: str, max_rows: int) -> None:
        if max_rows <= 0:
            return
        cursor.execute(f'SELECT COUNT(*) as cnt FROM {table}')
        row = cursor.fetchone()
        count = row[0] if row else 0
        excess = count - max_rows
        if excess > 0:
            cursor.execute(
                f'''DELETE FROM {table} WHERE {order_col} IN (
                    SELECT {order_col} FROM {table}
                    ORDER BY {order_col} ASC LIMIT ?
                )''',
                (excess,),
            )

    def _vacuum_and_trim(self, max_rows: int) -> None:
        with self._get_connection() as conn:
            cursor = conn.cursor()
            for table in ('traffic_flow', 'violations', 'vehicles'):
                self._trim_table_to_max_rows(cursor, table, 'id', max_rows)
            cursor.execute('VACUUM')

    def clear_table(self, table: str) -> int:
        """
        清空指定表的所有记录，并重置自增序列。
        返回被删除的行数。
        """
        allowed = {'vehicles', 'violations', 'traffic_flow'}
        if table not in allowed:
            raise ValueError(f"不允许清空的表: {table}，仅支持 {allowed}")
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(f"SELECT COUNT(*) as cnt FROM {table}")
            row = cursor.fetchone()
            count = row[0] if row else 0
            cursor.execute(f"DELETE FROM {table}")
            self._reset_autoincrement_if_empty(cursor, table)
            logger.info("Cleared table '%s': %d rows deleted.", table, count)
            return count

    @staticmethod
    def _reset_autoincrement_if_empty(cursor: sqlite3.Cursor, table: str) -> None:
        """
        当指定表为空时，重置其自增序列，使后续插入从1重新开始。
        注意：不会重排已有记录，只在“清空后重新写入”的场景生效。
        """
        cursor.execute(f"SELECT COUNT(*) as cnt FROM {table}")
        row = cursor.fetchone()
        if row and row[0] == 0:
            # 对 AUTOINCREMENT 表，重置 sqlite_sequence
            try:
                cursor.execute("DELETE FROM sqlite_sequence WHERE name = ?", (table,))
            except sqlite3.OperationalError:
                # sqlite_sequence 可能不存在，忽略即可
                pass
