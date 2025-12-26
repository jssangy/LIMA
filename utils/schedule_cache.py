# utils/schedule_cache.py
from __future__ import annotations

import os
import sqlite3
import pickle
import zlib
import hashlib
from typing import Optional, Sequence, Tuple


def freeze_stacks(stacks) -> Tuple[Tuple[int, ...], ...]:
    return tuple(tuple(int(x) for x in s) for s in stacks)


def make_cache_key(
    initial_stacks,
    stack_capacities: Sequence[int],
) -> bytes:
    """
    key는 solve_h2()의 입력을 결정하는 최소 정보만:
      - initial_stacks
      - stack_capacities
    """
    payload = (
        freeze_stacks(initial_stacks),
        tuple(int(x) for x in stack_capacities),
    )
    b = pickle.dumps(payload, protocol=4)
    return hashlib.blake2b(b, digest_size=16).digest()  # 16 bytes


# (src,dst) 한 move를 1바이트로 저장: (src<<4 | dst)
# src,dst는 0~15 범위여야 함 (삼/사거리 OK)
def encode_actions(actions) -> bytes:
    raw = bytearray()
    for (src, dst) in actions:
        if not (0 <= src <= 15 and 0 <= dst <= 15):
            raise ValueError(f"src/dst too large for 1-byte encoding: {(src, dst)}")
        raw.append((int(src) << 4) | int(dst))
    return zlib.compress(bytes(raw), level=3)


def decode_actions(blob: bytes):
    raw = zlib.decompress(blob)
    return [(b >> 4, b & 0x0F) for b in raw]


class CacheReader:
    """
    워커용: read-only로만 연다.
    (메인 프로세스가 CacheWriter로 DB 파일을 먼저 만들어둬야 mode=ro가 성공함)
    """
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._pid = None
        self._conn = None

    def _connect_ro(self):
        uri = f"file:{self.db_path}?mode=ro"
        self._conn = sqlite3.connect(uri, uri=True, timeout=30)

    def _ensure(self):
        pid = os.getpid()
        if self._conn is None or self._pid != pid:
            self._pid = pid
            self._connect_ro()

    def get_blob(self, key: bytes) -> Optional[bytes]:
        try:
            self._ensure()
            row = self._conn.execute("SELECT v FROM cache WHERE k=?", (key,)).fetchone()
            return row[0] if row else None
        except sqlite3.OperationalError:
            return None

    def get_actions(self, key: bytes):
        blob = self.get_blob(key)
        return decode_actions(blob) if blob else None


class CacheWriter:
    """메인용: write 전담"""
    def __init__(self, db_path: str):
        d = os.path.dirname(db_path)
        if d:
            os.makedirs(d, exist_ok=True)

        self.conn = sqlite3.connect(db_path, timeout=30)
        self.conn.execute("PRAGMA journal_mode=WAL;")
        self.conn.execute("PRAGMA synchronous=NORMAL;")
        self.conn.execute("CREATE TABLE IF NOT EXISTS cache (k BLOB PRIMARY KEY, v BLOB)")
        self.conn.commit()

    def put_blob(self, key: bytes, blob: bytes) -> None:
        try:
            self.conn.execute("INSERT OR REPLACE INTO cache(k,v) VALUES(?,?)", (key, blob))
            self.conn.commit()
        except sqlite3.OperationalError:
            # 캐시는 best-effort
            pass
