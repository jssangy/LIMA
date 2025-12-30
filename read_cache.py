# inspect_cache_db.py
import argparse
import os
import sqlite3

def human_bytes(n: int) -> str:
    n = float(n)
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0:
            return f"{n:.2f} {unit}"
        n /= 1024.0
    return f"{n:.2f} PB"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="cache/cache.sqlite")
    args = ap.parse_args()

    db_path = args.db
    if not os.path.exists(db_path):
        raise FileNotFoundError(db_path)

    # 1) 파일 크기
    size_bytes = os.path.getsize(db_path)
    print(f"[FILE] path: {os.path.abspath(db_path)}")
    print(f"[FILE] size: {size_bytes:,} bytes ({human_bytes(size_bytes)})")

    # 2) DB 열기 (read-only)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=30)
    conn.row_factory = sqlite3.Row

    # cache 테이블 존재 확인
    has_cache = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='cache'"
    ).fetchone() is not None
    if not has_cache:
        print("[DB] table 'cache' not found.")
        conn.close()
        return

    # 3) 캐시 엔트리 수 + blob 크기 통계
    count = conn.execute("SELECT COUNT(*) AS c FROM cache").fetchone()["c"]
    sizes = conn.execute("""
        SELECT
            SUM(length(k) + length(v)) AS sum_kv,
            AVG(length(k) + length(v)) AS avg_kv,
            MAX(length(k) + length(v)) AS max_kv,
            MIN(length(k) + length(v)) AS min_kv
        FROM cache
    """).fetchone()

    sum_kv = sizes["sum_kv"] or 0
    avg_kv = sizes["avg_kv"] or 0
    max_kv = sizes["max_kv"] or 0
    min_kv = sizes["min_kv"] or 0

    print("\n[CACHE]")
    print(f"  entries: {count:,}")
    print(f"  sum(k+v): {sum_kv:,} bytes ({human_bytes(sum_kv)})")
    print(f"  avg(k+v): {avg_kv:.1f} bytes")
    print(f"  min(k+v): {min_kv:,} bytes")
    print(f"  max(k+v): {max_kv:,} bytes")

    # 4) SQLite 페이지 구성(PRAGMA)
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    page_count = conn.execute("PRAGMA page_count").fetchone()[0]
    freelist_count = conn.execute("PRAGMA freelist_count").fetchone()[0]
    auto_vacuum = conn.execute("PRAGMA auto_vacuum").fetchone()[0]  # 0 none, 1 full, 2 incremental
    encoding = conn.execute("PRAGMA encoding").fetchone()[0]

    # 실제로 “사용 중”이라고 추정되는 페이지(프리리스트 제외)
    used_pages = max(0, page_count - freelist_count)
    used_bytes_est = used_pages * page_size
    total_bytes_est = page_count * page_size

    print("\n[SQLITE PAGES]")
    print(f"  page_size: {page_size:,} bytes")
    print(f"  page_count: {page_count:,}")
    print(f"  freelist_count: {freelist_count:,}")
    print(f"  used_pages (est): {used_pages:,}")
    print(f"  total_bytes (page_size*page_count): {total_bytes_est:,} ({human_bytes(total_bytes_est)})")
    print(f"  used_bytes  (est, excl freelist): {used_bytes_est:,} ({human_bytes(used_bytes_est)})")
    print(f"  auto_vacuum: {auto_vacuum} (0:none, 1:full, 2:incremental)")
    print(f"  encoding: {encoding}")

    # 추가 정보(원하면)
    # WAL 모드 여부는 RO 연결에서 정확히 못 볼 수 있어서 생략해도 됨.
    # 그래도 보고 싶으면 아래 주석 해제:
    # journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
    # print(f"  journal_mode: {journal_mode}")

    conn.close()

if __name__ == "__main__":
    main()
