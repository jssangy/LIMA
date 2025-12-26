import os
import sqlite3

db_path = "cache/schedule_cache.sqlite"  # 너 경로로 수정

# 1) 파일 크기
size_bytes = os.path.getsize(db_path)
print(f"DB file size: {size_bytes:,} bytes  ({size_bytes/1024/1024:.2f} MB, {size_bytes/1024/1024/1024:.2f} GB)")

# 2) 캐시 엔트리 개수 + 평균 저장 크기
conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
count = conn.execute("SELECT COUNT(*) FROM cache").fetchone()[0]
avg_blob = conn.execute("SELECT AVG(length(k) + length(v)) FROM cache").fetchone()[0] or 0
max_blob = conn.execute("SELECT MAX(length(k) + length(v)) FROM cache").fetchone()[0] or 0
print(f"cache entries: {count:,}")
print(f"avg (k+v) bytes: {avg_blob:.1f}")
print(f"max (k+v) bytes: {max_blob:,}")

# 3) SQLite 페이지 정보(실제 파일 구성 참고)
page_size = conn.execute("PRAGMA page_size").fetchone()[0]
page_count = conn.execute("PRAGMA page_count").fetchone()[0]
print(f"page_size: {page_size} bytes, page_count: {page_count:,}, (page_size*page_count ≈ {page_size*page_count/1024/1024:.2f} MB)")
conn.close()

# 4) (추정) 현재 평균 크기 기준으로 1GB당 몇 엔트리 정도 들어가는지
if avg_blob > 0:
    approx_per_gb = int((1024**3) / (avg_blob * 1.3))  # 오버헤드 여유로 1.3배 잡음
    print(f"rough estimate: ~{approx_per_gb:,} entries per 1GB (including overhead margin)")
