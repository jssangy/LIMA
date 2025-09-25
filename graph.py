import os, glob
import pandas as pd
import matplotlib.pyplot as plt

# 1) 파일에서 평균 completed_time 수집
rows = []
missing = []
for n in range(3, 16):
    path = f"results/cross_1_BFS_RL_{n}.csv"
    if not os.path.exists(path):
        missing.append(n)
        continue
    df = pd.read_csv(path)
    if "rl_action_count" not in df.columns:
        raise KeyError(f"{path}에 'rl_action_count' 컬럼이 없습니다.")
    rows.append({"num_amrs": n, "rl_action_count": df["rl_action_count"].mean()})

if missing:
    print(f"[경고] 파일 없음 → num_amrs={missing}")

summary = pd.DataFrame(rows).sort_values("num_amrs")
print(summary)


# 2) 그래프 그리기 (matplotlib)
plt.figure()
plt.plot(summary["num_amrs"], summary["rl_action_count"], marker="o")
plt.title("RL Action Count by num_amrs")
plt.xlabel("num_amrs")
plt.ylabel("rl_action_count")
plt.xticks(range(3, 16))
plt.grid(True)
plt.tight_layout()
plt.show()
