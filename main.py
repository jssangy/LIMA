import os
import sys

def _ensure_fixed_hash_seed(seed: str = "0"):
    # 이미 고정돼 있으면 그대로 진행
    if os.environ.get("PYTHONHASHSEED") == seed:
        return

    # 환경변수 세팅 후, 같은 파이썬으로 프로세스 재시작
    os.environ["PYTHONHASHSEED"] = seed
    os.execv(sys.executable, [sys.executable] + sys.argv)

_ensure_fixed_hash_seed("0")

import argparse

from GUI import GUI
from Environment import ENV

def main():
    parser = argparse.ArgumentParser(description="Run GUI simulation (random or scen tasks).")

    parser.add_argument("--map", type=str, default="warehouse-10-20",
                        help="Map name located in assets/ folder.")
    
    parser.add_argument("--density", type=int, default=10, 
                        help="Percentage(%) of AMRs in the environment. (0~100)")
    
    parser.add_argument("--num-amrs", type=int, default=0,
                        help="If >0, overrides density to set the number of AMRs directly.")
    
    parser.add_argument("--max-steps", type=int, default=100000)

    parser.add_argument("--planner", choices=["bfs", "cbs"], default="bfs",)

    parser.add_argument("--workers", type=int, default=8)

    parser.add_argument("--cache-db-path", type=str, default="./cache/cache.sqlite",
                        help="Path to the schedule cache database.")
    
    parser.add_argument("--task-mode", choices=["random", "scen"], default="scen",
                        help="random: 기존 랜덤 생성 / scen: .scen 기반(고정)")
    
    parser.add_argument("--scen-idx", type=int, default=0,
                        help="scen index (예: s0~s9면 0~9)")
    
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed for reproducibility.")

    
    args = parser.parse_args()

    map_path = f"assets/{args.map}/{args.map}.map"
    scen_path = f"assets/{args.map}/scen/{args.map}_s{args.scen_idx}.scen"

    # ENV 생성자에는 max_steps 없음 → 생성 후 속성으로 지정
    env = ENV(
        map_path, 
        density=args.density, 
        num_amrs=args.num_amrs,
        max_steps=args.max_steps,
        planner=args.planner, 
        workers=args.workers,
        cache_db_path=args.cache_db_path,
        task_mode=args.task_mode,
        scen_path=scen_path,
        seed=args.seed
    )

    env.reset()
    
    # 2. 생성된 환경을 GUI 클래스에 전달하여 실행
    app = GUI(env)

if __name__ == '__main__':
    main()
