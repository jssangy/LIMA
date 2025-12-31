import os
import sys

def _ensure_fixed_hash_seed(seed: str = "0"):
    if os.environ.get("PYTHONHASHSEED") == seed:
        return

    os.environ["PYTHONHASHSEED"] = seed
    os.execv(sys.executable, [sys.executable] + sys.argv)

_ensure_fixed_hash_seed("0")

import argparse

from GUI import GUI
from Environment import ENV

def main():
    parser = argparse.ArgumentParser(description="Run GUI simulation (random or scen tasks).")

    parser.add_argument("--map", type=str, default="cross-30-30",
                        help="Map name located in assets/ folder.")
    
    parser.add_argument("--density", type=int, default=10, 
                        help="Percentage(%) of AMRs in the environment. (0~100)")
    
    parser.add_argument("--num-amrs", type=int, default=0,
                        help="If >0, overrides density to set the number of AMRs directly.")
    
    parser.add_argument("--max-steps", type=int, default=10000)

    parser.add_argument("--planner", choices=["bfs", "cbs"], default="bfs",)

    parser.add_argument("--workers", type=int, default=8)

    parser.add_argument("--cache-db-path", type=str, default="./cache/cache.sqlite",
                        help="Path to the schedule cache database.")
    
    parser.add_argument("--task-mode", choices=["random", "scen"], default="random",
                        help="random: random tasks, scen: scenario-based tasks")
    
    parser.add_argument("--scen-idx", type=int, default=0,
                        help="scen index (e.g., 0~9 for s0~s9)")
    
    parser.add_argument("--seed", type=int, default=7,
                        help="Random seed for reproducibility.")

    
    args = parser.parse_args()

    map_path = f"assets/{args.map}/{args.map}.map"
    scen_path = f"assets/{args.map}/scen/{args.map}_s{args.scen_idx}.scen"

    # ENV constructor does not have max_steps -> assign as attribute after creation
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
    
    # 2. Pass the created environment to the GUI class and run
    app = GUI(env)

if __name__ == '__main__':
    main()
