import os
import sys
import argparse
import numpy as np

from GUI import GUI
from Environment import ENV

def main():
    parser = argparse.ArgumentParser(description="Run simulations and log ENV.make_info() to CSV.")
    parser.add_argument("--prob", type=str, default="cross_3030")
    parser.add_argument("--algo", type=int, default=0, choices=[0,1,2,3,4],
                        help="0=BFS, 1=A*, 2=D*, 3=PIBT, 4=CBS")
    parser.add_argument("--num-amrs", type=int, default=7000)
    parser.add_argument("--max-arm-h", type=int, default=5)
    parser.add_argument("--max-arm-v", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=100000000)
    parser.add_argument("--traffic-mode", type=str, default="task", choices=["traffic","task"])
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    prob_path = f"problems/cross/{args.prob}.json"

    # ENV 생성자에는 max_steps 없음 → 생성 후 속성으로 지정
    env = ENV(
        prob_path, 
        max_arm_len_h=args.max_arm_h, 
        max_arm_len_v=args.max_arm_v, 
        num_amrs=args.num_amrs, 

        max_steps=args.max_steps, 
        running_opt=args.algo, 
        traffic_mode=args.traffic_mode,
        workers=args.workers,
    )

    env.reset()
    
    # 2. 생성된 환경을 GUI 클래스에 전달하여 실행
    app = GUI(env)

if __name__ == '__main__':
    main()