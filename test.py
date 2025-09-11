import os
import sys
import argparse
import csv
import time
from datetime import datetime
from typing import Dict

import torch
import numpy as np

from Environment import ENV
from module.model import ActorCritic

class RLPolicy:
    def __init__(self, model: ActorCritic, device="cpu"):
        self.model = model.to(device).eval()
        self.device = device

    @torch.no_grad()
    def __call__(self, obs: dict, action_mask: np.ndarray | None) -> int:
        am = None
        if action_mask is not None:
            am = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        a, _, _ = self.model.act(obs, action_mask=am)
        return int(a.item())

def load_policy(model_path: str, state_dim: int, device="cpu") -> RLPolicy:
    model = ActorCritic(state_dim=state_dim, action_dim=4)
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd, strict=True)
    return RLPolicy(model, device=device)

def run_simulation(env: ENV) -> Dict:
    """한 에피소드의 시뮬레이션을 실행하고 결과를 반환합니다."""
    start_time = time.time()
    env.reset()
    
    while True:
        run_signal = env.step(actions={}, train=False)
        
        if run_signal is False:
            break
    
    end_time = time.time()

    completed_steps_list = env.completed_agv_steps
    agvs_steps = np.mean(completed_steps_list) if completed_steps_list else 0

    final_progress = env.traffic_generator.get_progress()
    total_tasks = final_progress['spawned_total']
    completed_tasks = final_progress['completed_total']
    
    # [수정] 성공률을 (완료된 Task / 전체 Task) 비율로 계산
    success_rate = (completed_tasks / total_tasks) if total_tasks > 0 else 0.0
    completion_time = end_time - start_time
    
    throughput = (completed_tasks / completion_time) * 60 if completion_time > 0 else 0

    return {
        "success_rate": success_rate,
        "throughput": throughput,
        "agv_steps": agvs_steps,
        "completion_time": completion_time,
        "completed_tasks": completed_tasks,
        "total_tasks": total_tasks,
    }

def get_unique_filename(base_path: str) -> str:
    """기존 파일을 덮어쓰지 않는 고유한 파일 경로를 반환합니다."""
    counter = 0
    file_path = base_path
    while os.path.exists(file_path):
        counter += 1
        name, extension = os.path.splitext(base_path)
        if '_' in name:
            name = '_'.join(name.split('_')[:-1])
        file_path = f"{name}_{counter}{extension}"
    return file_path

def main():
    parser = argparse.ArgumentParser(description="Run DAA-CPS Simulation for evaluation.")
    parser.add_argument('--prob', type=str, default='problems/cross/cross_3030.json', help="Path to the problem file.")
    parser.add_argument('--runs', type=int, default=30, help="Number of simulation runs.")
    parser.add_argument('--algo', type=int, default=0, choices=[0, 1, 2], 
                        help="Planning algorithm: 0=D*, 1=PIBT, 2=D*+PIBT_on_conflict")
    parser.add_argument('--use_rl', action='store_true', default=False, help="Enable intersection RL for deadlock resolution.")
    args = parser.parse_args()

    # --- 설정 ---
    NUM_RUNS = args.runs
    PROB_PATH = args.prob
    MODEL_PATH = os.path.join('checkpoint', 'final_mlp_policy.pt')
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # --- 환경 로드 및 알고리즘 설정 ---
    env = ENV(PROB_PATH)
    env.set_traffic_mode('task')
    
    env.controller.running_opt = args.algo
    env.use_rl = args.use_rl

    if env.use_rl:
        print("Intersection RL is ENABLED.")
        try:
            state_dim = int(np.asarray(next(iter(env.intersections.values())).get_state()).shape[-1])
            env.rl_policy = load_policy(MODEL_PATH, state_dim, device=DEVICE)
        except FileNotFoundError:
            print(f"Error: RL policy model not found at {MODEL_PATH}. Disabling RL.")
            env.use_rl = False
    else:
        print("Intersection RL is DISABLED.")

    # --- 로깅 설정 ---
    algo_map = {0: "DStar", 1: "PIBT", 2: "DStar_PIBT"}
    algo_name = algo_map.get(args.algo, "Unknown")
    rl_suffix = "_RL" if env.use_rl else ""
    
    base_log_filename = f"results/eval_{algo_name}{rl_suffix}.csv"
    os.makedirs("results", exist_ok=True)
    log_filename = get_unique_filename(base_log_filename)
    
    all_results = []

    with open(log_filename, 'w', newline='') as csvfile:
        # [수정] CSV 헤더를 'success_rate'로 변경
        fieldnames = ["run", "success_rate", "throughput", "agv_steps", "completion_time", "completed_tasks", "total_tasks"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        print(f"Starting evaluation for {NUM_RUNS} runs...")
        print(f"Algorithm: {algo_name}, RL: {'On' if env.use_rl else 'Off'}")
        print(f"Logging results to {log_filename}")

        # --- 시뮬레이션 반복 실행 ---
        for i in range(NUM_RUNS):
            print(f"--- Running simulation {i+1}/{NUM_RUNS} ---")
            
            result = run_simulation(env)
            result['run'] = i + 1
            all_results.append(result)
            
            writer.writerow(result)
            csvfile.flush()

            # [수정] 각 실행 결과 출력 포맷 변경
            print(f"Run {i+1}: Success Rate={result['success_rate']:.2%}, Throughput={result['throughput']:.2f}/min, Agv Steps={result['agv_steps']:.2f}, Time={result['completion_time']:.2f}s")

    # --- 최종 평균 계산 및 출력 ---
    if not all_results:
        print("\nNo simulations were run.")
        return
        
    # [수정] 평균 성공률 계산
    agv_success_rate = np.mean([r['success_rate'] for r in all_results])
    agv_throughput = np.mean([r['throughput'] for r in all_results])
    agv_time = np.mean([r['completion_time'] for r in all_results])
    overall_agv_steps = np.mean([r['agv_steps'] for r in all_results if r['completed_tasks'] > 0])

    print("\n--- Evaluation Summary ---")
    # [수정] 최종 요약 정보 출력 포맷 변경
    print(f"Average Success Rate: {agv_success_rate:.2%}")
    print(f"Average Throughput: {agv_throughput:.2f} tasks/min")
    print(f"Average AGV Steps: {overall_agv_steps:.2f}")
    print(f"Average Completion Time: {agv_time:.2f} s")
    print("--------------------------")

if __name__ == '__main__':
    main()