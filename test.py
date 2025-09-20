import os
import csv
import time
import argparse
import  torch
import numpy as np
from Environment import ENV
from module.model import ActorCritic

class RLPolicy:
    def __init__(self, model: ActorCritic, device="cpu", greedy=False):
        self.model = model.to(device).eval()
        self.device = device
        self.greedy = greedy

    @torch.no_grad()
    def __call__(self, obs: dict, action_mask: np.ndarray | None):
        if self.greedy:
            logits, _ = self.model.forward(obs)
            if action_mask is not None:
                am = torch.as_tensor(action_mask, dtype=torch.bool, device=logits.device).unsqueeze(0)
                logits = logits.masked_fill(~am, -1e9)
            a = torch.argmax(logits, dim=-1)
            return int(a.item())
        else:
            am = None
            if action_mask is not None:
                am = torch.as_tensor(action_mask, dtype=torch.bool, device=self.device).unsqueeze(0)
            a, _, _ = self.model.act(obs, action_mask=am)
            return int(a.item())

def load_policy(model_path: str, state_dim: int, device="cpu", greedy=False):
    model = ActorCritic(state_dim=state_dim, action_dim=4)
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd, strict=True)
    return RLPolicy(model, device=device, greedy=greedy)

def algo_name(algo: int) -> str:
    return {0:"BFS", 1:"AStar", 2:"DStar", 3:"PIBT", 4:"CBS"}.get(algo, f"Algo{algo}")

def compute_avg_steps(env: ENV) -> float:
    completed = list(env.completed_agv_steps) if getattr(env, "completed_agv_steps", None) else []
    active = [agv.steps for agv in env.agv_list.values()] if getattr(env, "agv_list", None) else []
    all_steps = completed + active
    return float(np.mean(all_steps)) if all_steps else 0.0

def run_one_episode(env: ENV, use_rl: bool) -> dict:
    env.use_rl = bool(use_rl)   # ← 여기서 on/off
    env.reset()
    while True:
        if env.step(actions={}, train=False) is False:
            break
    info = env.make_info()
    return {
        "success_rate": float(info.get("success_rate", 0.0)),
        "throughput":   float(info.get("throughput", 0.0)),
        "avg_integrity": float(info.get("avg_path_integrity", 0.0)),
        "avg_agv_steps": float(compute_avg_steps(env)),
        "avg_action_count": float(info.get("avg_action_count", 0.0)),
        "avg_inference_time": float(info.get("avg_inference_time", 0.0)),
    }

def main():
    parser = argparse.ArgumentParser(description="Run simulations and log ENV.make_info() to CSV.")
    parser.add_argument("--prob", type=str, default="cross_9")
    parser.add_argument("--algo", type=int, default=0, choices=[0,1,2,3,4],
                        help="0=BFS, 1=A*, 2=D*, 3=PIBT, 4=CBS")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--use-rl", action="store_true", default=False)
    parser.add_argument("--num-amrs", type=int, default=20)
    parser.add_argument("--max-arm-h", type=int, default=5)
    parser.add_argument("--max-arm-v", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--model-path", type=str, default="checkpoint/final_mlp_policy.pt")
    args = parser.parse_args()

    prob_path = f"problems/cross/{args.prob}.json"

    # ENV 생성자에는 max_steps 없음 → 생성 후 속성으로 지정
    env = ENV(prob_path, max_arm_len_h=args.max_arm_h, max_arm_len_v=args.max_arm_v, num_amrs=args.num_amrs, max_steps=args.max_steps, running_opt=args.algo)

    # RL 정책은 --use-rl 켠 경우에만 로드
    if args.use_rl:
        try:
            state_dim = int(np.asarray(next(iter(env.intersections.values())).get_state()).shape[-1])
            device = "cuda" if torch.cuda.is_available() else "cpu"
            env.rl_policy = load_policy(args.model_path, state_dim, device=device, greedy=False)
            print(f"[info] RL policy loaded from: {args.model_path} (device={device})")
        except FileNotFoundError:
            print(f"[warn] RL model not found at {args.model_path}. Disabling RL.")
            args.use_rl = False
        except Exception as e:
            print(f"[warn] Failed to load RL model ({e}). Disabling RL.")
            args.use_rl = False

    # CSV 경로/헤더
    os.makedirs("results", exist_ok=True)
    prob_base = os.path.splitext(os.path.basename(args.prob))[0]
    algo_label = algo_name(args.algo)
    rl_suffix = "_RL" if args.use_rl else ""
    csv_path = f"results/{prob_base}_{algo_label}{rl_suffix}_{args.num_amrs}.csv"
    fields = ["run", "success_rate", "throughput", "avg_integrity", "avg_inference_time", "avg_agv_steps", "avg_action_count"]

    print(f"Problem: {args.prob}")
    print(f"Algorithm: {algo_label} (running_opt={args.algo})")
    print(f"Runs: {args.runs}")
    print(f"RL: {'ON' if args.use_rl else 'OFF'}")
    print(f"Num AMRs: {args.num_amrs}")
    print(f"Arm caps: H={args.max_arm_h}, V={args.max_arm_v}")
    print(f"Max steps: {args.max_steps}")
    print(f"Saving to: {csv_path}")

    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for i in range(1, args.runs + 1):
            t0 = time.time()
            res = run_one_episode(env, use_rl=args.use_rl)
            dt = time.time() - t0
            w.writerow({
                "run": i,
                "success_rate":    f"{res['success_rate']:.6f}",
                "throughput":      f"{res['throughput']:.6f}",
                "avg_integrity":    f"{res['avg_integrity']:.6f}",
                "avg_inference_time": f"{res['avg_inference_time']:.6f}",
                "avg_agv_steps":   f"{res['avg_agv_steps']:.6f}",
                "avg_action_count":f"{res['avg_action_count']:.6f}",
            })
            f.flush()
            print(f"[{algo_label}] Run {i}/{args.runs} | SR={res['success_rate']:.2%} "
                  f"| TH={res['throughput']:.2f}/min | Steps={res['avg_agv_steps']:.2f} "
                  f"| Act={res['avg_action_count']:.2f} | {dt:.2f}s")

    print(f"\nSaved results to: {csv_path}")

if __name__ == "__main__":
    main()
