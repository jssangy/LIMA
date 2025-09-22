import os
import sys
import argparse
import torch
import numpy as np

from GUI import GUI
from Environment import ENV
from module.model import ActorCritic

class RLPolicy:
    def __init__(self, model: ActorCritic, device="cpu", greedy=False):
        self.model = model.to(device).eval()
        self.device = device
        self.greedy = greedy

    @torch.no_grad()
    def __call__(self, obs: dict, action_mask: np.ndarray | None):
        # dict obs 그대로 사용 ({"state","edge_index"})
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

def main():
    parser = argparse.ArgumentParser(description="Run simulations and log ENV.make_info() to CSV.")
    parser.add_argument("--prob", type=str, default="cross_9")
    parser.add_argument("--algo", type=int, default=0, choices=[0,1,2,3,4],
                        help="0=BFS, 1=A*, 2=D*, 3=PIBT, 4=CBS")
    parser.add_argument("--num-amrs", type=int, default=20)
    parser.add_argument("--max-arm-h", type=int, default=5)
    parser.add_argument("--max-arm-v", type=int, default=5)
    parser.add_argument("--max-steps", type=int, default=10000)
    parser.add_argument("--traffic-mode", type=str, default="task", choices=["traffic","task"])
    parser.add_argument("--model-path", type=str, default="checkpoint/final_policy.pt")
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
        traffic_mode=args.traffic_mode
    )        

    # RL 정책 로드 & 연결
    state_dim = int(np.asarray(next(iter(env.intersections.values())).get_state()).shape[-1])
    env.rl_policy = load_policy(args.model_path, state_dim, device=("cuda" if torch.cuda.is_available() else "cpu"), greedy=False)

    env.use_rl = True  # GUI 체크박스도 자동으로 켜지게 하려면 True로
    env.reset()
    
    # 2. 생성된 환경을 GUI 클래스에 전달하여 실행
    app = GUI(env)

if __name__ == '__main__':
    main()