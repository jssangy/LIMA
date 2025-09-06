import os
import sys
from GUI import GUI
from Environment import ENV

import torch, numpy as np
from module.model import ActorCritic

class RLPolicy:
    def __init__(self, model: ActorCritic, device="cpu", greedy=True):
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

def load_policy(model_path: str, state_dim: int, device="cpu", hidden=128, gnn_layers=2, greedy=True):
    model = ActorCritic(state_dim=state_dim, action_dim=4, hidden=hidden, gnn_layers=gnn_layers)
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd, strict=True)
    return RLPolicy(model, device=device, greedy=greedy)

def main():

    # 환경 설정 파일 경로
    prob_path = os.path.join('problems', 'cross', 'cross_1.json')
    model_path = os.path.join('checkpoint', 'best_policy.pt')

    # 1. ENV 환경 인스턴스 생성
    env = ENV(prob_path)
    # RL 정책 로드 & 연결
    state_dim = int(np.asarray(next(iter(env.intersections.values())).get_state()).shape[-1])
    env.rl_policy = load_policy(model_path, state_dim, device=("cuda" if torch.cuda.is_available() else "cpu"), hidden=128, gnn_layers=2, greedy=True)
    
    env.use_rl = True  # GUI 체크박스도 자동으로 켜지게 하려면 True로

    env.reset()
    
    # 2. 생성된 환경을 GUI 클래스에 전달하여 실행
    app = GUI(env)

if __name__ == '__main__':
    main()