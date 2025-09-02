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
    def __call__(self, obs, action_mask=None):
        """
        obs 형식:
          - 단일 교차로: {"state": np.ndarray, "edge_index": np.ndarray}
          - 다중 교차로: { iid(str): {"state": ..., "edge_index": ...}, ... }
        action_mask 형식:
          - 단일: np.ndarray(bool)
          - 다중: { iid(str): np.ndarray(bool), ... }
        반환:
          - 단일: int
          - 다중: { iid(str): int }
        """
        # 다중 교차로 입력
        if isinstance(obs, dict) and "state" not in obs:
            results = {}
            mask_map = action_mask if isinstance(action_mask, dict) else {}
            for iid, o in obs.items():
                m = mask_map.get(iid, None)
                results[iid] = self._act_one(o, m)
            return results
        # 단일 교차로 입력
        return self._act_one(obs, action_mask)

    def _act_one(self, obs_one: dict, action_mask: np.ndarray | None):
        # 모델은 obs(dict)를 그대로 받는 기존 인터페이스 유지
        logits, _ = self.model.forward(obs_one)
        if action_mask is not None:
            am = torch.as_tensor(action_mask, dtype=torch.bool, device=logits.device).unsqueeze(0)
            logits = logits.masked_fill(~am, -1e9)
        if self.greedy:
            a = torch.argmax(logits, dim=-1)
            return int(a.item())
        else:
            am = None
            if action_mask is not None:
                am = torch.as_tensor(action_mask, dtype=torch.bool, device=logits.device).unsqueeze(0)
            a, _, _ = self.model.act(obs_one, action_mask=am)
            return int(a.item())


def load_policy(model_path: str, state_dim: int, action_dim: int,
                device="cpu", hidden=128, gnn_layers=2, greedy=True):
    model = ActorCritic(state_dim=state_dim, action_dim=action_dim,
                        hidden=hidden, gnn_layers=gnn_layers)
    sd = torch.load(model_path, map_location=device)
    model.load_state_dict(sd, strict=True)
    return RLPolicy(model, device=device, greedy=greedy)


def _infer_dims_from_env(env: ENV):
    """
    env에서 임의의 교차로 하나를 골라 state_dim / action_dim 추정.
    """
    # 교차로가 구성되어 있어야 함 (reset 이후 보장)
    if not getattr(env, "intersections", None):
        raise RuntimeError("No intersections found in env. Ensure env.reset() ran successfully.")

    # 임의의 첫 교차로
    first_iid = next(iter(env.intersections))
    I = env.intersections[first_iid]

    # state_dim
    state = np.asarray(I.get_state())
    if state.ndim == 0:
        raise RuntimeError("Intersection.get_state() returned scalar; expected 1D vector.")
    state_dim = int(state.shape[-1])

    # action_dim (마스크 길이 사용)
    mask = I.calculate_action_mask()
    action_dim = int(len(mask)) if mask is not None else 4  # fallback 4

    return state_dim, action_dim


def main():
    # 환경/모델 경로
    prob_path = os.path.join('problems', 'multicross', 'cross_25.json')
    model_path = os.path.join('checkpoint', 'policy.pt')

    # 1) ENV 생성 및 초기화
    env = ENV(prob_path)
    env.reset()  # 교차로 구성/초기 상태 보장

    # 2) RL 정책 로드
    state_dim, action_dim = _infer_dims_from_env(env)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    env.rl_policy = load_policy(
        model_path,
        state_dim=state_dim,
        action_dim=action_dim,
        device=device,
        hidden=128,
        gnn_layers=2,
        greedy=True,
    )
    env.use_rl = True  # GUI에서 RL 사용

    # 3) GUI 실행
    app = GUI(env)


if __name__ == '__main__':
    main()
