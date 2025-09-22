import torch

@torch.no_grad()
def compute_gae(batch, gamma=0.99, lam=0.95):
    """
    SMDP-GAE: 이벤트 단위 전이(batch)에서 γ^τ, (γλ)^τ를 사용해 A_k와 return을 계산.

    기대 키:
      - rewards: [N]         R_k (이벤트 누적 보상)
      - taus: [N]            τ_k (이벤트 지속 스텝 수, float/long 모두 허용)
      - values: [N]          V(s_k)
      - next_values: [N]     V(s_{k+1}) (터미널/미상일 경우 0으로 채워져 있어야 함)
      - term: [N] (bool)     종료 플래그(terminated)
      - trunc: [N] (bool)    종료 플래그(truncated)

    반환:
      - batch에 'advantages', 'returns' 키를 추가해 반환
    """
    # 1) 입력 텐서들 준비
    r   = batch["rewards"]          # [N]
    tau = batch["taus"]             # [N]
    v    = batch["values"]          # [N]
    vnext= batch["next_values"]     # [N] (truncated이면 V(s_{k+1}), terminated면 0 권장)
    term = batch["terminated"].float()
    trunc= batch["truncated"].float()

    # 2) 두 마스크
    #    - 부트스트랩 마스크: terminated면 0 (다음 가치 사용 금지)
    #    - 재귀 마스크: done(= term|trunc)이면 0 (GAE가 다음으로 전달되지 않음)
    done = (term + trunc).clamp(max=1.0)
    bootstrap_mask = 1.0 - term          # terminated면 0
    recursion_mask = 1.0 - done          # term/trunc면 0

    # 3) δ_k = R_k + γ^{τ_k} * V(s_{k+1})_masked - V(s_k)
    gamma_tau = torch.pow(gamma, tau)
    delta = r + gamma_tau * vnext * bootstrap_mask - v

    # 4) GAE 역방향 누적: A_k = δ_k + (γλ)^{τ_k} * A_{k+1} * (1 - done)
    adv = torch.zeros_like(r)
    last = torch.zeros((), device=r.device, dtype=r.dtype)
    gl = gamma * lam
    gamma_lam_tau = torch.pow(torch.full_like(tau, gl, dtype=v.dtype, device=v.device), tau)
    for t in reversed(range(r.numel())):
        last = delta[t] + gamma_lam_tau[t] * last * recursion_mask[t]
        adv[t] = last
    ret = adv + v

    # 정규화(선택)
    adv = (adv - adv.mean()) / (adv.std(unbiased=False).clamp_min(1e-8))

    out = dict(batch)
    out["advantages"] = adv
    out["returns"]    = ret
    return out