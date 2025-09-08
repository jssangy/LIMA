import os
import argparse
import torch

from module.trainer import TrainConfig, Trainer
from Environment import ENV


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prob_path", type=str, default="problems/cross/cross_1.json",
                        help="Path to your problem JSON for ENV(prob_path)")
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--minibatch", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--entropy_coef", type=float, default=0.01)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--value_coef", type=float, default=0.5)
    parser.add_argument("--max_grad_norm", type=float, default=0.5)
    parser.add_argument("--total_steps", type=int, default=500_000)
    parser.add_argument("--steps_per_update", type=int, default=256)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out_model", type=str, default="checkpoints/policy_final.pt")
    args = parser.parse_args()

    # 1) env를 여기서 생성해 의존성 주입
    env = ENV(args.prob_path)

    # 2) 학습 설정 (env 경로는 Config에 넣지 않음)
    cfg = TrainConfig(
        epochs=args.epochs,
        minibatch_size=args.minibatch,
        gamma=args.gamma,
        lam=args.lam,
        entropy_coef=args.entropy_coef,
        clip_eps=args.clip,
        lr=args.lr,
        seed=args.seed,
        device=args.device,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        total_steps=args.total_steps,
        steps_per_update=args.steps_per_update,
    )

    # 3) Trainer에 env 주입 후 학습 시작
    trainer = Trainer(cfg, env=env)
    trainer.train()


if __name__ == "__main__":
    main()
