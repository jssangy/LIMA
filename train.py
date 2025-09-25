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
    parser.add_argument("--events_per_update", type=int, default=256)  # 'steps_per_update' -> 'events_per_update'로 변경
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-arm-h", type=int, default=5)
    parser.add_argument("--max-arm-v", type=int, default=5)
    parser.add_argument("--num-amrs", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=1024)
    parser.add_argument("--algo", type=int, default=0, choices=[0,1,2,3,4],
                        help="0=BFS, 1=A*, 2=D*, 3=PIBT, 4=CBS")
    parser.add_argument("--traffic-mode", type=str, default="task", choices=["traffic","task"])
    args = parser.parse_args()


    # 2) 학습 설정 (env 경로는 Config에 넣지 않음)
    cfg = TrainConfig(
        epochs=args.epochs,
        minibatch_size=args.minibatch,
        gamma=args.gamma,
        lam=args.lam,
        entropy_coef=args.entropy_coef,
        clip_eps=args.clip,
        lr=args.lr,
        device=args.device,
        value_coef=args.value_coef,
        max_grad_norm=args.max_grad_norm,
        total_steps=args.total_steps,
        events_per_update=args.events_per_update,  # 'steps_per_update' -> 'events_per_update'로 변경
        prob_path = args.prob_path,
        max_arm_h = args.max_arm_h,
        max_arm_v = args.max_arm_v,
        num_amrs = args.num_amrs,
        max_steps = args.max_steps,
        running_opt = args.algo,
        traffic_mode = args.traffic_mode,
    )

    # 3) Trainer에 env 주입 후 학습 시작
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    main()
