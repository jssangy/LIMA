import torch
import argparse
from tqdm import tqdm

# TorchRL 모듈 임포트
from torchrl.collectors import SyncDataCollector
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.envs.libs.gym import GymWrapper
from torchrl.modules import ActorValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

# tensordict 라이브러리에서 직접 임포트합니다.
from tensordict.nn import TensorDictModule

# 직접 만든 환경 및 분리된 모델 임포트
from gym_env import GymEnv
from model import CommonNet, PolicyHead, ValueHead

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

base_env = GymEnv(prob_path='problems/cross/cross_1.json')
env = GymWrapper(base_env, device=DEVICE)


td = env.reset()[0]            # TensorDict
policy(td.clone())             # 정책 통과 → action·_log_prob 생성?
value_module(td.clone())       # value 키 생성?
