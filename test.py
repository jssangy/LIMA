import os
import sys
from GUI import GUI
from train.gym_env import GymEnv

def main():
    # 프로젝트 루트 경로를 sys.path에 추가
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

    # 환경 설정 파일 경로
    prob_path = os.path.join(project_root, 'problems', 'cross', 'cross_1.json')

    # 1. GymEnv 환경 인스턴스 생성
    env = GymEnv(prob_path)

    env.reset()
    
    # 2. 생성된 환경을 GUI 클래스에 전달하여 실행
    app = GUI(env)

if __name__ == '__main__':
    main()