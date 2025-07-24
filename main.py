from Environment import ENV
from GUI import *
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--problem', '-p',
        type=str,
        default='problems/cross/cross_1.json',
        help='Problem json file path (e.g. problems/cross/cross_1.json)'
    )
    args = parser.parse_args()

    problem_json_path = args.problem

    env = ENV(problem_json_path)
    gui = GUI(env)
    return

if __name__ == "__main__":
    main()

