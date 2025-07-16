from Environment import ENV
from GUI import *
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--problem', '--p',
        type=str,
        required=True,
        help='Problem json file path (e.g. example_problems/random.domain/random_32_32_20_100.json)'
    )
    args = parser.parse_args()

    problem_json_path = args.problem

    env = ENV(problem_json_path)
    gui = GUI(env)
    return

if __name__ == "__main__":
    main()

