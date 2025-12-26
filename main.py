import argparse

from GUI import GUI
from Environment import ENV

def main():
    parser = argparse.ArgumentParser(description="Run simulations and log ENV.make_info() to CSV.")
    parser.add_argument("--prob", type=str, default="warehouse_1")
    parser.add_argument("--algo", type=int, default=0, choices=[0,1,2,3,4],
                        help="0=BFS, 1=A*, 2=D*, 3=PIBT, 4=CBS")
    parser.add_argument("--density", type=int, default=40, 
                        help="Percentage(%) of AMRs in the environment.")
    parser.add_argument("--num-amrs", type=int, default=9,
                        help="If >0, overrides density to set the number of AMRs directly.")
    parser.add_argument("--max-steps", type=int, default=100000)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--cache-db-path", type=str, default="./cache/schedule_cache.sqlite",
                        help="Path to the schedule cache database.")
    args = parser.parse_args()

    prob_path = f"problems/cross/{args.prob}.json"

    # ENV 생성자에는 max_steps 없음 → 생성 후 속성으로 지정
    env = ENV(
        prob_path, 
        density=args.density, 
        num_amrs=args.num_amrs,
        max_steps=args.max_steps, 
        workers=args.workers,
        cache_db_path=args.cache_db_path,
    )

    env.reset()
    
    # 2. 생성된 환경을 GUI 클래스에 전달하여 실행
    app = GUI(env)

if __name__ == '__main__':
    main()