import argparse, json
from common import ROOT
from arknights_sim.environment.replay import replay_solution

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("solution", nargs="?", default="outputs/mcts_0-1_solution.json")
    args = parser.parse_args()
    print(json.dumps(replay_solution(ROOT / args.solution), indent=2))
