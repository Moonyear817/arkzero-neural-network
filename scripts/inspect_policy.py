"""Inspect neural preferences and visit-count policy on a real simulator state."""

import argparse
import json
from pathlib import Path
from common import ROOT


def main():
    import torch
    from arknights_sim.environment import ArknightsEnv
    from network.policy_value_net import PolicyValueNetwork
    from network.inference import NeuralEvaluator
    from mcts.search import PUCTSearch, PUCTConfig

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--stage", default="0-1")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--time", type=float, default=None)
    parser.add_argument("--simulations", type=int, default=32)
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--device", choices=("cpu", "mps"), default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs/alphazero/policy_inspection.json")
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model = PolicyValueNetwork()
    iteration = None
    if args.checkpoint:
        document = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(document["model_state_dict"])
        model.reward_version = document.get('reward_version', 1)
        iteration = document["iteration"]
    evaluator = NeuralEvaluator(model, device=args.device)
    env = ArknightsEnv()
    state = env.reset(args.stage, seed=args.seed)
    if args.time is not None:
        state = env.advance_to_time(state, args.time)
    else:
        while len(env.legal_actions(state)) == 1 and not env.is_terminal(state):
            state = env.advance_to_next_decision_event(state)
    if env.is_terminal(state):
        parser.error("Inspection state is terminal; no legal policy distribution")
    actions = env.legal_actions(state)
    priors, value = evaluator.evaluate(env, state, actions)
    search = PUCTSearch(evaluator, PUCTConfig(simulations=args.simulations, seed=args.seed,
                                             prior_uniform_mix=0.0, temperature=0.0))
    result = search.search(env, state, training=False)
    rows = [{"action": action.to_dict(), "label": str(action), "neural_policy": float(prior),
             "mcts_policy": float(pi), "visits": int(visits)}
            for action, prior, pi, visits in zip(actions, priors, result.visit_policy, result.visit_counts)]
    rows.sort(key=lambda row: (-row["mcts_policy"], -row["neural_policy"], row["label"]))
    document = {"stage": args.stage, "seed": args.seed, "time": state.game.current_time,
                "dp": state.game.dp, "checkpoint": str(args.checkpoint) if args.checkpoint else None,
                "iteration": iteration, "value": value, "value_is_probability": False,
                "state_key": env.state_key(state), "terminal": False,
                "selected_action": result.selected_action.to_dict(), "actions": rows,
                "mcts_simulations": args.simulations, "evaluation_noise": False,
                "mcts_policy_definition": "Raw visit fractions N / sum(N); selection itself uses temperature zero."}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n")
    print(f"Time={state.game.current_time:.3f}  DP={state.game.dp:.3f}  Value={value:+.5f} (not a probability)")
    print(f"{'Action':60} {'Neural':>9} {'MCTS':>9} {'N':>7}")
    for row in rows[:args.top]:
        print(f"{row['label']:60} {row['neural_policy']:9.5f} {row['mcts_policy']:9.5f} {row['visits']:7d}")
    print(args.output)


if __name__ == "__main__":
    main()
