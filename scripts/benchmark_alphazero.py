"""Compare CPU and MPS with synchronized small-network inference and batch updates.

The batch-update targets here are synthetic timing probes, not self-play data;
benchmark parameters are discarded and never become a training checkpoint.
"""

import argparse
import copy
import json
import platform
import statistics
import time
from pathlib import Path
from common import ROOT


def main():
    import torch
    from arknights_sim.environment import ArknightsEnv
    from network.state_encoder import StateEncoder, collate_states
    from network.action_encoder import ActionEncoder
    from network.policy_value_net import PolicyValueNetwork

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "research/alphazero/device_benchmark.json")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    torch.manual_seed(12345)
    env = ArknightsEnv()
    state = env.reset(seed=12345)
    while len(env.legal_actions(state)) == 1 and not env.is_terminal(state):
        state = env.advance_to_next_decision_event(state)
    actions = env.legal_actions(state)
    encoder, action_encoder = StateEncoder(), ActionEncoder()
    encoded = encoder.encode(state)
    encoded_actions = action_encoder.encode(state, actions)
    base_model = PolicyValueNetwork()

    def samples(call, count):
        result = []
        for _ in range(count):
            started = time.perf_counter()
            call()
            result.append((time.perf_counter() - started) * 1000)
        return {"median_ms": statistics.median(result), "mean_ms": statistics.mean(result), "samples_ms": result}

    encoding = samples(lambda: (encoder.encode(state), action_encoder.encode(state, actions)), args.repeats)
    rows = []
    for device in ("cpu", "mps"):
        if device == "mps" and not torch.backends.mps.is_available():
            rows.append({"device": device, "available": False})
            continue
        row = {"device": device, "available": True}
        try:
            model = copy.deepcopy(base_model).to(device)
            features = {k: v.to(device) for k, v in encoded.items()}
            action_features = encoded_actions.to(device)
            batch = {k: v.to(device) for k, v in collate_states([encoded] * args.batch_size).items()}
            batch_actions = encoded_actions.unsqueeze(0).repeat(args.batch_size, 1, 1).to(device)
            mask = torch.ones(batch_actions.shape[:2], dtype=torch.bool, device=device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)

            def sync():
                if device == "mps":
                    torch.mps.synchronize()

            def infer():
                with torch.inference_mode():
                    logits, value = model(features, action_features)
                    torch.softmax(logits, -1).detach().cpu()
                    value.detach().cpu()
                sync()

            def update():
                optimizer.zero_grad(set_to_none=True)
                logits, value = model(batch, batch_actions, mask)
                target = torch.full_like(logits, 1 / len(actions))
                loss = -(target * torch.log_softmax(logits, -1)).sum(-1).mean() + (value + 1).square().mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                sync()

            model.eval()
            for _ in range(5):
                infer()
            row["single_state_inference"] = samples(infer, args.repeats)
            model.train()
            for _ in range(3):
                update()
            row["batch_update"] = samples(update, max(5, args.repeats // 3))
            if device == "mps":
                row["mps_current_allocated_bytes"] = torch.mps.current_allocated_memory()
                row["mps_driver_allocated_bytes"] = torch.mps.driver_allocated_memory()
        except Exception as exc:
            import traceback
            row["error"] = str(exc)
            row["traceback"] = traceback.format_exc()
        rows.append(row)
    valid = [row for row in rows if row.get("available") and "error" not in row]
    chosen = min(valid, key=lambda row: row["single_state_inference"]["median_ms"])["device"] if valid else None
    result = {
        "python": platform.python_version(), "platform": platform.platform(), "torch": torch.__version__,
        "threads": args.threads, "batch_size": args.batch_size, "parameters": sum(p.numel() for p in base_model.parameters()),
        "state_time": state.game.current_time, "legal_action_count": len(actions),
        "encoding": encoding, "devices": rows, "selected_for_search": chosen,
        "selection_reason": "Lowest synchronized single-state inference median; neural MCTS makes many single-state calls.",
        "method": "Warm caches; data loading excluded; output transfers and MPS synchronization included. Batch targets are synthetic timing probes, never retained as labels or checkpoints.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k not in ("devices", "encoding")}, indent=2))
    for row in rows:
        print(row["device"], {k: v.get("median_ms") if isinstance(v, dict) else v for k, v in row.items() if k != "traceback"})


if __name__ == "__main__":
    main()

