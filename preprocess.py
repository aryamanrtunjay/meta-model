import os
import json
import torch
from tqdm import tqdm
from train import TRACE_DIR, H_DRIVE_BASE, build_sequence_from_graph, FEATURE_DIM, VOCAB_PATH, NORM_STATS_PATH


def load_traces(trace_dir):
    return [f for f in os.listdir(trace_dir) if f.endswith('.json')]


def build_target_vocab(trace_files):
    vocab = {"<UNK>": 0}
    next_index = 1
    for trace_file in tqdm(trace_files, desc="Building target vocab"):
        trace_path = os.path.join(TRACE_DIR, trace_file)
        try:
            with open(trace_path, "r", encoding="utf-8") as f:
                trace = json.load(f)
        except Exception:
            continue

        for node in trace.get("graph", []):
            target_str = str(node.get("target", ""))
            if target_str not in vocab:
                vocab[target_str] = next_index
                next_index += 1

    return vocab


def compute_norm_stats(trace_files, vocab):
    all_vectors = []
    vocab_size = len(vocab)

    for trace_file in tqdm(trace_files, desc="Collecting feature vectors"):
        trace_path = os.path.join(TRACE_DIR, trace_file)
        try:
            with open(trace_path, "r", encoding="utf-8") as f:
                trace = json.load(f)
        except Exception:
            continue

        sequence = build_sequence_from_graph(trace, target_vocab=vocab, vocab_size=vocab_size)
        if not sequence:
            continue
        tensor = torch.tensor(sequence, dtype=torch.float)
        all_vectors.append(tensor)

    if not all_vectors:
        raise RuntimeError("No valid feature vectors collected from traces.")

    big_tensor = torch.vstack(all_vectors)
    min_vals = torch.min(big_tensor, dim=0).values
    max_vals = torch.max(big_tensor, dim=0).values

    torch.save({"min": min_vals.cpu(), "max": max_vals.cpu()}, NORM_STATS_PATH)
    with open(VOCAB_PATH, "w", encoding="utf-8") as f:
        json.dump(vocab, f)

    return min_vals, max_vals


def main():
    if not os.path.exists(TRACE_DIR):
        raise FileNotFoundError(f"Trace directory not found: {TRACE_DIR}")

    trace_files = load_traces(TRACE_DIR)
    if not trace_files:
        raise RuntimeError("No trace files found for preprocessing.")

    print(f"Found {len(trace_files)} trace files. Building vocabulary...")
    vocab = build_target_vocab(trace_files)
    print(f"Vocabulary size: {len(vocab)}")

    print("Computing normalization statistics...")
    min_vals, max_vals = compute_norm_stats(trace_files, vocab)

    print(f"Saved normalization stats to {NORM_STATS_PATH}")
    print(f"Saved vocabulary to {VOCAB_PATH}")
    print("Feature ranges:")
    for i in range(FEATURE_DIM):
        print(f"  Feature {i}: min={min_vals[i].item():.6f}, max={max_vals[i].item():.6f}")


if __name__ == "__main__":
    main()
