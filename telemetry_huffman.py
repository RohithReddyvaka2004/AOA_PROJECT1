#!/usr/bin/env python3

from __future__ import annotations
import math, random, heapq, os, argparse, time, json
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ------------------ Core Huffman ------------------

@dataclass(order=True)
class HuffmanNode:
    freq: int
    id: int = field(compare=False)
    symbol: Optional[str] = field(default=None, compare=False)
    left: Optional['HuffmanNode'] = field(default=None, compare=False)
    right: Optional['HuffmanNode'] = field(default=None, compare=False)

def build_huffman(frequencies: Dict[str, int]) -> HuffmanNode:
    heap: List[HuffmanNode] = []
    uid = 0
    for s, f in frequencies.items():
        heap.append(HuffmanNode(freq=int(f), id=uid, symbol=s))
        uid += 1
    heapq.heapify(heap)
    if len(heap) == 1:
        only = heapq.heappop(heap)
        return HuffmanNode(freq=only.freq, id=uid, left=only, right=None)
    while len(heap) >= 2:
        a = heapq.heappop(heap)
        b = heapq.heappop(heap)
        uid += 1
        parent = HuffmanNode(freq=a.freq + b.freq, id=uid, left=a, right=b)
        heapq.heappush(heap, parent)
    return heap[0]

def extract_codes(node: HuffmanNode, prefix: str="", table: Optional[Dict[str, str]]=None) -> Dict[str, str]:
    if table is None:
        table = {}
    if node.symbol is not None and node.left is None and node.right is None:
        table[node.symbol] = prefix if prefix != "" else "0"
        return table
    if node.left is not None:
        extract_codes(node.left, prefix + "0", table)
    if node.right is not None:
        extract_codes(node.right, prefix + "1", table)
    return table

def encode_stream(stream: List[str], codes: Dict[str, str]) -> str:
    return "".join(codes[s] for s in stream)

def decode_stream(bits: str, root: HuffmanNode) -> List[str]:
    out: List[str] = []
    node = root
    for bit in bits:
        node = node.left if bit == "0" else node.right
        if node.symbol is not None and node.left is None and node.right is None:
            out.append(node.symbol)
            node = root
    return out

# ------------------ Utilities ------------------

def empirical_entropy(frequencies: Dict[str, int]) -> float:
    N = sum(frequencies.values())
    H = 0.0
    for f in frequencies.values():
        if f > 0:
            p = f / N
            H -= p * math.log2(p)
    return H

def avg_code_length(codes: Dict[str, str], frequencies: Dict[str, int]) -> float:
    N = sum(frequencies.values())
    return sum((frequencies[s] / N) * len(code) for s, code in codes.items())

def simulate_telemetry_frequencies(k: int=32, length:int=200_000, seed: int=7) -> Tuple[List[str], Dict[str,int]]:
    rng = random.Random(seed)
    symbols = [f"S{str(i).zfill(2)}" for i in range(k)]
    s = 1.1
    weights = [1.0 / ((i+1) ** s) for i in range(k)]
    total_w = sum(weights)
    probs = [w / total_w for w in weights]
    cdf = []
    csum = 0.0
    for p in probs:
        csum += p
        cdf.append(csum)
    stream_syms: List[str] = []
    for _ in range(length):
        r = rng.random()
        lo, hi = 0, k-1
        while lo < hi:
            mid = (lo + hi) // 2
            if r <= cdf[mid]:
                hi = mid
            else:
                lo = mid + 1
        stream_syms.append(symbols[lo])
    freq: Dict[str, int] = {s: 0 for s in symbols}
    for s_ in stream_syms:
        freq[s_] += 1
    return stream_syms, freq

# ------------------ Experiments ------------------

def run_end_to_end(artifacts_dir: str, k=32, length=120_000, seed=21):
    os.makedirs(artifacts_dir, exist_ok=True)
    stream, freq = simulate_telemetry_frequencies(k=k, length=length, seed=seed)
    root = build_huffman(freq)
    codes = extract_codes(root)
    bits = encode_stream(stream, codes)
    back = decode_stream(bits, root)
    ok = (stream == back)
    H = empirical_entropy(freq)
    L = avg_code_length(codes, freq)
    fixed_bits = math.ceil(math.log2(len(freq)))
    ratio = (fixed_bits / L) if L > 0 else float('inf')
    summary = {
        "alphabet_size": len(freq),
        "stream_length": len(stream),
        "entropy_bits_per_symbol": H,
        "avg_huffman_bits_per_symbol": L,
        "fixed_length_bits_per_symbol": fixed_bits,
        "theoretical_gap_L_minus_H": L - H,
        "compression_ratio_vs_fixed": ratio,
        "lossless_decode_ok": ok,
    }
    pd.DataFrame([summary]).to_csv(os.path.join(artifacts_dir, "huffman_summary.csv"), index=False)

    rows = [{"symbol": s, "freq": freq[s], "prob": freq[s]/sum(freq.values()), "code": codes[s], "code_len": len(codes[s])} for s in freq]
    df_codes = pd.DataFrame(rows).sort_values(by="prob", ascending=False).reset_index(drop=True)
    df_codes.to_csv(os.path.join(artifacts_dir, "huffman_codes.csv"), index=False)

    # Plots (no seaborn, single plots, no custom colors)
    plt.figure()
    plt.scatter(df_codes["prob"], df_codes["code_len"])
    plt.xlabel("Symbol probability")
    plt.ylabel("Code length (bits)")
    plt.title("Huffman: Probabilities vs Code Lengths")
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "huffman_prob_vs_codelen.png"))
    plt.close()

    plt.figure()
    bars_x = ["Entropy H", "Huffman Avg L", "Fixed-length"]
    bars_y = [H, L, fixed_bits]
    plt.bar(bars_x, bars_y)
    plt.ylabel("bits per symbol")
    plt.title("Deep-Space Telemetry: Bits/Symbol Comparison")
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "huffman_bits_per_symbol.png"))
    plt.close()

    print(json.dumps(summary, indent=2))

def huffman_timing_sweep(k_values: List[int], samples_per_symbol:int=2000, seed: int=9) -> pd.DataFrame:
    import time
    rows = []
    for k in k_values:
        _, freq_k = simulate_telemetry_frequencies(k=k, length=k*samples_per_symbol, seed=seed+k)
        t0 = time.perf_counter()
        _ = build_huffman(freq_k)
        t1 = time.perf_counter()
        rows.append({"k": k, "time_sec": (t1 - t0)})
    return pd.DataFrame(rows)

def run_timing(artifacts_dir: str):
    os.makedirs(artifacts_dir, exist_ok=True)
    df = huffman_timing_sweep([8, 16, 32, 64, 128, 256], samples_per_symbol=2000, seed=9)
    df.to_csv(os.path.join(artifacts_dir, "huffman_timing.csv"), index=False)

    # Empirical plot with theoretical O(k log k) reference curve (normalized)
    k_vals = df["k"].to_numpy()
    times = df["time_sec"].to_numpy()
    # Normalize constant using the middle point (k=32) to make curves comparable
    ref_idx = min(2, len(k_vals)-1)
    c = times[ref_idx] / (k_vals[ref_idx] * np.log2(k_vals[ref_idx]))
    theory_time = c * k_vals * np.log2(k_vals)

    plt.figure()
    plt.plot(k_vals, theory_time, 'r--', label='O(k log k) theoretical')
    plt.plot(k_vals, times, 'bo-', label='Empirical')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Alphabet size k (log)")
    plt.ylabel("Build time (sec, log)")
    plt.title("Huffman Construction Empirical Runtime (~O(k log k))")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "huffman_timing_loglog.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Deep-Space Telemetry Compression (Huffman)")
    parser.add_argument("--out", type=str, default="./artifacts_huffman", help="Output directory for plots/CSVs")
    parser.add_argument("--k", type=int, default=32)
    parser.add_argument("--length", type=int, default=120000)
    parser.add_argument("--seed", type=int, default=21)
    parser.add_argument("--timing", action="store_true", help="Also run timing sweep")
    args = parser.parse_args()

    run_end_to_end(args.out, k=args.k, length=args.length, seed=args.seed)
    if args.timing:
        run_timing(args.out)
    print(f"Artifacts written to: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
