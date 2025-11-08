#!/usr/bin/env python3

from __future__ import annotations
import os, argparse, math, json, time
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ------------------ Core Quadtree ------------------

@dataclass
class QuadNode:
    x: int
    y: int
    size: int
    leaf: bool
    value: Optional[float] = None
    children: Optional[Tuple['QuadNode','QuadNode','QuadNode','QuadNode']] = None

def is_uniform(block: np.ndarray, tol: float) -> Tuple[bool, float]:
    vmin = float(block.min())
    vmax = float(block.max())
    if vmax - vmin <= tol:
        return True, float(block.mean())
    return False, float(block.mean())

def build_quadtree(grid: np.ndarray, x: int, y: int, size: int, tol: float) -> QuadNode:
    block = grid[y:y+size, x:x+size]
    uniform, rep = is_uniform(block, tol)
    if uniform or size == 1:
        return QuadNode(x=x, y=y, size=size, leaf=True, value=rep)
    half = size // 2
    nw = build_quadtree(grid, x, y, half, tol)
    ne = build_quadtree(grid, x+half, y, half, tol)
    sw = build_quadtree(grid, x, y+half, half, tol)
    se = build_quadtree(grid, x+half, y+half, half, tol)
    return QuadNode(x=x, y=y, size=size, leaf=False, children=(nw, ne, sw, se))

def reconstruct_from_quadtree(node: QuadNode, out: np.ndarray):
    if node.leaf:
        out[node.y:node.y+node.size, node.x:node.x+node.size] = node.value
        return
    for child in node.children:
        reconstruct_from_quadtree(child, out)

def pad_to_power_of_two(grid: np.ndarray) -> Tuple[np.ndarray, int]:
    h, w = grid.shape
    n = 1 << (max(h, w) - 1).bit_length()
    if h == n and w == n:
        return grid.copy(), n
    padded = np.zeros((n, n), dtype=grid.dtype)
    padded[:h, :w] = grid
    if h < n:
        padded[h:n, :w] = grid[h-1:h, :w]
    if w < n:
        padded[:, w:n] = padded[:, w-1:w]
    return padded, n

def count_nodes(node: QuadNode) -> int:
    if node.leaf:
        return 1
    return 1 + sum(count_nodes(c) for c in node.children)

# ------------------ Synthetic Weather ------------------

def generate_weather_grid(n: int, hotspots: int=3, amplitude: float=50.0, noise: float=2.0, seed: Optional[int]=None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    grid = np.zeros((n, n), dtype=float)
    centers = [(rng.integers(0, n), rng.integers(0, n)) for _ in range(hotspots)]
    for cx, cy in centers:
        sigma = rng.uniform(n/16, n/6)
        y_idx, x_idx = np.indices((n, n))
        bump = amplitude * np.exp(-(((x_idx-cx)**2 + (y_idx-cy)**2) / (2*sigma*sigma)))
        grid += bump
    grid += rng.normal(0.0, noise, size=(n, n))
    grid = np.clip(grid, 0.0, None)
    return grid

# ------------------ Experiments ------------------

def verify_and_reconstruct(artifacts_dir: str, n: int=256, tol: float=1.0, seed: int=7):
    os.makedirs(artifacts_dir, exist_ok=True)
    grid = generate_weather_grid(n=n, hotspots=3, amplitude=50.0, noise=2.0, seed=seed)
    padded, N = pad_to_power_of_two(grid)
    qt = build_quadtree(padded, 0, 0, N, tol)
    recon = np.zeros_like(padded)
    reconstruct_from_quadtree(qt, recon)
    mse = float(np.mean((padded - recon)**2))
    maxe = float(np.max(np.abs(padded - recon)))
    stats = {"n": n, "padded_n": N, "tol": tol, "mse": mse, "max_abs_error": maxe, "nodes": count_nodes(qt)}
    pd.DataFrame([stats]).to_csv(os.path.join(artifacts_dir, "quadtree_verify.csv"), index=False)

    # Save heatmaps of original and reconstructed arrays for the report
    import matplotlib.pyplot as plt
    plt.figure()
    plt.imshow(padded)
    plt.colorbar()
    plt.title("Original Grid (padded)")
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "quadtree_original.png"))
    plt.close()

    plt.figure()
    plt.imshow(recon)
    plt.colorbar()
    plt.title("Reconstructed from Quadtree")
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "quadtree_reconstructed.png"))
    plt.close()

    print(json.dumps(stats, indent=2))

def timing_quadtree_scaling(artifacts_dir: str, sizes: List[int], tol: float=1.0, seed: int=123):
    os.makedirs(artifacts_dir, exist_ok=True)
    rows = []
    for n in sizes:
        grid = generate_weather_grid(n=n, hotspots=3, amplitude=50.0, noise=2.0, seed=seed+n)
        padded, N = pad_to_power_of_two(grid)
        t0 = time.perf_counter()
        qt = build_quadtree(padded, 0, 0, N, tol)
        t1 = time.perf_counter()
        rows.append({"n": n, "padded_n": N, "time_sec": t1 - t0, "nodes": count_nodes(qt)})
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(artifacts_dir, "quadtree_timing.csv"), index=False)

    # Empirical plot with theoretical O(n^2) and O(n^2 log n) reference curves
    n_vals = df["padded_n"].to_numpy()
    times = df["time_sec"].to_numpy()
    # Normalize constants using a middle point (e.g., second point) to compare shapes
    ref_idx = min(1, len(n_vals)-1)
    c1 = times[ref_idx] / (n_vals[ref_idx]**2)
    c2 = times[ref_idx] / (n_vals[ref_idx]**2 * np.log2(n_vals[ref_idx]))
    theory_n2 = c1 * n_vals**2
    theory_n2logn = c2 * n_vals**2 * np.log2(n_vals)

    plt.figure()
    plt.plot(n_vals, theory_n2, 'g--', label='O(n^2) bound')
    plt.plot(n_vals, theory_n2logn, 'r--', label='O(n^2 log n) bound')
    plt.plot(n_vals, times, 'bo-', label='Empirical')
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("grid size (padded) (log)")
    plt.ylabel("time_sec (log)")
    plt.title("Quadtree Build: Empirical Runtime")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(artifacts_dir, "quadtree_timing_loglog.png"))
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Weather Station Monitoring via Quadtree")
    parser.add_argument("--out", type=str, default="./artifacts_quadtree", help="Output directory for plots/CSVs")
    parser.add_argument("--n", type=int, default=256, help="Base grid size before padding to power of two")
    parser.add_argument("--tol", type=float, default=1.0, help="Uniformity tolerance")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--timing", action="store_true", help="Run timing sweep as well")
    args = parser.parse_args()

    verify_and_reconstruct(args.out, n=args.n, tol=args.tol, seed=args.seed)
    if args.timing:
        timing_quadtree_scaling(args.out, sizes=[64, 128, 256, 512], tol=args.tol, seed=123)
    print(f"Artifacts written to: {os.path.abspath(args.out)}")

if __name__ == "__main__":
    main()
