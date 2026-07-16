#!/usr/bin/env python3
"""Step-by-step timing breakdown: NumPy (out=) vs MLX for a * b + c.

Shows where time actually goes at two sizes:
  10,000 elements  —  small, GPU launch overhead dominates
  100,000,000      —  large, memory bandwidth dominates

Run:
  python examples/04_perf_breakdown.py
"""

import time
from statistics import mean, stdev

import mlx.core as mx
import numpy as np

WARMUP = 5
REPEATS = 30


def t_fn(fn):
    t0 = time.perf_counter()
    r = fn()
    return r, (time.perf_counter() - t0) * 1e6  # μs


# ── NumPy breakdown ────────────────────────────────────────────────────────────

def numpy_breakdown(a, b, c, o):
    steps = ["empty_like", "np.multiply", "np.add"]
    times: dict[str, list[float]] = {s: [] for s in steps}

    for _ in range(WARMUP):
        _buf = np.empty_like(o)
        np.multiply(a, b, out=_buf)
        np.add(_buf, c, out=o)

    for _ in range(REPEATS):
        _, t = t_fn(lambda: np.empty_like(o)); times["empty_like"].append(t)
        _buf = np.empty_like(o)

        _, t = t_fn(lambda: np.multiply(a, b, out=_buf)); times["np.multiply"].append(t)
        v2 = np.multiply(a, b, out=_buf)

        _, t = t_fn(lambda: np.add(v2, c, out=o)); times["np.add"].append(t)
        np.add(v2, c, out=o)

    return {k: (mean(v), stdev(v)) for k, v in times.items()}


# ── MLX breakdown ─────────────────────────────────────────────────────────────

def mlx_breakdown(a, b, c, o):
    steps = ["mx.array(a)", "mx.array(b)", "mul (lazy)",
             "mx.array(c)", "add (lazy)", "mx.eval", "np.array (d2h)"]
    times: dict[str, list[float]] = {s: [] for s in steps}

    for _ in range(WARMUP):
        v0, v1 = mx.array(a), mx.array(b)
        v2 = v0 * v1
        v3 = mx.array(c)
        v4 = v2 + v3
        mx.eval(v4)
        o[...] = np.array(v4)

    for _ in range(REPEATS):
        _, t = t_fn(lambda: mx.array(a)); times["mx.array(a)"].append(t)
        v0 = mx.array(a)

        _, t = t_fn(lambda: mx.array(b)); times["mx.array(b)"].append(t)
        v1 = mx.array(b)

        _, t = t_fn(lambda: v0 * v1); times["mul (lazy)"].append(t)
        v2 = v0 * v1

        _, t = t_fn(lambda: mx.array(c)); times["mx.array(c)"].append(t)
        v3 = mx.array(c)

        _, t = t_fn(lambda: v2 + v3); times["add (lazy)"].append(t)
        v4 = v2 + v3

        _, t = t_fn(lambda: mx.eval(v4)); times["mx.eval"].append(t)
        mx.eval(v4)

        _, t = t_fn(lambda: np.array(v4)); times["np.array (d2h)"].append(t)
        o[...] = np.array(v4)

    return {k: (mean(v), stdev(v)) for k, v in times.items()}


# ── Display ────────────────────────────────────────────────────────────────────

def print_breakdown(size: int):
    sz_label = f"{size:,}"
    mb = size * 4 / 1024 / 1024
    print(f"\n{'═'*70}")
    print(f"  {sz_label} elements  ({mb:.1f} MB per array, {3*mb:.0f} MB total input)")
    print(f"{'═'*70}\n")

    rng = np.random.default_rng(0)
    a = rng.random(size, dtype=np.float32)
    b = rng.random(size, dtype=np.float32)
    c = rng.random(size, dtype=np.float32)
    o_np  = np.zeros(size, dtype=np.float32)
    o_mlx = np.zeros(size, dtype=np.float32)

    np_times  = numpy_breakdown(a, b, c, o_np)
    mlx_times = mlx_breakdown(a, b, c, o_mlx)

    W = 24
    print(f"  {'Step':<{W}}  {'Mean':>10}  {'±':>8}  {'Notes'}")
    print(f"  {'-'*W}  {'-'*10}  {'-'*8}  {'-'*32}")

    print(f"\n  NumPy (out= ufuncs, vectorized C)")
    np_total = 0.0
    for step, (mu, sd) in np_times.items():
        print(f"  {step:<{W}}  {mu:>8.1f}μs  {sd:>6.1f}μs")
        np_total += mu
    print(f"  {'TOTAL':<{W}}  {np_total:>8.1f}μs")

    print(f"\n  MLX (lazy graph → Metal GPU)")
    mlx_total = 0.0
    for step, (mu, sd) in mlx_times.items():
        note = ""
        if "lazy" in step:
            note = "← no GPU work, builds graph"
        elif step == "mx.eval":
            note = "← GPU executes here"
        elif "d2h" in step:
            note = "← writeback to CPU"
        elif "mx.array" in step:
            note = "← host→device"
        print(f"  {step:<{W}}  {mu:>8.1f}μs  {sd:>6.1f}μs  {note}")
        mlx_total += mu
    print(f"  {'TOTAL':<{W}}  {mlx_total:>8.1f}μs")

    winner = "NumPy" if np_total < mlx_total else "MLX"
    ratio  = max(np_total, mlx_total) / min(np_total, mlx_total)
    print(f"\n  → {winner} is {ratio:.1f}x faster at this size\n")

    # Show what fraction of MLX time is transfer vs compute
    h2d = sum(mu for s, (mu, _) in mlx_times.items() if "mx.array" in s)
    d2h = sum(mu for s, (mu, _) in mlx_times.items() if "d2h" in s)
    compute = sum(mu for s, (mu, _) in mlx_times.items() if s == "mx.eval")
    overhead = mlx_total - h2d - d2h - compute
    print(f"  MLX time breakdown:")
    print(f"    h2d transfers : {h2d:>8.1f}μs  ({100*h2d/mlx_total:.0f}%)")
    print(f"    GPU compute   : {compute:>8.1f}μs  ({100*compute/mlx_total:.0f}%)")
    print(f"    d2h writeback : {d2h:>8.1f}μs  ({100*d2h/mlx_total:.0f}%)")
    print(f"    other overhead: {overhead:>8.1f}μs  ({100*overhead/mlx_total:.0f}%)")


if __name__ == "__main__":
    print("a * b + c  —  per-step timing breakdown")
    print_breakdown(10_000)
    print_breakdown(100_000_000)
