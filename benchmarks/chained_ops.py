#!/usr/bin/env python3
"""Chained-op benchmark: keep data on the device between ops.

The transparent MLX backend pays h2d/d2h transfers on every kernel call.
This benchmark measures the device-resident alternative: transfer inputs
once, chain N iterations of x = x * b + c without leaving the device,
and transfer the result back once. NumPy runs the same chain with in-place
out= ufuncs (no intermediate allocations), matching what lower_to_numpy
generates for a single step.

Usage:
  python benchmarks/chained_ops.py [--size N] [--chains 1,2,4,8,16] [--json out.json]
"""

import argparse
import json
import statistics
import time

import mlx.core as mx
import numpy as np


def numpy_chain(a, b, c, o, n):
  np.multiply(a, b, out=o)
  np.add(o, c, out=o)
  for _ in range(n - 1):
    np.multiply(o, b, out=o)
    np.add(o, c, out=o)


def mlx_chain(a, b, c, o, n):
  vb, vc = mx.array(b), mx.array(c)   # h2d, paid once
  x = mx.array(a)                     # h2d, paid once
  for _ in range(n):
    x = x * vb + vc                   # lazy, stays on device
  mx.eval(x)                          # GPU executes the whole chain
  o[...] = np.array(x)                # d2h, paid once


def bench(fn, *args, warmup=1, rounds=3):
  for _ in range(warmup):
    fn(*args)
  times = []
  for _ in range(rounds):
    t0 = time.perf_counter()
    fn(*args)
    times.append(time.perf_counter() - t0)
  return statistics.median(times)


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--size", type=int, default=100_000_000,
    help="elements per array (default: 100,000,000)")
  parser.add_argument("--chains", default="1,2,4,8,16",
    help="comma-separated chain lengths (default: 1,2,4,8,16)")
  parser.add_argument("--rounds", type=int, default=3,
    help="timed rounds per point (default: 3)")
  parser.add_argument("--json", help="also write results to this JSON file")
  args = parser.parse_args()

  chains = [int(n) for n in args.chains.split(",")]
  rng = np.random.default_rng(0)
  a = rng.random(args.size, dtype=np.float32)
  b = rng.random(args.size, dtype=np.float32)
  c = rng.random(args.size, dtype=np.float32)
  o_np = np.zeros(args.size, dtype=np.float32)
  o_mx = np.zeros(args.size, dtype=np.float32)

  print(f"x = x * b + c chained N times — {args.size:,} float32 elements\n")
  results = []
  for n in chains:
    t_np = bench(numpy_chain, a, b, c, o_np, n, rounds=args.rounds)
    t_mx = bench(mlx_chain, a, b, c, o_mx, n, rounds=args.rounds)
    np.testing.assert_allclose(o_np, o_mx, rtol=1e-4)
    faster = "MLX" if t_mx < t_np else "NumPy"
    ratio = max(t_np, t_mx) / min(t_np, t_mx)
    print(f"  N={n:>3}  numpy={t_np*1e3:8.1f}ms  mlx={t_mx*1e3:8.1f}ms  "
          f"{faster} {ratio:.1f}x faster")
    results.append({"n": n, "numpy_ms": t_np * 1e3, "mlx_ms": t_mx * 1e3})

  if args.json:
    with open(args.json, "w") as f:
      json.dump({"size": args.size, "results": results}, f, indent=2)
    print(f"\nwrote {args.json}")


if __name__ == "__main__":
  main()
