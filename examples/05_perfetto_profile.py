#!/usr/bin/env python3
"""Generate Perfetto trace files for both backends.

Usage:
  python examples/05_perfetto_profile.py [--size N]

Output files (open in ui.perfetto.dev):
  numpy_trace.json    — numpy backend, per-op spans
  mlx_trace.json      — mlx backend, h2d / op_lazy / sync / d2h spans
  comparison.json     — both merged, PID 1 = numpy, PID 2 = mlx

Viewing:
  1. Go to https://ui.perfetto.dev
  2. Click "Open trace file" and upload any of the above files
  3. comparison.json shows both backends as separate timeline rows
"""

import argparse

import numpy as np

import picokernel
from picokernel.profiler import Profiler


@picokernel.kernel
def k(a, b, c, o):
    o[...] = a[...] * b[...] + c[...]


@picokernel.kernel(backend="mlx")
def k_mlx(a, b, c, o):
    o[...] = a[...] * b[...] + c[...]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=100_000,
                        help="Number of elements (default: 100_000)")
    parser.add_argument("--repeats", type=int, default=5,
                        help="Timed repetitions per backend (default: 5)")
    args = parser.parse_args()

    size, repeats = args.size, args.repeats
    rng = np.random.default_rng(0)
    a = rng.random(size, dtype=np.float32)
    b = rng.random(size, dtype=np.float32)
    c = rng.random(size, dtype=np.float32)
    o_np  = np.zeros(size, dtype=np.float32)
    o_mlx = np.zeros(size, dtype=np.float32)

    mb = size * 4 / 1024 / 1024
    print(f"\nProfiling a * b + c  —  {size:,} elements ({mb:.1f} MB/array), {repeats} repeats\n")

    # Warmup — exclude JIT/shader-compile cost from the trace
    print("Warming up...")
    for _ in range(3):
        k(a, b, c, o_np)
        k_mlx(a, b, c, o_mlx)

    print(f"\nProfiling numpy  ({repeats} repeats)...")
    p_numpy = k.run_profiled(a, b, c, o_np, n_repeats=repeats)

    print(f"Profiling mlx    ({repeats} repeats)...")
    p_mlx = k_mlx.run_profiled(a, b, c, o_mlx, n_repeats=repeats)

    print("\nSaving traces:")
    p_numpy.save("numpy_trace.json")
    p_mlx.save("mlx_trace.json")
    Profiler.merge(p_numpy, p_mlx, name="numpy vs mlx").save("comparison.json")

    print("\nTo view in Perfetto:")
    print("  1. Go to https://ui.perfetto.dev")
    print("  2. Click 'Open trace file'")
    print("  3. Upload any of the JSON files above")
    print("     comparison.json → PID 1 = numpy, PID 2 = mlx (side-by-side)")
    print("\nEvent categories in the trace:")
    print("  alloc    — scratch buffer allocation (numpy)")
    print("  op       — ufunc execution (numpy, vectorized C)")
    print("  h2d      — host→device transfer (mlx, mx.array)")
    print("  op_lazy  — lazy graph node (mlx, ~0μs, no GPU work)")
    print("  sync     — mx.eval(), where GPU actually executes")
    print("  d2h      — device→host writeback (mlx, np.array)")
    print("  runtime  — total kernel_call span (wraps everything)")


if __name__ == "__main__":
    main()
