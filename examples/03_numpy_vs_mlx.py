#!/usr/bin/env python3
"""Compare NumPy (CPU) vs MLX (Metal GPU) for a * b + c."""

import time

import numpy as np

import picokernel


@picokernel.kernel
def numpy_kernel(a, b, c, o):
  o[...] = a[...] * b[...] + c[...]


@picokernel.kernel(backend="mlx")
def mlx_kernel(a, b, c, o):
  o[...] = a[...] * b[...] + c[...]


def bench(fn, *arrays, warmup=3, repeats=20):
  for _ in range(warmup):
    fn(*arrays)
  t0 = time.perf_counter()
  for _ in range(repeats):
    fn(*arrays)
  return (time.perf_counter() - t0) / repeats


def run(size: int):
  rng = np.random.default_rng(0)
  a = rng.random(size, dtype=np.float32)
  b = rng.random(size, dtype=np.float32)
  c = rng.random(size, dtype=np.float32)
  out_np  = np.zeros(size, dtype=np.float32)
  out_mlx = np.zeros(size, dtype=np.float32)

  t_np  = bench(numpy_kernel, a, b, c, out_np)
  t_mlx = bench(mlx_kernel,   a, b, c, out_mlx)

  np.testing.assert_allclose(out_np, out_mlx, rtol=1e-5)

  speedup = t_np / t_mlx
  winner = "MLX" if speedup > 1 else "NumPy"
  print(f"  size={size:>10,}  numpy={t_np*1e3:7.3f}ms  mlx={t_mlx*1e3:6.3f}ms  "
        f"{winner} {max(speedup, 1/speedup):.1f}x faster")


if __name__ == "__main__":
  a0, b0, c0, o0 = (np.zeros(4, dtype=np.float32) for _ in range(4))

  print("a * b + c  —  numpy vs mlx\n")
  print("--- numpy (out= ufuncs, vectorized C) ---")
  print(numpy_kernel.lower(a0, b0, c0, o0))
  print("\n--- mlx (Metal GPU) ---")
  print(mlx_kernel.lower(a0, b0, c0, o0))
  print()

  curr_size = 1_000
  max_size = 100_000_000
  while curr_size <= max_size:
    run(curr_size)
    curr_size *= 10
