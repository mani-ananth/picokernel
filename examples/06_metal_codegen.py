#!/usr/bin/env python3
"""Metal backend: inspect generated MSL and run it on the GPU.

The metal backend lowers through a mid-level loop IR before codegen:

    KernelIR (array ops) -> loop IR (grid + scalar body) -> MSL -> GPU

Requires metalcompute (pip install metalcompute) and Apple Silicon.
"""

import numpy as np
from pprint import pprint

import picokernel
from picokernel.core import pretty_print
from picokernel.metal_lowering import lower_to_metal
from picokernel.loop_ir import lower_to_loops
from picokernel.trace import trace_kernel


@picokernel.kernel(backend="metal")
def fma(a, b, c, o):
  o[...] = a[...] * b[...] + c[...]


@picokernel.kernel(backend="metal")
def matmul(a, b, o):
  o[...] = a[...] @ b[...]


if __name__ == "__main__":
  a, b, c = (np.random.rand(8).astype(np.float32) for _ in range(3))
  o = np.zeros(8, dtype=np.float32)

  print("=== elementwise fma: a*b+c ===")
  fma(a, b, c, o)
  print("\nGPU result :", o)
  print("NumPy check:", a * b + c)
  
  loops = lower_to_loops(fma.ir)
  msl = lower_to_metal(loops)
  pprint("--- generated KernelIR ---")
  pprint(pretty_print(fma.ir))
  pprint("--- generated loops ---")
  pprint(loops)
  pprint("--- generated MSL ---")
  pprint(fma.lower(a, b, c, o))
  pprint(msl)

  # print("\n=== matmul: (2x3) @ (3x4) ===")
  # am = np.arange(6, dtype=np.float32).reshape(2, 3)
  # bm = np.arange(12, dtype=np.float32).reshape(3, 4)
  # om = np.zeros((2, 4), dtype=np.float32)
  # print("--- generated MSL ---")
  # print(matmul.lower(am, bm, om))
  # matmul(am, bm, om)
  # print("\nGPU result:\n", om)
  # print("NumPy check:\n", am @ bm)
