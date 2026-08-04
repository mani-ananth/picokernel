"""Execution: compile lowered source and run it."""

from typing import Callable

import numpy as np

from .core import KernelIR
from .lowering import lower_to_numpy
from .metal_runtime import compile_metal
from .mlx_lowering import lower_to_mlx

_numpy_cache: dict[int, tuple[KernelIR, Callable]] = {}
_mlx_cache: dict[int, tuple[KernelIR, Callable]] = {}


def compile_numpy(ir: KernelIR) -> Callable:
  """Compile a KernelIR to a callable NumPy function (vectorized C, out= lowering)."""
  key = id(ir)
  if key in _numpy_cache and _numpy_cache[key][0] is ir:
    return _numpy_cache[key][1]

  source = lower_to_numpy(ir)
  namespace: dict = {"np": np}
  exec(source, namespace)
  fn = namespace[ir.name]
  _numpy_cache[key] = (ir, fn)
  return fn


def compile_mlx(ir: KernelIR) -> Callable:
  """Compile a KernelIR to a callable MLX function (runs on Metal/GPU)."""
  import mlx.core as mx

  key = id(ir)
  if key in _mlx_cache and _mlx_cache[key][0] is ir:
    return _mlx_cache[key][1]

  source = lower_to_mlx(ir)
  namespace: dict = {"mx": mx, "np": np}
  exec(source, namespace)
  fn = namespace[ir.name]
  _mlx_cache[key] = (ir, fn)
  return fn
