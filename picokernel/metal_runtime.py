"""Execution: compile a KernelIR to a Metal kernel and dispatch it.

Unlike the NumPy/MLX backends (which exec() Python source in-process), Metal runs
out of process on the GPU. The path is:

  1. lower_to_loops -> lower_to_metal   : KernelIR -> MSL source
  2. device.kernel(src).function(name)  : runtime MSL compile (no Xcode/metallib)
  3. per call: allocate unified-memory buffers, copy inputs in (cast to float32),
     dispatch `grid` threads, copy the output buffer back into the user's array.

metalcompute buffers are unified memory shared with the GPU; np.frombuffer gives a
zero-copy view we fill before dispatch and read after.
"""

from math import prod
from typing import Callable

import numpy as np

from .core import KernelIR

_device = None
_cache: dict[int, tuple[KernelIR, Callable]] = {}


def _get_device():
  global _device
  if _device is None:
    import metalcompute as mc

    _device = mc.Device()
  return _device


def compile_metal(ir: KernelIR) -> Callable:
  """Compile a KernelIR to a callable that runs on the Metal GPU (float32)."""
  key = id(ir)
  if key in _cache and _cache[key][0] is ir:
    return _cache[key][1]

  from .loop_ir import lower_to_loops
  from .metal_lowering import lower_to_metal

  prog = lower_to_loops(ir)
  source = lower_to_metal(prog)

  dev = _get_device()
  metal_fn = dev.kernel(source).function(ir.name)

  buffers = prog.buffers
  grid = prog.grid
  out_shape = prog.out_shape
  out_numel = prod(out_shape) if out_shape else 1
  ref_index = {name: i for i, name in enumerate(ir.ref_params)}
  const_data = {
    b.name: np.ascontiguousarray(b.const_value, dtype=np.float32).ravel()
    for b in buffers
    if b.role == "const"
  }

  def run(*arrays):
    mc_buffers = []
    out_view = None
    for b in buffers:
      if b.role == "input":
        data = np.ascontiguousarray(arrays[ref_index[b.name]], dtype=np.float32).ravel()
      elif b.role == "const":
        data = const_data[b.name]
      else:  # output
        data = None

      size = out_numel if b.role == "output" else data.size
      mbuf = dev.buffer(size * 4)  # float32
      view = np.frombuffer(mbuf, dtype=np.float32)
      if data is not None:
        view[:] = data
      else:
        out_view = view
      mc_buffers.append(mbuf)

    metal_fn(grid, *mc_buffers)

    out_arr = arrays[ref_index[prog.out_buffer]]
    out_arr[...] = out_view.reshape(out_shape).astype(out_arr.dtype)

  _cache[key] = (ir, run)
  return run
