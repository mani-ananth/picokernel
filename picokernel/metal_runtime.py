"""Execution: compile a KernelIR to Metal kernels and dispatch them in order.

Unlike the NumPy/MLX backends (which exec() Python source in-process), Metal runs
out of process on the GPU. A kernel lowers to an ordered list of segments (see
loop_ir.py); the path is:

  1. lower_to_loops -> lower_to_metal   : KernelIR -> MSL source (one kernel/segment)
  2. device.kernel(src)                 : runtime compile (no Xcode/metallib); each
                                          segment's function fetched via .function(name)
  3. per call: allocate unified-memory buffers (inputs copied in, consts baked,
     intermediates device-only, output allocated), dispatch each segment's grid in
     order, then copy the output buffer back into the user's array.

metalcompute buffers are unified memory shared with the GPU; np.frombuffer gives a
zero-copy view we fill before dispatch and read after. Sequential dispatches over a
shared intermediate buffer are ordered, so a later segment sees an earlier one's
writes without an explicit barrier.
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

  lowered = lower_to_loops(ir)
  source = lower_to_metal(lowered)

  dev = _get_device()
  program = dev.kernel(source)
  # (function, grid, [buffer names in dispatch order]) per segment
  segments = [
    (program.function(seg.name), seg.grid, [b.name for b in seg.buffers])
    for seg in lowered.segments
  ]

  ref_index = {name: i for i, name in enumerate(ir.ref_params)}
  intermediates = dict(lowered.intermediates)  # name -> numel
  out_buffer = lowered.out_buffer
  out_shape = lowered.out_shape
  out_numel = prod(out_shape) if out_shape else 1

  # Static per-buffer metadata: role, and for consts the baked data.
  meta: dict[str, tuple] = {}  # name -> ("input"|"const"|"intermediate"|"output", data)
  for seg in lowered.segments:
    for b in seg.buffers:
      if b.name in meta:
        continue
      if b.role == "const":
        data = np.ascontiguousarray(b.const_value, dtype=np.float32).ravel()
        meta[b.name] = ("const", data)
      else:
        meta[b.name] = (b.role, None)

  def run(*arrays):
    bufs: dict[str, object] = {}
    out_view = None
    for name, (role, data) in meta.items():
      if role == "input":
        arr = np.ascontiguousarray(arrays[ref_index[name]], dtype=np.float32).ravel()
        size = arr.size
      elif role == "const":
        arr = data
        size = arr.size
      elif role == "intermediate":
        arr = None
        size = intermediates[name]
      else:  # output
        arr = None
        size = out_numel

      mbuf = dev.buffer(size * 4)  # float32
      view = np.frombuffer(mbuf, dtype=np.float32)
      if arr is not None:
        view[:] = arr
      if role == "output":
        out_view = view
      bufs[name] = mbuf

    for fn, grid, names in segments:
      fn(grid, *(bufs[n] for n in names))

    out_arr = arrays[ref_index[out_buffer]]
    out_arr[...] = out_view.reshape(out_shape).astype(out_arr.dtype)

  _cache[key] = (ir, run)
  return run
