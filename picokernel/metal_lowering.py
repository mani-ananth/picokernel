"""Lowering: convert a lowered program to Metal Shading Language source.

A kernel lowers to an ordered list of segments (see loop_ir.py); each segment
becomes one `kernel void` in the emitted source, dispatched in turn by the
runtime. Metal compute kernels run one thread per grid point; both schedules use
a 1D grid indexed by `thread_position_in_grid`, with shape dims (numel, M, N, K)
baked in as literals since the IR is retraced per shape anyway.

A buffer is `device float*` (mutable) if the segment writes it, else
`device const float*`. The same intermediate is const in the segment that reads
it and mutable in the segment that wrote it.

Metal has no float64 - everything is float32. The runtime casts on the buffer
boundary; this emitter only ever sees float.
"""

from .core import OpType
from .loop_ir import ElementwiseProgram, LoweredProgram, MatmulProgram, ScalarOp

_BINOP = {
  OpType.ADD: "+",
  OpType.SUB: "-",
  OpType.MUL: "*",
  OpType.TRUEDIV: "/",
}


def lower_to_metal(lowered: LoweredProgram) -> str:
  """Generate MSL source for a lowered program (one kernel per segment)."""
  kernels = [_lower_segment(seg) for seg in lowered.segments]
  return "\n".join(["#include <metal_stdlib>", "using namespace metal;", "", *kernels])


def _lower_segment(seg) -> str:
  if isinstance(seg, ElementwiseProgram):
    return _lower_elementwise(seg)
  if isinstance(seg, MatmulProgram):
    return _lower_matmul(seg)
  raise NotImplementedError(f"No Metal lowering for {type(seg).__name__}")


def _fmt_literal(value) -> str:
  # repr() always carries a '.' or 'e', so "2.0f" / "1e-09f" are valid MSL
  # (unlike "2f", which a bare %g would produce).
  return f"{float(value)!r}f"


def _emit_scalar(op: ScalarOp) -> str:
  if op.op == OpType.LOAD:
    return f"  float {op.result} = {op.buffer}[i];"
  if op.op == OpType.CONST:
    return f"  float {op.result} = {_fmt_literal(op.literal)};"
  if op.op == OpType.NEG:
    return f"  float {op.result} = -{op.args[0]};"
  if op.op in _BINOP:
    a, b = op.args
    return f"  float {op.result} = {a} {_BINOP[op.op]} {b};"
  raise NotImplementedError(f"Metal backend cannot lower scalar op {op.op}")


def _signature(name: str, buffers, writes) -> str:
  params = []
  for idx, b in enumerate(buffers):
    qual = "device float*" if b.name in writes else "device const float*"
    params.append(f"    {qual} {b.name} [[buffer({idx})]]")
  params.append("    uint i [[thread_position_in_grid]]")
  return f"kernel void {name}(\n" + ",\n".join(params) + ")"


def _lower_elementwise(seg: ElementwiseProgram) -> str:
  lines = [_signature(seg.name, seg.buffers, seg.writes) + " {"]
  lines.append(f"  if (i >= {seg.grid}u) return;")
  for op in seg.body:
    lines.append(_emit_scalar(op))
  for out_temp, out_buffer in seg.outputs:
    lines.append(f"  {out_buffer}[i] = {out_temp};")
  lines.append("}")
  return "\n".join(lines)


def _lower_matmul(seg: MatmulProgram) -> str:
  lines = [_signature(seg.name, seg.buffers, seg.writes) + " {"]
  lines.append(f"  if (i >= {seg.grid}u) return;")
  lines.append(f"  uint row = i / {seg.N}u;")
  lines.append(f"  uint col = i % {seg.N}u;")
  lines.append("  float acc = 0.0f;")
  lines.append(f"  for (uint k = 0u; k < {seg.K}u; k++) {{")
  lines.append(
    f"    acc += {seg.a_buffer}[row * {seg.K}u + k]"
    f" * {seg.b_buffer}[k * {seg.N}u + col];"
  )
  lines.append("  }")
  lines.append(f"  {seg.out_buffer}[i] = acc;")
  lines.append("}")
  return "\n".join(lines)
