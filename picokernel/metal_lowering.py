"""Lowering: convert a loop-level program to Metal Shading Language source.

Metal compute kernels run one thread per grid point. Both schedules below use a
1D grid indexed by `thread_position_in_grid`; shape-specialized dims (numel, M,
N, K) are baked in as literals since the IR is retraced per shape anyway.

Metal has no float64 - everything is float32. The runtime casts on the buffer
boundary; this emitter only ever sees float.
"""

from .core import OpType
from .loop_ir import ElementwiseProgram, MatmulProgram, ScalarOp

_BINOP = {
  OpType.ADD: "+",
  OpType.SUB: "-",
  OpType.MUL: "*",
  OpType.TRUEDIV: "/",
}


def _ptr(buf, mutable: bool, index: int) -> str:
  qual = "device float*" if mutable else "device const float*"
  return f"    {qual} {buf.name} [[buffer({index})]]"


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


def lower_to_metal(prog) -> str:
  """Generate MSL source for a loop program. Function name == prog.name."""
  if isinstance(prog, ElementwiseProgram):
    return _lower_elementwise(prog)
  if isinstance(prog, MatmulProgram):
    return _lower_matmul(prog)
  raise NotImplementedError(f"No Metal lowering for {type(prog).__name__}")


def _signature(name: str, buffers) -> str:
  params = [
    _ptr(b, mutable=(b.role == "output"), index=i)
    for i, b in enumerate(buffers)
  ]
  params.append("    uint i [[thread_position_in_grid]]")
  joined = ",\n".join(params)
  return f"kernel void {name}(\n{joined})"


def _lower_elementwise(prog: ElementwiseProgram) -> str:
  lines = ["#include <metal_stdlib>", "using namespace metal;", ""]
  lines.append(_signature(prog.name, prog.buffers) + " {")
  lines.append(f"  if (i >= {prog.grid}u) return;")
  for op in prog.body:
    lines.append(_emit_scalar(op))
  lines.append(f"  {prog.out_buffer}[i] = {prog.out_temp};")
  lines.append("}")
  return "\n".join(lines)


def _lower_matmul(prog: MatmulProgram) -> str:
  lines = ["#include <metal_stdlib>", "using namespace metal;", ""]
  lines.append(_signature(prog.name, prog.buffers) + " {")
  lines.append(f"  if (i >= {prog.grid}u) return;")
  lines.append(f"  uint row = i / {prog.N}u;")
  lines.append(f"  uint col = i % {prog.N}u;")
  lines.append("  float acc = 0.0f;")
  lines.append(f"  for (uint k = 0u; k < {prog.K}u; k++) {{")
  lines.append(
    f"    acc += {prog.a_buffer}[row * {prog.K}u + k]"
    f" * {prog.b_buffer}[k * {prog.N}u + col];"
  )
  lines.append("  }")
  lines.append(f"  {prog.out_buffer}[i] = acc;")
  lines.append("}")
  return "\n".join(lines)
