"""Mid-level loop IR: the bridge between array ops and target codegen.

This is where MLIR-style progressive lowering starts. KernelIR is array-level
(`ADD`, `MUL`, `MATMUL` over whole arrays, 1:1 with NumPy/MLX calls). Targets
like C or Metal need explicit per-element loops instead, so we lower into a small
loop dialect first:

    KernelIR (array ops)
        -> lower_to_loops()
    LoopProgram (a parallel grid + a scalar body, or a matmul schedule)
        -> lower_to_metal() / (future) lower_to_c()
    target source

Two schedules cover the current op set:

  * ElementwiseProgram - a flat grid of `numel` independent threads, each running
    a straight-line scalar body (one ScalarOp per array op).
  * MatmulProgram - a grid of M*N threads, each accumulating a K-length reduction.

Anything outside this (broadcasting between mismatched shapes, matmul fused with
elementwise, >2D matmul) raises NotImplementedError rather than silently
miscompiling - this backend is a teaching vehicle, not a complete compiler.
"""

from dataclasses import dataclass, field
from math import prod
from typing import Any, Optional

import numpy as np

from .core import KernelIR, OpType


@dataclass
class Buffer:
  """A GPU buffer the kernel reads or writes.

  role is one of "input" (a kernel ref param), "const" (a baked-in array
  constant, carried in const_value), or "output" (the STORE target).
  """

  name: str
  role: str
  const_value: Any = None


@dataclass
class ScalarOp:
  """One operation in the per-element body, in SSA form over scalar temps.

  - LOAD:  result = buffer[i]        (buffer set, args empty)
  - CONST: result = literal          (literal set)
  - binop/NEG: result = op(args...)  (args are temp names)
  """

  result: str
  op: OpType
  args: tuple = ()
  buffer: Optional[str] = None
  literal: Any = None


@dataclass
class ElementwiseProgram:
  name: str
  grid: int  # number of threads == output element count
  out_shape: tuple
  out_buffer: str
  out_temp: str
  buffers: list  # inputs, then consts, then output (codegen + dispatch order)
  body: list  # list[ScalarOp]


@dataclass
class MatmulProgram:
  name: str
  grid: int  # M * N threads
  M: int
  N: int
  K: int
  a_buffer: str
  b_buffer: str
  out_buffer: str
  out_shape: tuple
  buffers: list  # [a, b, output]


def lower_to_loops(ir: KernelIR):
  """Lower array-level KernelIR into a loop-level program.

  Requires shape-specialized IR (trace with arrays); the grid size and matmul
  dims come from the traced shapes.
  """
  has_matmul = any(op.op_type == OpType.MATMUL for op in ir.ops)
  if has_matmul:
    return _lower_matmul(ir)
  return _lower_elementwise(ir)


def _require_shape(value, what: str) -> tuple:
  if value.shape is None:
    raise NotImplementedError(
      f"Metal backend needs shape-specialized IR ({what} has unknown shape); "
      "call the kernel with arrays or use lower(*arrays)."
    )
  return value.shape


def _temp(value_id: int) -> str:
  return f"t{value_id}"


def _lower_elementwise(ir: KernelIR) -> ElementwiseProgram:
  store_op = next(op for op in ir.ops if op.op_type == OpType.STORE)
  store_val = store_op.operands[0]
  out_shape = _require_shape(store_val, "output")
  numel = prod(out_shape) if out_shape else 1

  inputs: list[Buffer] = []
  consts: list[Buffer] = []
  seen_inputs: set[str] = set()
  body: list[ScalarOp] = []

  for op in ir.ops:
    if op.op_type == OpType.LOAD:
      shape = _require_shape(op.result, f"input '{op.ref_name}'")
      if (prod(shape) if shape else 1) != numel:
        raise NotImplementedError(
          f"Metal backend V1 requires matching shapes (no broadcasting): "
          f"input '{op.ref_name}' has shape {shape}, output {out_shape}."
        )
      if op.ref_name not in seen_inputs:
        seen_inputs.add(op.ref_name)
        inputs.append(Buffer(op.ref_name, role="input"))
      body.append(ScalarOp(_temp(op.result.id), OpType.LOAD, buffer=op.ref_name))

    elif op.op_type == OpType.CONST:
      arr = np.asarray(op.const_value)
      if arr.ndim == 0:
        body.append(
          ScalarOp(_temp(op.result.id), OpType.CONST, literal=float(arr))
        )
      else:
        if arr.size != numel:
          raise NotImplementedError(
            f"Metal backend V1 requires matching shapes (no broadcasting): "
            f"array constant has size {arr.size}, output numel {numel}."
          )
        cname = f"c{op.result.id}"
        consts.append(Buffer(cname, role="const", const_value=arr))
        body.append(ScalarOp(_temp(op.result.id), OpType.LOAD, buffer=cname))

    elif op.op_type == OpType.STORE:
      continue

    else:  # ADD, SUB, MUL, TRUEDIV, NEG
      args = tuple(_temp(v.id) for v in op.operands)
      body.append(ScalarOp(_temp(op.result.id), op.op_type, args=args))

  output = Buffer(store_op.ref_name, role="output")
  return ElementwiseProgram(
    name=ir.name,
    grid=numel,
    out_shape=out_shape,
    out_buffer=store_op.ref_name,
    out_temp=_temp(store_val.id),
    buffers=inputs + consts + [output],
    body=body,
  )


def _lower_matmul(ir: KernelIR) -> MatmulProgram:
  matmuls = [op for op in ir.ops if op.op_type == OpType.MATMUL]
  extra = [
    op for op in ir.ops
    if op.op_type not in (OpType.LOAD, OpType.STORE, OpType.MATMUL)
  ]
  if len(matmuls) != 1 or extra:
    raise NotImplementedError(
      "Metal backend V1 supports a single pure matmul kernel "
      "(no elementwise ops fused with matmul)."
    )

  mm = matmuls[0]
  a_val, b_val = mm.operands
  a_shape = _require_shape(a_val, "matmul lhs")
  b_shape = _require_shape(b_val, "matmul rhs")
  if len(a_shape) != 2 or len(b_shape) != 2:
    raise NotImplementedError(
      f"Metal backend V1 supports 2D matmul only, got {a_shape} @ {b_shape}."
    )

  load_ref = {op.result.id: op.ref_name for op in ir.ops if op.op_type == OpType.LOAD}
  if a_val.id not in load_ref or b_val.id not in load_ref:
    raise NotImplementedError(
      "Metal backend V1 matmul operands must come directly from kernel inputs."
    )
  a_ref, b_ref = load_ref[a_val.id], load_ref[b_val.id]
  store_op = next(op for op in ir.ops if op.op_type == OpType.STORE)

  M, K = a_shape
  _, N = b_shape
  return MatmulProgram(
    name=ir.name,
    grid=M * N,
    M=M,
    N=N,
    K=K,
    a_buffer=a_ref,
    b_buffer=b_ref,
    out_buffer=store_op.ref_name,
    out_shape=(M, N),
    buffers=[
      Buffer(a_ref, role="input"),
      Buffer(b_ref, role="input"),
      Buffer(store_op.ref_name, role="output"),
    ],
  )
