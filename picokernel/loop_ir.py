"""Mid-level loop IR: the bridge between array ops and target codegen.

This is where MLIR-style progressive lowering starts. KernelIR is array-level
(`ADD`, `MUL`, `MATMUL` over whole arrays, 1:1 with NumPy/MLX calls). Targets
like C or Metal need explicit per-element loops instead, so we lower into a small
loop dialect first:

    KernelIR (array ops)
        -> lower_to_loops()
    LoweredProgram: an ordered list of segments (each a grid + body)
        -> lower_to_metal() / (future) lower_to_c()
    target source

A kernel is split into an ordered sequence of **segments**, each becoming one
GPU kernel dispatch:

  * ElementwiseProgram - a flat grid of `numel` threads, each running a
    straight-line scalar body; may produce several outputs.
  * MatmulProgram - a grid of M*N threads, each accumulating a K-length reduction.

Because the two have different iteration spaces, a MATMUL can't share a grid with
its elementwise neighbours. So a graph like `(a @ b) + bias` lowers to two
segments dispatched in order, with an **intermediate buffer** (device-only, never
copied to/from the host) carrying the matmul result into the elementwise stage.
This is multi-kernel scheduling, not fusion: each op-cluster is its own dispatch.

Still unsupported (raises NotImplementedError): broadcasting between mismatched
shapes, >2D/batched matmul. float32 only. Fusing an elementwise epilogue into the
matmul kernel is a deliberate future step, not done here.
"""

from dataclasses import dataclass
from math import prod
from typing import Any, Optional

import numpy as np

from .core import KernelIR, OpType


@dataclass
class Buffer:
  """A device buffer, identified by name across the whole kernel.

  role is its *global* nature, which decides how the runtime fills it:
    "input"        - a kernel ref param, copied host->device each call
    "const"        - a baked-in array constant (carried in const_value)
    "intermediate" - device-only, written by one segment and read by a later one
    "output"       - the STORE target, copied device->host after the last segment
  Per-segment read/write mutability is tracked separately on each Program.
  """

  name: str
  role: str
  const_value: Any = None


@dataclass
class ScalarOp:
  """One operation in an elementwise body, in SSA form over scalar temps.

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
  buffers: list  # Buffer objects in codegen/dispatch order (reads, then writes)
  body: list  # list[ScalarOp]
  outputs: list  # list[(out_temp, out_buffer_name)] stored at the end of the body
  writes: set  # buffer names this segment writes (mutable params in MSL)


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
  buffers: list  # [a, b, output]
  writes: set  # {out_buffer}


@dataclass
class LoweredProgram:
  """A whole kernel lowered to an ordered list of dispatchable segments."""

  name: str
  segments: list  # list[ElementwiseProgram | MatmulProgram], dispatch order
  intermediates: list  # list[(name, numel)] - device-only buffers to allocate
  out_buffer: str
  out_shape: tuple


_EW_OPS = (OpType.ADD, OpType.SUB, OpType.MUL, OpType.TRUEDIV, OpType.NEG)


def lower_to_loops(ir: KernelIR) -> LoweredProgram:
  """Lower array-level KernelIR into an ordered list of loop-level segments.

  Requires shape-specialized IR (trace with arrays); grid sizes and matmul dims
  come from the traced shapes.
  """
  store_op = next(op for op in ir.ops if op.op_type == OpType.STORE)
  store_ref = store_op.ref_name
  store_val = store_op.operands[0]
  out_shape = _require_shape(store_val, "output")

  # Leaf materializations: LOAD results live in input buffers, array CONSTs in
  # const buffers, scalar CONSTs are literals.
  load_ref: dict[int, str] = {}
  scalar_const: dict[int, float] = {}
  array_const: dict[int, np.ndarray] = {}
  for op in ir.ops:
    if op.op_type == OpType.LOAD:
      load_ref[op.result.id] = op.ref_name
    elif op.op_type == OpType.CONST:
      arr = np.asarray(op.const_value)
      if arr.ndim == 0:
        scalar_const[op.result.id] = float(arr)
      else:
        array_const[op.result.id] = arr

  # Partition compute ops into ordered segments; a MATMUL is always its own.
  seg_ops = _segment(ir)
  seg_of: dict[int, int] = {}  # value id -> producing segment index
  for i, (_, ops) in enumerate(seg_ops):
    for op in ops:
      seg_of[op.result.id] = i

  # A produced value needs a device buffer iff it leaves its segment: it is the
  # STORE value, or a consumer lives in a different segment (matmuls are always a
  # different segment, so matmul operands are covered here too).
  materialized: set[int] = set()
  for i, (_, ops) in enumerate(seg_ops):
    for op in ops:
      for v in op.operands:
        if v.id in seg_of and seg_of[v.id] != i:
          materialized.add(v.id)
  if store_val.id in seg_of:
    materialized.add(store_val.id)

  def result_buffer(vid: int) -> tuple:
    """(name, role) for a materialized compute-op result."""
    if vid == store_val.id:
      return store_ref, "output"
    return f"m{vid}", "intermediate"

  def read_ref(v) -> tuple:
    """How to read a value inside an elementwise body.

    Returns ("literal", value) or ("buffer", name, role, const_value).
    """
    if v.id in scalar_const:
      return ("literal", scalar_const[v.id])
    if v.id in array_const:
      return ("buffer", f"c{v.id}", "const", array_const[v.id])
    if v.id in load_ref:
      return ("buffer", load_ref[v.id], "input", None)
    name, role = result_buffer(v.id)  # produced by another segment
    return ("buffer", name, role, None)

  segments = []
  intermediates: dict[str, int] = {}

  def note_intermediate(name: str, role: str, numel: int):
    if role == "intermediate":
      intermediates[name] = numel

  for i, (kind, ops) in enumerate(seg_ops):
    name = ir.name if len(seg_ops) <= 1 else f"{ir.name}_{i}"
    if kind == "matmul":
      prog = _build_matmul(ops[0], name, read_ref, result_buffer, note_intermediate)
    else:
      prog = _build_elementwise(
        ops, name, materialized, seg_of, i, read_ref, result_buffer,
        store_val.id, note_intermediate,
      )
    segments.append(prog)

  # Identity/passthrough kernels (o = x, o = const_array) have no compute op
  # producing the store value; synthesize a copy segment.
  if store_val.id not in seg_of:
    segments.append(
      _build_passthrough(ir.name, store_val, store_ref, read_ref)
    )

  return LoweredProgram(
    name=ir.name,
    segments=segments,
    intermediates=list(intermediates.items()),
    out_buffer=store_ref,
    out_shape=out_shape,
  )


def _segment(ir: KernelIR) -> list:
  """Split compute ops into ordered [(kind, [ops])]; each MATMUL is isolated."""
  segments: list = []
  cur = None
  for op in ir.ops:
    if op.op_type in (OpType.LOAD, OpType.STORE, OpType.CONST):
      continue
    if op.op_type == OpType.MATMUL:
      segments.append(("matmul", [op]))
      cur = None
    elif op.op_type in _EW_OPS:
      if cur is None:
        cur = ("elementwise", [])
        segments.append(cur)
      cur[1].append(op)
    else:
      raise NotImplementedError(f"Metal backend cannot lower op {op.op_type}")
  return segments


def _require_shape(value, what: str) -> tuple:
  if value.shape is None:
    raise NotImplementedError(
      f"Metal backend needs shape-specialized IR ({what} has unknown shape); "
      "call the kernel with arrays or use lower(*arrays)."
    )
  return value.shape


def _numel(shape) -> int:
  return prod(shape) if shape else 1


def _temp(value_id: int) -> str:
  return f"t{value_id}"


def _build_elementwise(
  ops, name, materialized, seg_of, seg_idx, read_ref, result_buffer,
  store_val_id, note_intermediate,
) -> ElementwiseProgram:
  numel = _numel(ops[0].result.shape)

  produced = {op.result.id for op in ops}
  body: list[ScalarOp] = []
  reads: list[Buffer] = []
  writes: list[Buffer] = []
  emitted_reads: set[int] = set()
  seen_read_names: set[str] = set()
  write_names: set[str] = set()

  def ensure_read(v):
    """Emit a LOAD/CONST scalar op for an operand produced outside this segment."""
    if v.id in produced or v.id in emitted_reads:
      return
    emitted_reads.add(v.id)
    kind, *rest = read_ref(v)
    if kind == "literal":
      body.append(ScalarOp(_temp(v.id), OpType.CONST, literal=rest[0]))
      return
    _, bname, role, const_value = (kind, *rest)
    if _numel(v.shape) != numel:
      raise NotImplementedError(
        f"Metal backend requires matching shapes (no broadcasting): "
        f"operand shape {v.shape}, segment numel {numel}."
      )
    if bname not in seen_read_names:
      seen_read_names.add(bname)
      reads.append(Buffer(bname, role=role, const_value=const_value))
    body.append(ScalarOp(_temp(v.id), OpType.LOAD, buffer=bname))

  for op in ops:
    if _numel(op.result.shape) != numel:
      raise NotImplementedError(
        f"Metal backend requires matching shapes (no broadcasting): "
        f"op result shape {op.result.shape}, segment numel {numel}."
      )
    for v in op.operands:
      ensure_read(v)
    body.append(ScalarOp(_temp(op.result.id), op.op_type,
                         args=tuple(_temp(v.id) for v in op.operands)))

  outputs = []
  for op in ops:
    if op.result.id not in materialized:
      continue
    bname, role = result_buffer(op.result.id)
    if bname not in write_names:
      write_names.add(bname)
      writes.append(Buffer(bname, role=role))
      note_intermediate(bname, role, numel)
    outputs.append((_temp(op.result.id), bname))

  return ElementwiseProgram(
    name=name,
    grid=numel,
    buffers=reads + writes,
    body=body,
    outputs=outputs,
    writes=write_names,
  )


def _build_matmul(
  op, name, read_ref, result_buffer, note_intermediate,
) -> MatmulProgram:
  a_val, b_val = op.operands
  a_shape = _require_shape(a_val, "matmul lhs")
  b_shape = _require_shape(b_val, "matmul rhs")
  if len(a_shape) != 2 or len(b_shape) != 2:
    raise NotImplementedError(
      f"Metal backend supports 2D matmul only, got {a_shape} @ {b_shape}."
    )

  a_read, b_read = read_ref(a_val), read_ref(b_val)
  if a_read[0] != "buffer" or b_read[0] != "buffer":
    raise NotImplementedError("Metal backend matmul operands must be array-valued.")
  a_buf = Buffer(a_read[1], role=a_read[2], const_value=a_read[3])
  b_buf = Buffer(b_read[1], role=b_read[2], const_value=b_read[3])

  out_name, out_role = result_buffer(op.result.id)
  note_intermediate(out_name, out_role, _numel(op.result.shape))
  out_buf = Buffer(out_name, role=out_role)

  M, K = a_shape
  _, N = b_shape
  return MatmulProgram(
    name=name,
    grid=M * N,
    M=M, N=N, K=K,
    a_buffer=a_buf.name,
    b_buffer=b_buf.name,
    out_buffer=out_name,
    buffers=[a_buf, b_buf, out_buf],
    writes={out_name},
  )


def _build_passthrough(name, store_val, store_ref, read_ref) -> ElementwiseProgram:
  """A trivial copy kernel for `o = x` / `o = const_array`."""
  numel = _numel(store_val.shape)
  src = read_ref(store_val)
  if src[0] != "buffer":
    # scalar broadcast into an array output isn't a same-shape copy
    raise NotImplementedError("Metal backend cannot broadcast a scalar into an array output.")
  _, bname, role, const_value = src
  src_buf = Buffer(bname, role=role, const_value=const_value)
  out_buf = Buffer(store_ref, role="output")
  body = [ScalarOp(_temp(store_val.id), OpType.LOAD, buffer=bname)]
  return ElementwiseProgram(
    name=name,
    grid=numel,
    buffers=[src_buf, out_buf],
    body=body,
    outputs=[(_temp(store_val.id), store_ref)],
    writes={store_ref},
  )
