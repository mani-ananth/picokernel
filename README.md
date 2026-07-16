# picokernel

A minimal kernel language that compiles Python functions to NumPy or MLX — built to understand how array compiler pipelines work from tracing through code generation.

Inspired by [JAX Pallas](https://jax.readthedocs.io/en/latest/pallas/index.html). Not for production use.

---

## What it does

You write a kernel function using array references. picokernel traces it into an SSA IR, lowers it to executable code, and runs it — with guard-based retracing when shapes change, just like JAX or `torch.compile`.

```python
import numpy as np
import picokernel

@picokernel.kernel
def add(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]

x = np.array([1.0, 2.0, 3.0, 4.0])
y = np.array([5.0, 6.0, 7.0, 8.0])
out = np.zeros(4)

add(x, y, out)
# out → [6. 8. 10. 12.]
```

Switch backends with a single decorator argument:

```python
@picokernel.kernel(backend="mlx")   # Apple GPU via Metal
def add_gpu(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]
```

---

## Installation

```bash
git clone https://github.com/mani-ananth/picokernel
cd picokernel
pip install -e .

# Optional: MLX backend (Apple Silicon only)
pip install mlx
```

Requires Python ≥ 3.10.

---

## The pipeline

```
@kernel fn
    ↓  trace_kernel()        Python function → SSA IR (TracerRef proxies)
KernelIR
    ↓  lower_to_numpy()      IR → NumPy source (out= ufuncs, zero allocs)
    ↓  lower_to_mlx()        IR → MLX source  (h2d → lazy ops → sync → d2h)
    ↓  exec() + cache
callable
```

You can inspect every stage:

```python
@picokernel.kernel
def fma(a, b, c, o):
    o[...] = a[...] * b[...] + c[...]

# SSA IR
print(fma.show_ir())
# kernel fma(a, b, c, o):
#   v0<4:float32> = LOAD [a]()
#   v1<4:float32> = LOAD [b]()
#   v2<4:float32> = MUL(v0, v1)
#   v3<4:float32> = LOAD [c]()
#   v4<4:float32> = ADD(v2, v3)
#   STORE [o](v4)

# Generated NumPy code
a, b, c, o = (np.ones(4, dtype=np.float32) for _ in range(4))
print(fma.lower(a, b, c, o))
# def fma(a, b, c, o):
#   _buf = np.empty_like(o)
#   v2 = np.multiply(a, b, out=_buf)
#   np.add(v2, c, out=o)
```

---

## Backends

| Backend | How to use | Lowering strategy |
|---------|-----------|-------------------|
| `"numpy"` (default) | `@kernel` | `np.ufunc out=` — vectorized C, zero intermediate allocations |
| `"mlx"` | `@kernel(backend="mlx")` | array-level MLX on Metal GPU |

**NumPy lowering** eliminates intermediate arrays using `out=` parameters. For `a*b+c`:

```python
def fma(a, b, c, o):
    _buf = np.empty_like(o)          # one scratch buffer
    v2 = np.multiply(a, b, out=_buf) # no temp allocation
    np.add(v2, c, out=o)             # writes directly to output
```

**MLX lowering** wraps inputs in `mx.array()`, builds a lazy compute graph, evaluates it with `mx.eval()`, then writes back:

```python
def fma(a, b, c, o):
    v0 = mx.array(a)     # h2d
    v1 = mx.array(b)     # h2d
    v2 = v0 * v1         # lazy (no GPU work yet)
    v3 = mx.array(c)     # h2d
    v4 = v2 + v3         # lazy
    mx.eval(v4)          # GPU executes here (fused single pass)
    o[...] = np.array(v4) # d2h
```

---

## Performance characteristics (M4, float32)

For `a * b + c` across sizes:

```
size=     1,000  numpy=  0.002ms  mlx=  0.24ms   NumPy  149x faster
size=    10,000  numpy=  0.003ms  mlx=  0.21ms   NumPy   61x faster
size=   100,000  numpy=  0.020ms  mlx=  0.28ms   NumPy   15x faster
size= 1,000,000  numpy=  0.29ms   mlx=  1.39ms   NumPy    5x faster
size=10,000,000  numpy=  4.01ms   mlx=  8.47ms   NumPy    2x faster
```

NumPy is faster at all sizes in the current design because each MLX kernel call pays h2d + d2h transfer costs (h2d ~11ms per 381MB array at 100M elements; the d2h writeback is ~44ms because `o[...] = np.array(v)` makes two passes over the result). The GPU compute itself is faster (MLX fuses multiply+add into one pass: ~25ms vs NumPy's two-pass ~48ms at 100M), but transfers dominate.

Keeping data device-resident amortizes the transfers (`benchmarks/chained_ops.py`), but element-wise kernels stay DRAM-bandwidth-bound on unified memory, so MLX only approaches parity — pulling ahead takes work with higher arithmetic intensity (e.g. matmul).

---

## Profiling

picokernel has a built-in profiler that emits [Perfetto](https://ui.perfetto.dev) / Chrome Trace Event JSON.

```python
# Profile a single backend
p = fma.run_profiled(a, b, c, o, n_repeats=5)
p.save("trace.json")   # open in ui.perfetto.dev

# Compare both backends
from picokernel.profiler import Profiler

p_numpy = fma.run_profiled(a, b, c, o, n_repeats=5)
p_mlx   = fma_mlx.run_profiled(a, b, c, o, n_repeats=5)

Profiler.merge(p_numpy, p_mlx).save("comparison.json")
```

MLX trace categories visible in Perfetto:

| Category | What it is |
|----------|-----------|
| `h2d` | `mx.array()` host→device transfers |
| `op_lazy` | lazy graph construction (~0μs, no GPU work) |
| `sync` | `mx.eval()` — GPU executes here |
| `d2h` | `np.array()` device→host writeback |

```bash
# CLI comparison of two trace files
python tools/compare_traces.py numpy_trace.json mlx_trace.json

# Merge for side-by-side view in Perfetto (PID 1 = numpy, PID 2 = mlx)
python tools/compare_traces.py numpy_trace.json mlx_trace.json --merge comparison.json
```

---

## Examples

| File | What it shows |
|------|--------------|
| `examples/01_vector_add.py` | Basic kernel, IR dump, lowered source |
| `examples/02_matrix_multiply.py` | Matmul kernel |
| `examples/03_numpy_vs_mlx.py` | Benchmark sweep 1K–100M elements |
| `examples/04_perf_breakdown.py` | Per-step timing: where does the time go? |
| `examples/05_perfetto_profile.py` | Generate Perfetto trace files |

---

## Development

```bash
pip install -e ".[test]"
pytest                        # 107 tests
pytest tests/test_integration.py::test_vector_add   # single test
```

### Module map

| Module | Responsibility |
|--------|---------------|
| `core.py` | IR: `KernelIR`, `IROp`, `IRValue`, `OpType` |
| `trace.py` | Tracing via `TracerRef` proxies → SSA IR |
| `lowering.py` | NumPy code generation (`out=` ufuncs) |
| `mlx_lowering.py` | MLX code generation |
| `runtime.py` | `exec()` + `id(ir)`-keyed caching |
| `profiler.py` | Chrome Trace Event JSON emitter |
| `profiled_lowering.py` | Timing-instrumented lowering variants |
| `__init__.py` | Public API: `@kernel`, `KernelFunction` |

### Key design decisions

- **Kernel contract:** functions take only `ref` parameters; last param is conventionally the output. `ref[...]` (ellipsis only) is the sole indexing form.
- **Guard-based retracing:** each `(shapes, dtypes)` tuple gets its own `(KernelIR, compiled_fn)` cache entry — same model as JAX/Dynamo.
- **`out=` lowering:** avoids intermediate allocations by aliasing LOADs directly to ref names and routing single-use values through one pre-allocated scratch buffer.
- **Lazy MLX:** arithmetic ops build a graph; `mx.eval()` dispatches to Metal and fuses the full chain into a single GPU kernel.
- **Profiling overhead:** ~60μs flat per `run_profiled()` call (dominated by `exec()` recompile). Use `__call__` for production timing, `run_profiled` for trace generation.

---

## What's next

- C codegen backend (LLVM IR or direct `.c` emission)
- Metal compute shader codegen
- Matmul tiling and loop-level fusion
- MLX-native kernel API (inputs/outputs stay on-device, eliminating h2d/d2h)
