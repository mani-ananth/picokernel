# AGENTS.md

This file provides guidance to Codex (Codex.ai/code) when working with code in this repository.

## Commands

```bash
# Install in editable mode (required before running anything)
pip install -e .

# Run all tests
pytest

# Run a single test
pytest tests/test_integration.py::test_vector_add

# Examples
python examples/01_vector_add.py
python examples/03_numpy_vs_mlx.py
python examples/04_perf_breakdown.py        # step-by-step timing breakdown
python examples/05_perfetto_profile.py --size 10000 --repeats 5

# Profiling / trace comparison
python tools/compare_traces.py numpy_trace.json mlx_trace.json
python tools/compare_traces.py numpy_trace.json mlx_trace.json --merge comparison.json
```

## Architecture

`picokernel` is a minimal Pallas-like kernel language that compiles Python functions into executable NumPy or MLX code. The pipeline is:

```
@kernel fn  →  trace_kernel  →  KernelIR  →  lower_to_numpy / lower_to_mlx  →  exec()  →  callable
```

**`core.py`** — IR definitions. `KernelIR` holds a list of `IROp`s in SSA form. Each `IRValue` has a unique integer ID, name, shape, and dtype. `OpType` enumerates all operations (LOAD, STORE, CONST, ADD, SUB, MUL, TRUEDIV, NEG, MATMUL).

**`trace.py`** — Tracing. `trace_kernel(fn)` calls the user's function with `TracerRef` proxies (one per parameter). Indexing a `TracerRef` with `[...]` emits LOAD/STORE ops; arithmetic on `TracerValue`s emits the corresponding binary ops. Shape/dtype propagation happens here.

**`lowering.py`** — Code generation. `lower_to_numpy(ir)` generates NumPy source using `np.ufunc out=` to eliminate intermediate allocations. LOADs are aliased directly to their ref (no `.copy()`); single-use intermediates route through a pre-allocated `_buf`; the final op writes directly into the output ref via `out=store_ref`.

**`mlx_lowering.py`** — MLX code generation. `lower_to_mlx(ir)` generates array-level MLX source. LOADs become `mx.array(ref)` (host→device); STOREs become `mx.eval(result)` + `ref[...] = np.array(result)` (sync + device→host).

**`runtime.py`** — Execution. `compile_numpy(ir)` and `compile_mlx(ir)` exec the lowered source and cache the callable keyed by `id(ir)`.

**`profiler.py`** — `Profiler` class emitting Chrome Trace Event JSON (loadable in ui.perfetto.dev). Records `complete` (X) events, `counter` (C) events, and `span` (B/E) context managers. `Profiler.merge(*profilers)` combines traces with separate PIDs for side-by-side Perfetto view.

**`profiled_lowering.py`** — Instrumented lowering variants. `compile_numpy_profiled(ir, profiler)` and `compile_mlx_profiled(ir, profiler)` generate code with `_p.complete()` timing around each op. The profiler is injected via the `exec()` namespace as `_p`. MLX ops are tagged with semantic categories: `h2d`, `op_lazy`, `sync`, `d2h`.

**`__init__.py`** — Public API. The `@kernel` decorator wraps a function in `KernelFunction`. `KernelFunction.__call__(*arrays)` compiles and runs. `KernelFunction.run_profiled(*arrays)` compiles a fresh profiled version, runs it, and returns a `Profiler` with trace events.

**`tools/compare_traces.py`** — CLI tool. Compares two Perfetto JSON files (mean duration per event, side-by-side table) and merges them into a single file for Perfetto side-by-side view.

## Backends

Two backends, selected via `@kernel(backend=...)`:

| Backend | Default | Lowering | Notes |
|---------|---------|----------|-------|
| `"numpy"` | yes | `lower_to_numpy` | `np.ufunc out=`, vectorized C, zero intermediate allocs |
| `"mlx"` | no | `lower_to_mlx` | array-level MLX on Metal GPU |

## Key design conventions

- Kernel functions take only `ref` parameters (no return value); the last parameter is conventionally the output ref.
- `ref[...]` (Ellipsis indexing only) is the only supported indexing — no slicing or integer indices.
- Kernels are traced once per unique set of (shapes, dtypes); changes trigger retrace (guard-based caching).
- `lower()` without arrays traces without shape info; `lower(*arrays)` uses the shape-aware cached IR.
- `run_profiled()` always recompiles a fresh profiled function — not cached, intentionally.

## Performance notes (M4, float32)

- NumPy (`out=` ufuncs) wins for small arrays — GPU launch overhead dominates MLX below ~1M elements.
- MLX transfer overhead: `mx.array()` (h2d) and `np.array()` (d2h) cost ~9ms each at 100M elements.
- `mx.eval()` is where the GPU actually executes — prior MLX ops are lazy graph construction (~0μs).
- Profiler overhead: ~60μs flat per `run_profiled()` call (dominated by `exec()` recompile, not the timing primitives).
- Use `mx.metal.start_capture()` / Xcode Metal debugger to break down `mx.eval` at the shader level.
- Use `mx.disable_compile()` to disable MLX kernel fusion and see individual op dispatches.
