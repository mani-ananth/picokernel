"""picokernel — a minimal Pallas-like kernel language."""

from .core import pretty_print
from .lowering import lower_to_numpy
from .mlx_lowering import lower_to_mlx
from .runtime import compile_metal, compile_mlx, compile_numpy
from .trace import trace_kernel


class KernelFunction:
  """Wraps a user-defined kernel function with trace/lower/compile/run.

  Guard-based retracing: each unique (shapes, dtypes) signature gets its own
  cached (KernelIR, compiled_fn) entry, mirroring JAX/Dynamo's approach.

  backend: "numpy" (default, vectorized C via out= ufuncs) or "mlx" (Apple GPU).
  """

  def __init__(self, fn, backend: str = "numpy"):
    self._fn = fn
    self._backend = backend
    self._guard_cache: dict[tuple, tuple] = {}

  def _guard_key(self, arrays) -> tuple:
    shapes = tuple(arr.shape for arr in arrays)
    dtypes = tuple(arr.dtype for arr in arrays)
    return (shapes, dtypes)

  def _compile(self, ir):
    if self._backend == "mlx":
      return compile_mlx(ir)
    if self._backend == "metal":
      return compile_metal(ir)
    return compile_numpy(ir)

  def _get_or_trace(self, arrays) -> tuple:
    key = self._guard_key(arrays)
    if key not in self._guard_cache:
      shapes = [arr.shape for arr in arrays]
      dtypes = [arr.dtype for arr in arrays]
      ir = trace_kernel(self._fn, shapes, dtypes)
      self._guard_cache[key] = (ir, self._compile(ir))
    return self._guard_cache[key]

  @property
  def ir(self):
    if self._guard_cache:
      return next(iter(self._guard_cache.values()))[0]
    return trace_kernel(self._fn)

  def __call__(self, *arrays):
    _, compiled = self._get_or_trace(arrays)
    compiled(*arrays)

  def lower(self, *arrays) -> str:
    """Return lowered source. Pass arrays for shape-aware tracing.

    The metal backend needs shape-specialized IR, so it requires arrays.
    """
    if arrays:
      ir, _ = self._get_or_trace(arrays)
    else:
      ir = trace_kernel(self._fn)
    if self._backend == "mlx":
      return lower_to_mlx(ir)
    if self._backend == "metal":
      from .loop_ir import lower_to_loops
      from .metal_lowering import lower_to_metal
      return lower_to_metal(lower_to_loops(ir))
    return lower_to_numpy(ir)

  def run_profiled(self, *arrays, profiler=None, n_repeats: int = 1):
    """Run with per-op timing instrumentation. Returns a Profiler with trace events.

    Each repeat emits its own kernel_call span so Perfetto shows all runs and
    compare_traces.py can compute mean/stdev across them.

    Args:
      n_repeats: number of timed calls (default 1). Use >=5 for stable timings.

    Example:
      p = kernel.run_profiled(a, b, out, n_repeats=10)
      p.save("trace.json")  # open in ui.perfetto.dev
    """
    from .profiler import Profiler
    from .profiled_lowering import compile_mlx_profiled, compile_numpy_profiled

    if self._backend == "metal":
      raise NotImplementedError(
        "run_profiled is not supported for the metal backend yet; "
        "use __call__ for timing."
      )

    pid = 2 if self._backend == "mlx" else 1
    if profiler is None:
      profiler = Profiler(name=f"{self._fn.__name__} ({self._backend})", pid=pid)

    key = self._guard_key(arrays)
    if key in self._guard_cache:
      ir = self._guard_cache[key][0]
    else:
      shapes = [arr.shape for arr in arrays]
      dtypes = [arr.dtype for arr in arrays]
      ir = trace_kernel(self._fn, shapes, dtypes)

    if self._backend == "mlx":
      fn = compile_mlx_profiled(ir, profiler)
    else:
      fn = compile_numpy_profiled(ir, profiler)

    profiler.record_counters()
    if self._backend == "mlx":
      profiler.record_mlx_memory()

    for _ in range(n_repeats):
      with profiler.span("kernel_call", cat="runtime", size=arrays[0].size):
        fn(*arrays)

    profiler.record_counters()
    if self._backend == "mlx":
      profiler.record_mlx_memory()

    return profiler

  def show_ir(self, *arrays) -> str:
    if arrays:
      ir, _ = self._get_or_trace(arrays)
    else:
      ir = trace_kernel(self._fn)
    return pretty_print(ir)


def kernel(fn=None, *, backend: str = "numpy"):
  """Decorator: marks a function as a picokernel kernel.

  Usage:
    @kernel
    def fn(...): ...              # NumPy backend (default)

    @kernel(backend="mlx")
    def fn(...): ...              # MLX backend (Apple GPU via Metal)
  """
  if fn is not None:
    return KernelFunction(fn, backend=backend)
  def decorator(f):
    return KernelFunction(f, backend=backend)
  return decorator
