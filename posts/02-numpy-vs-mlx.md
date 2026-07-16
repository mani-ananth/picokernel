---
title: "Adding an Accelerator Backend to picokernel & time analysis"
subtitle: "A profiler-driven comparison of the NumPy and MLX backends — unified memory, data transfers, and lazy evaluation on Apple Silicon."
date: 2026-06-20
tags: [performance, mlx, apple-silicon, gpu, profiling]
---

[The previous post](./01-compiler-pipeline.md) built [`picokernel`](https://github.com/mani-ananth/picokernel), a small kernel compiler with a NumPy backend. NumPy was a good starting point: it runs on every CPU architecture Python supports — x86, ARM, POWER — and it picks up vendor-optimized SIMD and BLAS wherever it lands (Apple's Accelerate framework on my M4, MKL or OpenBLAS elsewhere). This allows for one backend with broad CPU coverage.

But NumPy is strictly a CPU library. It doesn't target GPUs or other accelerators, and accelerators are where most of the interesting kernel-compiler problems live: data transfers, lazy execution, kernel fusion, synchronization. To open up that space, this post adds a second backend built on [MLX](https://github.com/ml-explore/mlx), Apple's NumPy-shaped array library that targets the M4's GPU through Metal.

Two backends behind the same `@kernel` API also make for a clean measurement setup: same kernel, same input arrays, one variable — the hardware it runs on. That lets us ask where each backend is fast, and why.

Here's the first measurement: elementwise `a * b + c` on float32 arrays, timed end-to-end (NumPy arrays in, NumPy arrays out).

```
size=     1,000  numpy=  0.002ms   mlx=  0.24ms    NumPy 149x faster
size=    10,000  numpy=  0.003ms   mlx=  0.21ms    NumPy  61x faster
size=   100,000  numpy=  0.020ms   mlx=  0.28ms    NumPy  15x faster
size= 1,000,000  numpy=  0.29ms    mlx=  1.39ms    NumPy   5x faster
size=10,000,000  numpy=  4.01ms    mlx=  8.47ms    NumPy   2x faster
```

![Log-log chart of time per call vs array size for the NumPy and MLX backends running a*b+c on float32, median of 50 calls. The NumPy line is below the MLX line at every measured size from 1K to 10M elements; annotations mark the gap at 149x on the left and 2.1x on the right. X-axis: array size (log scale). Y-axis: time per call in ms (log scale).](./images/02-benchmark-curve.png)

For this kernel and this calling convention, the NumPy backend is faster at every size tested (medians of 50 calls after warmup, M4 MacBook). The gap narrows from about 150x to about 2x as arrays grow, but it doesn't close in the measured range.

The ratio itself isn't the interesting part. The interesting part is where MLX's time goes — and what that says about when an accelerator backend is the right choice.

To answer that, I built a profiler.

---

## The NumPy backend: vectorized C with no intermediates

Before looking at MLX, it's worth a quick look at why the NumPy backend is fast in the first place.

When you write `a * b + c` in plain NumPy, you get two array allocations: one for `a * b`, then a second for the addition. That's wasted memory traffic, and on large arrays it matters.

`picokernel` traces the kernel into an intermediate representation, then lowers it to NumPy code that uses the `out=` parameter on ufuncs to eliminate intermediate allocations entirely. The generated code looks like this:

```python
def kernel(a, b, c, o):
    _buf = np.empty_like(o)              # one scratch buffer
    v2 = np.multiply(a, b, out=_buf)     # no temporary array
    np.add(v2, c, out=o)                 # writes directly to output
```

Two operations with one pre-allocated scratch buffer and zero intermediate allocations. And — critically — every line stays inside NumPy's vectorized C implementation. The Python interpreter is invoked only to call the two ufuncs; the actual element-wise work runs at C speed, with SIMD vectorization courtesy of Apple's Accelerate framework.

This is the CPU baseline for the comparison.

(How does this lowering work? That's the topic of [the previous post](./01-compiler-pipeline.md).)

---

## MLX: lazy by default, eager when you ask

MLX is structured around a key insight: most array operations don't need to run immediately. When you write `v = a * b` in MLX, no GPU work happens. Instead, MLX builds a graph node representing "the result of multiplying `a` and `b`," and returns a handle. The actual computation is deferred until something asks for the result — typically a call to `mx.eval()`.

For our kernel, `picokernel` lowers to:

```python
def kernel(a, b, c, o):
    v0 = mx.array(a)        # h2d: copy host array into MLX-managed buffer
    v1 = mx.array(b)        # h2d
    v2 = v0 * v1            # lazy — no GPU work yet
    v3 = mx.array(c)        # h2d
    v4 = v2 + v3            # lazy
    mx.eval(v4)             # GPU executes here — fuses multiply + add into one Metal kernel
    o[...] = np.array(v4)   # d2h: copy result back to NumPy land
```

The lowered code breaks into four steps:

1. **h2d transfers** — `mx.array()` copies a NumPy array into a Metal-managed buffer.
2. **Lazy graph construction** — the arithmetic ops are nearly free, just building a DAG.
3. **`mx.eval()`** — this is where the GPU actually runs. MLX fuses the entire chain into a single Metal kernel, dispatches it, and blocks until it's done.
4. **d2h transfer** — `np.array()` on an MLX array copies the result back to NumPy.

Steps 1 and 4 might look unnecessary on a Mac: the M4's CPU and GPU share the same physical DRAM, so why copy anything? The short answer is that MLX can only compute on buffers its Metal allocator owns, so crossing the NumPy↔MLX boundary always means a real copy — the ["Unified memory doesn't eliminate copies"](#unified-memory-doesnt-eliminate-copies) section below walks through exactly what `mx.array()` has to do and what it costs.

This design is *elegant*: lazy evaluation lets MLX see your full computation before deciding how to schedule it, so it can fuse aggressively. The single-kernel fusion is a real benefit — separate `multiply` and `add` kernels would each have to move `a * b` through GPU memory, but the fused version computes it once and feeds it straight into the add.

Steps 1 and 4, though, turn out to dominate the timing.

---

## The profiler we needed

To understand where the time was going, I built a profiler that emits Chrome Trace Event JSON — the format [Perfetto](https://ui.perfetto.dev) consumes. Every operation in the lowered code gets wrapped with a timing call, and the resulting events show up as a flame graph in Perfetto.

The profiler categorizes MLX events into four buckets:

| Category | What it measures |
|----------|------------------|
| `h2d` | `mx.array()` — copying host data into MLX-managed buffers |
| `op_lazy` | arithmetic that just builds the graph (~0μs) |
| `sync` | `mx.eval()` — the GPU actually running |
| `d2h` | `np.array(mlx_array)` — copying results back to NumPy |

Here's what a single 100M-element call looks like, broken down (mean of 10 calls):

```
kernel_call (104ms total)
├─ h2d: mx.array(a)       11.6 ms
├─ h2d: mx.array(b)       11.5 ms
├─ op_lazy: multiply       6.2 μs
├─ h2d: mx.array(c)       11.2 ms
├─ op_lazy: add            4.0 μs
├─ sync: mx.eval          25.3 ms
└─ d2h: np.array          43.9 ms
```

![Timeline of one MLX kernel call at 100M elements, rendered from the profiler trace with mean durations over 10 calls. Three h2d transfer segments (~11ms each, orange) come first, then the mx.eval GPU-compute block (25.3ms, blue), then the d2h writeback (43.9ms, red). Thin tick marks show the two lazy ops (~10μs total, invisible at this scale). Total: 104ms.](./images/02-perfetto-trace.png)

![Horizontal stacked bar chart comparing both backends at 100M elements. The MLX bar (104ms total) splits into h2d 34.3ms (33%, orange), mx.eval GPU compute 25.3ms (24%, blue), and d2h 43.9ms (42%, red). Below it, the NumPy bar is a single blue compute segment of 47.9ms. The title notes that MLX's data movement alone (78ms) exceeds NumPy's entire run.](./images/02-time-breakdown.png)

Add it up: **34.3 ms** in `h2d`, **43.9 ms** in `d2h`, **25.3 ms** in actual GPU compute. The lazy graph construction is genuinely free. The compute itself is fast — NumPy needed 48 ms in two passes for the same arithmetic, so MLX's fused kernel saves about 23 ms of compute time.

But the data movement around it costs about 78 ms, so the end-to-end call comes out roughly 56 ms slower (104 ms vs 48 ms).

One detail worth calling out: the d2h step costs more than any single h2d (43.9 ms vs ~11 ms) not because that direction is slower, but because it does more work. The generated line `o[...] = np.array(v4)` makes two passes over the result: `np.array()` materializes the MLX array into a freshly allocated NumPy array (~21 ms measured in isolation), then `o[...] =` copies that into the output buffer (~16 ms), with first-touch page faults on the fresh allocation accounting for the rest.

The entire difference between the two backends is data movement.

---

## Unified memory doesn't eliminate copies

The M4 has unified memory: CPU and GPU share the same DRAM. There's no PCIe bus, no separate VRAM. So why does `mx.array(a)` cost 11 milliseconds when the data is already in the right physical chip?

The short answer is that *physical sharing isn't the same as buffer sharing*.

When you call `mx.array(numpy_array)`, MLX has to:

1. Allocate a Metal buffer through the GPU's memory allocator.
2. Copy the NumPy data into that buffer, because the NumPy array isn't owned by Metal and might be freed, resized, or modified at any time.
3. Tag the buffer for GPU access (handles, residency tracking, etc.).

Even when the source and destination live on the same physical DRAM, you're still doing a memcpy of ~381 MB (100M float32 elements). Apple's unified memory removes the PCIe bus, but it doesn't eliminate the cost of moving bytes from one logical buffer to another.

![Diagram of M4 unified memory architecture: a CPU box (labeled "NumPy runs here") and a GPU box (labeled "Metal shaders") at the top, both connected down into a single block labeled "Unified DRAM — one physical memory". Inside the DRAM block, two separate boxes labeled "NumPy buffer (malloc)" and "Metal buffer (MTLBuffer)", with a red arrow between them labeled "mx.array() copy ~11 ms / 381 MB". Caption: "Same physical memory, different logical buffers — the copy is still real."](./images/02-unified-memory.png)

The takeaway is that **on Apple Silicon, the GPU is cheap to reach, but handing it data still costs real time.**

---

## Amortizing the transfers: chaining ops on the device

If transfers dominate, the obvious move is to transfer less.

The benchmark above assumes you start with a NumPy array and want a NumPy array back, so every call pays the full transfer cost around a few tens of milliseconds of arithmetic. But that's a calling-convention choice, not an MLX limitation. The device-resident alternative is to move the inputs to the device once, chain N operations there without ever calling `np.array()`, and move the result back once at the end. This is exactly how PyTorch users keep tensors on the GPU between operations and how JAX users keep arrays on the accelerator across `jit`-compiled function boundaries.

[`benchmarks/chained_ops.py`](https://github.com/mani-ananth/picokernel/blob/master/benchmarks/chained_ops.py) measures that directly: N chained steps of `x = x * b + c` at 100M elements. NumPy runs the chain with in-place `out=` ufuncs; MLX transfers the inputs once, builds the whole chain lazily on the device, and calls `mx.eval()` once at the end.

```
N=  1  numpy=  26.9ms   mlx= 101.4ms   NumPy 3.8x faster
N=  2  numpy=  53.1ms   mlx= 120.6ms   NumPy 2.3x faster
N=  4  numpy= 104.9ms   mlx= 193.1ms   NumPy 1.8x faster
N=  8  numpy= 211.5ms   mlx= 277.7ms   NumPy 1.3x faster
N= 16  numpy= 404.6ms   mlx= 468.9ms   NumPy 1.2x faster
```

(NumPy's per-step cost here, ~25 ms, is about half the 48 ms single-call figure from the breakdown section. That's because the chain updates its output buffer in place, while the compiled kernel allocates a fresh scratch buffer on every call — and the first-touch page faults on that fresh 400 MB allocation roughly double the cost of its first pass.)

![Line chart of total time vs N, the number of chained steps of x = x * b + c at 100M elements. Both lines rise nearly linearly: NumPy (blue) from 27ms at N=1 to 405ms at N=16 with slope ~25.2 ms/step; MLX (orange) from 101ms at N=1 to 469ms at N=16 with slope ~24.4 ms/step. Annotations mark the gap at 3.8x at N=1 and 1.2x at N=16. The lines converge but do not cross in the measured range.](./images/02-amortization-curve.png)

The data says two things:

1. **Transfer amortization works.** MLX's ~78 ms of transfers is paid once regardless of N, so the gap shrinks from 3.8x at N=1 to 1.2x at N=16.
2. **It's not enough to pull ahead — and that's the more interesting finding.** The per-step costs converge: ~25.2 ms/step for NumPy, ~24.4 ms/step for MLX. Each elementwise step is a full pass over ~2 GB of array data, so both backends spend the step streaming DRAM — and on Apple Silicon it's the *same* DRAM. Once the transfers amortize away, the GPU has no bandwidth advantage left to exploit. A linear fit puts the crossover somewhere around N ≈ 100, but with slopes that close the exact number is noise; the fair summary is "parity, eventually."

For a memory-bound elementwise kernel on unified memory, then, the GPU never opens a meaningful lead — even with the transfers designed away. What the GPU needs to win is work with a higher compute-to-data ratio, where its parallelism matters and the DRAM stream isn't the bottleneck.

---

## What this means for picokernel's next step

The transfer cost also points at where the project should go next.

The current MLX backend is a *transparent* backend: the user passes NumPy arrays, gets NumPy arrays back, and MLX is hidden behind the kernel. That's a useful API for testing, but it's the least favorable API for performance.

A device-native kernel API — where inputs and outputs are MLX arrays, and `mx.eval()` is invoked once at a higher level — would remove the transfer cost, which the chained benchmark shows is worth up to ~4x here. But the same benchmark shows that for elementwise kernels, removing transfers only buys parity: closing the rest of the gap requires work with more arithmetic per byte, plus moving up the stack — targeting `mx.compile()` directly, generating Metal shader source, or both.

For now, the lesson is a familiar one in GPU programming: **moving data is usually more expensive than transforming it.** Apple Silicon makes that data movement cheaper than a discrete GPU does, but cheaper isn't free, and at the scales where an accelerator is worth considering, the transfers will dominate unless you design around them.

---

## A note on the profiler: what tracing buys, and what it costs

The findings above came out of a *trace-based* profiler — every operation in the generated code gets explicitly timed, and the result is a detailed event log loadable in Perfetto.

To be fair about what that buys, a standard Python profiler would have found the same top-level split. `cProfile` (a deterministic profiler — it hooks every function call, including C-implemented ones, and times each as an opaque unit) would report per-function totals for `mx.array`, `mx.eval`, and `np.array`; PyInstrument (a sampling profiler) would attribute the blocked time to the same call sites. What neither produces is the structure the analysis above leaned on: individual events in program order on a timeline, tagged with semantic categories (`h2d` / `op_lazy` / `sync` / `d2h`), matched one-to-one with the ops in the generated source, and mergeable with the NumPy backend's trace for side-by-side comparison in Perfetto.

One limit applies to every Python-side profiler, this one included: none of them see *inside* `mx.eval`. The 25.3 ms sync block is opaque from Python — breaking it into individual Metal shader dispatches requires `mx.metal.start_capture()` and Xcode's Metal debugger.

The tradeoff is overhead, and it's measurable: about 60 μs of fixed cost per profiled call — ~50 μs of which is re-`exec()`ing the instrumented version of the kernel — plus ~0.35 μs per timed event. For a 10 μs kernel that's catastrophic: you'd be measuring the profiler, not the code. For a 100 ms kernel, it's below the noise floor.

That overhead profile is why production profilers tend to be sampling-based and debug profilers tend to be trace-based. Use the one that matches the question you're asking.

---

## What's next

The next post in this series will be about matrix multiplication — an operation where the GPU is expected to be faster, because a large matmul does O(n³) arithmetic on O(n²) data: enough compute per byte to swamp the transfer cost *and* escape the DRAM-bandwidth ceiling that capped the elementwise chain above. That's also where the kernel compiler starts to earn its keep: matmul is where tiling, blocking, and loop-level lowering matter, and where the next backend (C codegen, or direct Metal shader generation) becomes worth building.

The general theme is that a fast GPU is not the same as a fast workload. The compiler's job is to make sure the GPU is doing useful work proportional to the data it's been handed. That's the bar for the next post.

---

*Code and benchmarks: [github.com/mani-ananth/picokernel](https://github.com/mani-ananth/picokernel)*

*Previous post: [Building a Kernel Compiler in 500 Lines of Python](./01-compiler-pipeline.md)*
