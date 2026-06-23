---
title: "I Built a Compiler to Race NumPy Against Apple's GPU. NumPy Won Every Round."
subtitle: "A perf deep-dive into MLX, unified memory, and the surprising tax that data transfers impose even when there's no PCIe bus."
date: 2026-06-20
tags: [performance, mlx, apple-silicon, gpu, profiling]
---

I had a hypothesis. The M4 MacBook has a GPU sitting two inches from its CPU, sharing the same DRAM. There's no PCIe bus to cross, no kernel driver to bounce through. Apple even ships [MLX](https://github.com/ml-explore/mlx) — a NumPy-shaped array library that targets that GPU through Metal. So if I wrote a tiny kernel like `(a + b) * c` and pointed both backends at the same arrays, the GPU should win, right? It has more parallelism, more bandwidth, dedicated hardware for exactly this kind of work.

I built [`picokernel`](https://github.com/mani-ananth/picokernel) — a small kernel compiler with both a NumPy backend and an MLX backend — and ran the experiment.

The GPU lost. Every single round.

```
size=     1,000  numpy=  0.002ms   mlx=  0.33ms    NumPy 219x faster
size=    10,000  numpy=  0.003ms   mlx=  0.46ms    NumPy 125x faster
size=   100,000  numpy=  0.023ms   mlx=  0.48ms    NumPy  20x faster
size= 1,000,000  numpy=  0.36ms    mlx=  1.39ms    NumPy   2x faster
size=10,000,000  numpy=  4.65ms    mlx= 12.18ms    NumPy   3x faster
```

![Log-log chart of execution time vs array size for NumPy and MLX backends running (a+b)*c on float32. NumPy line is consistently below MLX line at every measured size from 1K to 10M elements. X-axis: array size (log scale). Y-axis: time in milliseconds (log scale).](./images/02-benchmark-curve.png)

For `(a + b) * c` on float32 arrays, NumPy is faster at every size I tested. The gap narrows from 219x to about 3x as arrays grow, but it never closes.

The interesting question isn't *whether* NumPy won — the interesting question is *why*, and the answer reveals something important about when you should and shouldn't reach for the GPU on Apple Silicon.

To get to the answer I had to build a profiler.

---

## NumPy's secret weapon: vectorized C with no intermediates

Before we get to the GPU, a quick note on why NumPy is so fast in the first place.

When you write `(a + b) * c` in plain NumPy, you get two array allocations: one for `a + b`, then a second for the multiplication. That's wasted memory traffic, and on large arrays it matters.

`picokernel` traces the kernel into an intermediate representation, then lowers it to NumPy code that uses the `out=` parameter on ufuncs to eliminate intermediate allocations entirely. The generated code looks like this:

```python
def kernel(a, b, c, o):
    _buf = np.empty_like(o)              # one scratch buffer
    v2 = np.add(a, b, out=_buf)          # no temporary array
    np.multiply(v2, c, out=o)            # writes directly to output
```

Two operations. One pre-allocated scratch buffer. Zero intermediate allocations. And — critically — every line stays inside NumPy's vectorized C implementation. The Python interpreter is invoked only to call the two ufuncs; the actual element-wise work runs at C speed, with SIMD vectorization courtesy of Apple's Accelerate framework.

This is the baseline the GPU has to beat.

(How does this lowering work? That's the topic of [the previous post](./01-compiler-pipeline.md).)

---

## MLX: lazy by default, eager when you ask

MLX is structured around a key insight: most array operations don't need to run immediately. When you write `v = a * b` in MLX, no GPU work happens. Instead, MLX builds a graph node representing "the result of multiplying `a` and `b`," and returns a handle. The actual computation is deferred until something asks for the result — typically a call to `mx.eval()`.

For our kernel, `picokernel` lowers to:

```python
def kernel(a, b, c, o):
    v0 = mx.array(a)        # h2d: copy host array into MLX-managed buffer
    v1 = mx.array(b)        # h2d
    v2 = v0 + v1            # lazy — no GPU work yet
    v3 = mx.array(c)        # h2d
    v4 = v2 * v3            # lazy
    mx.eval(v4)             # GPU executes here — fuses add + multiply into one Metal kernel
    o[...] = np.array(v4)   # d2h: copy result back to NumPy land
```

Four steps:

1. **h2d transfers** — `mx.array()` copies a NumPy array into a Metal-managed buffer.
2. **Lazy graph construction** — the arithmetic ops are nearly free, just building a DAG.
3. **`mx.eval()`** — this is where the GPU actually runs. MLX fuses the entire chain into a single Metal kernel, dispatches it, and blocks until it's done.
4. **d2h transfer** — `np.array()` on an MLX array copies the result back to NumPy.

This design is *elegant*: lazy evaluation lets MLX see your full computation before deciding how to schedule it, so it can fuse aggressively. The single-kernel fusion is a real win — separate `add` and `multiply` kernels would each have to load `a + b` from GPU memory, but the fused version computes it once and feeds it straight into the multiply.

But there's a catch hidden in steps 1 and 4.

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

Here's what a single 100M-element call looks like, broken down:

```
kernel_call (82ms total)
├─ h2d: mx.array(a)        9.1 ms
├─ h2d: mx.array(b)        9.0 ms
├─ op_lazy: add            0.5 μs
├─ h2d: mx.array(c)        9.4 ms
├─ op_lazy: multiply       0.4 μs
├─ sync: mx.eval          31.6 ms
└─ d2h: np.array         22.1 ms
```

![Screenshot of a Perfetto timeline showing the MLX kernel_call span with nested events. Three h2d transfers (~9ms each, color: orange) appear first, followed by two nearly invisible lazy ops, then the wide mx.eval sync block (~31.6ms, color: blue), and finally the d2h transfer (~22ms, color: red). Total span: 82ms.](./images/02-perfetto-trace.png)

![Stacked bar chart: total 82ms split into three categories. h2d (27.5ms / 34%, orange), sync/GPU compute (31.6ms / 38%, blue), d2h (22.1ms / 28%, red). Labels show each bucket as both absolute time and percentage of total.](./images/02-time-breakdown.png)

Add it up: **27.5 ms** in `h2d`, **22 ms** in `d2h`, **31.6 ms** in actual GPU compute. The lazy graph construction is genuinely free. The compute itself is fast — NumPy needed 51 ms in two passes to do the same arithmetic, so MLX's fused kernel wins by 20 ms on raw compute.

But the data movement around it costs 50 ms. Net loss: 30 ms.

This is the entire story of why MLX lost.

---

## But wait — unified memory was supposed to fix this

This was the part that surprised me. The M4 has unified memory: CPU and GPU share the same DRAM. There's no PCIe bus, no separate VRAM. So why does `mx.array(a)` cost 9 milliseconds when the data is already in the right physical chip?

The short answer: *physical sharing isn't the same as buffer sharing*.

When you call `mx.array(numpy_array)`, MLX has to:

1. Allocate a Metal buffer through the GPU's memory allocator.
2. Copy the NumPy data into that buffer, because the NumPy array isn't owned by Metal and might be freed, resized, or modified at any time.
3. Tag the buffer for GPU access (handles, residency tracking, etc.).

Even when the source and destination live on the same physical DRAM, you're still doing a memcpy of ~381 MB (100M float32 elements). Apple's unified memory removes the PCIe bus, but it doesn't eliminate the cost of moving bytes from one logical buffer to another.

In rough terms, that's ~42 GB/s of effective copy bandwidth — which is bandwidth, not magic. The M4's DRAM peak is much higher, but the memcpy is single-threaded from Python's perspective and has to go through the allocator.

![Diagram of M4 unified memory architecture: a single DRAM block at the bottom with a CPU and GPU sharing access at the top. Two separate boxes labeled "NumPy buffer" and "Metal buffer" both inside the DRAM region, with an arrow labeled "mx.array() copy: 9 ms / 381 MB" between them. Caption: "Same physical memory, different logical buffers — the copy is still real."](./images/02-unified-memory.png)

The implication is uncomfortable: **on Apple Silicon, the GPU is "free" to reach, but data is not free to hand it.**

---

## The d2h asymmetry

One small mystery from the trace: the three h2d transfers each took ~9ms, but the single d2h cost 22ms — more than twice as long. Both are moving the same amount of data over the same DRAM. Why the asymmetry?

I don't have a fully confirmed answer, but the most likely explanation is GPU synchronization. `np.array(mlx_array)` has to ensure the GPU has finished writing the result before NumPy can read it — even after `mx.eval()` has returned. There may be additional buffer state transitions, format conversions, or cache flushes happening on the d2h path that the h2d direction doesn't need.

This would be a great question to investigate with `mx.metal.start_capture()` and the Xcode Metal debugger, which shows per-shader GPU timing. I'll leave that for a future post.

---

## When MLX actually wins

If transfers dominate, the way to make MLX win is obvious: **don't transfer**.

The benchmark above assumes you start with a NumPy array and want a NumPy array back. That's a terrible fit for the GPU. Every kernel call pays the full transfer tax, then does microseconds of arithmetic, then pays the tax again.

But that's a benchmark choice, not an MLX limitation. In a real MLX workflow, you'd:

1. Build your arrays in MLX from the start (`mx.random.normal(shape)` instead of `np.random.normal(shape)`).
2. Chain dozens of operations without ever calling `np.array()`.
3. Only convert back to NumPy at the very end, for plotting or saving.

In that workflow, the h2d cost is paid once, the d2h cost is paid once, and the GPU compute amortizes across everything in between. This is exactly how PyTorch users keep tensors on the GPU between operations and how JAX users keep arrays on the accelerator across `jit`-compiled function boundaries.

**The rule of thumb:** if you're computing N operations on the same data, the GPU starts winning when `N × (gpu_compute_savings_per_op) > 2 × transfer_cost`. For a fused `(a+b)*c`, the compute savings was 20ms and the transfer cost was 50ms, so N=1 loses by 30ms. If you were chaining 10 such operations on the same arrays, N=10 wins by 150ms.

![Line chart with N (number of chained ops on the same arrays) on the X-axis and total time on the Y-axis. NumPy line is a straight upward slope. MLX line starts higher (because of the one-time transfer cost) but has a shallower slope. The two lines cross at around N=3, and MLX pulls further ahead as N grows. Annotation on the crossover point: "GPU starts winning here."](./images/02-amortization-curve.png)

The benchmark above is the worst case for the GPU. Real workloads can look very different.

---

## What this means for picokernel's next step

The transfer tax also tells me where the project should go next.

The current MLX backend is a *transparent* backend: the user passes NumPy arrays, gets NumPy arrays back, and MLX is hidden behind the kernel. That's a useful API for testing, but it's the worst possible API for performance.

A device-native kernel API — where inputs and outputs are MLX arrays, and `mx.eval()` is invoked once at a higher level — would close the gap. So would moving up the stack: targeting `mx.compile()` directly, generating Metal shader source, or both.

For now, the lesson is the one every GPU programmer eventually learns the hard way: **moving data is almost always more expensive than transforming it.** Apple Silicon makes that data movement cheaper than a discrete GPU does, but "cheaper" isn't "free," and at the scales where you might want a GPU at all, the transfers will dominate unless you design around them.

---

## Trace-based vs. sampling profilers

A quick aside on tools. The findings above came out of a *trace-based* profiler — every operation in the generated code gets explicitly timed, and the result is a detailed event log loadable in Perfetto.

This was the right tool for this job, because I needed to see *inside* `mx.eval` — to know that the GPU compute was 31.6 ms versus the 50 ms of transfers around it. A sampling profiler (which periodically interrupts the process and records what it finds) wouldn't have helped, because `mx.eval` blocks in C code that PyInstrument or `cProfile` can't peer into.

The tradeoff is overhead. Trace-based profiling adds about 60 μs of fixed cost per call (mostly the cost of re-compiling the instrumented version of the kernel) and ~0.3 μs per timed event. For a 10 μs kernel that's catastrophic — you'd be measuring the profiler, not the code. For an 82 ms kernel, it's signal floor.

This is why production profilers tend to be sampling-based, and debug profilers tend to be trace-based. Use the right one for the question you're asking.

---

## What's next

The next post in this series will be about matrix multiplication — the one operation where the GPU genuinely wins at all sizes, because the compute-to-data ratio is high enough to swamp the transfer tax. That's also where the kernel compiler starts to earn its keep: matmul is where tiling, blocking, and loop-level lowering matter, and where the next backend (C codegen, or direct Metal shader generation) becomes worth building.

The general theme: a fast GPU is not the same as a fast workload. The compiler's job is to make sure the GPU is doing useful work proportional to the data it's been handed. That's the bar for the next post.

---

*Code and benchmarks: [github.com/mani-ananth/picokernel](https://github.com/mani-ananth/picokernel)*

*Previous post: [Building a Kernel Compiler in 500 Lines of Python](./01-compiler-pipeline.md)*
