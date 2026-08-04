"""Metal backend tests — skipped if metalcompute is not installed.

Metal computes in float32 (GPUs have no float64), so inputs/outputs are float32
and comparisons use almost_equal.
"""

import numpy as np
import pytest

metalcompute = pytest.importorskip("metalcompute")

import picokernel
from picokernel.loop_ir import lower_to_loops
from picokernel.metal_lowering import lower_to_metal
from picokernel.trace import trace_kernel


def _f32(*vals):
  return [np.asarray(v, dtype=np.float32) for v in vals]


def test_vector_add():
  @picokernel.kernel(backend="metal")
  def k(x_ref, y_ref, o_ref):
    o_ref[...] = x_ref[...] + y_ref[...]

  x, y = _f32([1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0])
  out = np.zeros(4, dtype=np.float32)
  k(x, y, out)
  np.testing.assert_array_almost_equal(out, x + y)


def test_elementwise_mul():
  @picokernel.kernel(backend="metal")
  def k(x, y, o):
    o[...] = x[...] * y[...]

  x, y = _f32([2.0, 3.0, 4.0], [5.0, 6.0, 7.0])
  out = np.zeros(3, dtype=np.float32)
  k(x, y, out)
  np.testing.assert_array_almost_equal(out, x * y)


def test_negate():
  @picokernel.kernel(backend="metal")
  def k(x, o):
    o[...] = -x[...]

  (x,) = _f32([1.0, -2.0, 3.0])
  out = np.zeros(3, dtype=np.float32)
  k(x, out)
  np.testing.assert_array_almost_equal(out, -x)


def test_chained_ops():
  @picokernel.kernel(backend="metal")
  def k(a, b, c, o):
    o[...] = (a[...] + b[...]) * c[...]

  a, b, c = _f32([1.0, 2.0], [3.0, 4.0], [2.0, 3.0])
  out = np.zeros(2, dtype=np.float32)
  k(a, b, c, out)
  np.testing.assert_array_almost_equal(out, (a + b) * c)


def test_scalar_const():
  @picokernel.kernel(backend="metal")
  def k(x, o):
    o[...] = x[...] * 2.0 + 1.0

  (x,) = _f32([1.0, 2.0, 3.0])
  out = np.zeros(3, dtype=np.float32)
  k(x, out)
  np.testing.assert_array_almost_equal(out, x * 2.0 + 1.0)


def test_array_const():
  @picokernel.kernel(backend="metal")
  def k(x, o):
    o[...] = x[...] + np.array([10.0, 20.0, 30.0], dtype=np.float32)

  (x,) = _f32([1.0, 2.0, 3.0])
  out = np.zeros(3, dtype=np.float32)
  k(x, out)
  np.testing.assert_array_almost_equal(out, [11.0, 22.0, 33.0])


def test_2d_elementwise():
  @picokernel.kernel(backend="metal")
  def k(x, y, o):
    o[...] = (x[...] + y[...]) * x[...]

  x = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
  y = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
  out = np.zeros((2, 2), dtype=np.float32)
  k(x, y, out)
  np.testing.assert_array_almost_equal(out, (x + y) * x)


def test_matrix_multiply():
  @picokernel.kernel(backend="metal")
  def k(a, b, o):
    o[...] = a[...] @ b[...]

  a = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
  b = np.array([[5.0, 6.0], [7.0, 8.0]], dtype=np.float32)
  out = np.zeros((2, 2), dtype=np.float32)
  k(a, b, out)
  np.testing.assert_array_almost_equal(out, a @ b)


def test_matrix_multiply_nonsquare():
  @picokernel.kernel(backend="metal")
  def k(a, b, o):
    o[...] = a[...] @ b[...]

  a = np.arange(6, dtype=np.float32).reshape(2, 3)
  b = np.arange(12, dtype=np.float32).reshape(3, 4)
  out = np.zeros((2, 4), dtype=np.float32)
  k(a, b, out)
  np.testing.assert_array_almost_equal(out, a @ b)


def test_retrace_on_shape_change():
  @picokernel.kernel(backend="metal")
  def k(x, o):
    o[...] = x[...] * 2.0

  x1 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
  out1 = np.zeros(3, dtype=np.float32)
  k(x1, out1)
  np.testing.assert_array_almost_equal(out1, x1 * 2.0)

  x2 = np.ones(5, dtype=np.float32)
  out2 = np.zeros(5, dtype=np.float32)
  k(x2, out2)
  np.testing.assert_array_almost_equal(out2, x2 * 2.0)


def test_lower_returns_metal_source():
  @picokernel.kernel(backend="metal")
  def k(x, y, o):
    o[...] = x[...] + y[...]

  x, y = _f32([1.0, 2.0], [3.0, 4.0])
  out = np.zeros(2, dtype=np.float32)
  source = k.lower(x, y, out)
  assert "kernel void k(" in source
  assert "thread_position_in_grid" in source
  assert "device const float*" in source


def test_lower_matmul_has_reduction_loop():
  @picokernel.kernel(backend="metal")
  def k(a, b, o):
    o[...] = a[...] @ b[...]

  a = np.zeros((2, 3), dtype=np.float32)
  b = np.zeros((3, 4), dtype=np.float32)
  out = np.zeros((2, 4), dtype=np.float32)
  source = k.lower(a, b, out)
  assert "for (uint k" in source
  assert "acc +=" in source


def test_lower_to_loops_then_metal_directly():
  def k(x, y, o):
    o[...] = x[...] + y[...]

  ir = trace_kernel(k, shapes=[(4,), (4,), (4,)])
  prog = lower_to_loops(ir)
  source = lower_to_metal(prog)
  assert "kernel void k(" in source


def test_broadcasting_raises():
  @picokernel.kernel(backend="metal")
  def k(x, y, o):
    o[...] = x[...] + y[...]

  x = np.ones((2, 3), dtype=np.float32)
  y = np.ones(3, dtype=np.float32)
  out = np.zeros((2, 3), dtype=np.float32)
  with pytest.raises(NotImplementedError, match="broadcasting"):
    k(x, y, out)


def test_matmul_with_fused_elementwise_raises():
  @picokernel.kernel(backend="metal")
  def k(a, b, bias, o):
    o[...] = (a[...] @ b[...]) + bias[...]

  a = np.eye(2, dtype=np.float32)
  b = np.eye(2, dtype=np.float32)
  bias = np.ones((2, 2), dtype=np.float32)
  out = np.zeros((2, 2), dtype=np.float32)
  with pytest.raises(NotImplementedError, match="matmul"):
    k(a, b, bias, out)
