import numpy as np
import pytest
import torch
import nectarml
from nectarml import typing as nml_typing

###############################################################################
# Configuration
###############################################################################

DEVICE = 'cuda'
ATOL_FWD = {nml_typing.float32: 1e-4, nml_typing.float16: 1e-2}
ATOL_BWD = {nml_typing.float32: 1e-3, nml_typing.float16: 5e-2}
ATOL_GRAD = {nml_typing.float32: 1e-3, nml_typing.float16: 1e-1}

DTYPE_IDS = ['float32', 'float16']
DTYPES = [nml_typing.float32, nml_typing.float16]
TMAP = {nml_typing.float32: torch.float32, nml_typing.float16: torch.float16}

###############################################################################
# Fixtures & helpers
###############################################################################


@pytest.fixture(autouse=True)
def fixed_seed():
    np.random.seed(42)
    torch.manual_seed(42)


def nml(data, dtype, grad=False):
    return nectarml.Tensor(data, requires_grad=grad).to(DEVICE, dtype=dtype)


def tch(data, dtype, grad=False):
    return torch.tensor(
        data, dtype=TMAP[dtype], device=DEVICE, requires_grad=grad
    )


def rdata(*shape):
    return np.random.uniform(0.1, 0.9, shape).astype(np.float32)


def rpos(*shape):
    return np.random.uniform(0.1, 2.0, shape).astype(np.float32)


def runit(*shape):
    return np.random.uniform(-0.9, 0.9, shape).astype(np.float32)


def racosh(*shape):
    return np.random.uniform(1.01, 3.0, shape).astype(np.float32)


def to_np(t):
    if isinstance(t, nectarml.Tensor):
        return t.detach().cpu().numpy().astype(np.float32)
    return t.detach().cpu().float().numpy()


def assert_close(nml_r, torch_r, dtype, atol=None, label=''):
    if atol is None:
        atol = ATOL_FWD[dtype]
    a, b = to_np(nml_r), to_np(torch_r)
    assert (
        a.shape == b.shape
    ), f'{label} shape mismatch: nml={a.shape} torch={b.shape}'
    assert np.allclose(a, b, atol=atol, equal_nan=False), (
        f'{label} max_diff={np.nanmax(np.abs(a-b)):.6f} atol={atol}\n'
        f'  nml:   {a.flat[:8]}\n  torch: {b.flat[:8]}'
    )


def assert_grad(nml_g, torch_g, dtype, atol=None, label=''):
    if atol is None:
        atol = ATOL_BWD[dtype]
    assert nml_g is not None, f'{label} nml grad is None'
    assert torch_g is not None, f'{label} torch grad is None'
    a, b = to_np(nml_g), to_np(torch_g)
    assert (
        a.shape == b.shape
    ), f'{label} grad shape mismatch: nml={a.shape} torch={b.shape}'
    assert np.allclose(a, b, atol=atol, equal_nan=False), (
        f'{label} grad max_diff={np.nanmax(np.abs(a-b)):.6f} atol={atol}\n'
        f'  nml:   {a.flat[:8]}\n  torch: {b.flat[:8]}'
    )


def bwd(out):
    out.sum().backward()


def idx_nml(arr):
    return nectarml.Tensor(arr.astype(np.int32)).to(
        DEVICE, dtype=nml_typing.int32
    )


def idx_tch(arr):
    return torch.tensor(arr.astype(np.int64), device=DEVICE)


###############################################################################
# ROUNDING: forward only (not differentiable)
###############################################################################


class TestRoundingForward:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_floor(self, dtype):
        d = rdata(4, 8)
        assert_close(nml(d, dtype).floor(), tch(d, dtype).floor(), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_ceil(self, dtype):
        d = rdata(4, 8)
        assert_close(nml(d, dtype).ceil(), tch(d, dtype).ceil(), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_round(self, dtype):
        d = rdata(4, 8)
        assert_close(nml(d, dtype).round(), tch(d, dtype).round(), dtype)


###############################################################################
# MATH DUNDERS: forward + backward
###############################################################################


class TestMathDundersForward:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_add(self, dtype):
        a, b = rdata(4, 8), rdata(4, 8)
        assert_close(
            nml(a, dtype) + nml(b, dtype), tch(a, dtype) + tch(b, dtype), dtype
        )
        assert_close(nml(a, dtype) + 2.0, tch(a, dtype) + 2.0, dtype)
        assert_close(2.0 + nml(a, dtype), 2.0 + tch(a, dtype), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_sub(self, dtype):
        a, b = rdata(4, 8), rdata(4, 8)
        assert_close(
            nml(a, dtype) - nml(b, dtype), tch(a, dtype) - tch(b, dtype), dtype
        )
        assert_close(nml(a, dtype) - 0.5, tch(a, dtype) - 0.5, dtype)
        assert_close(1.0 - nml(a, dtype), 1.0 - tch(a, dtype), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_neg(self, dtype):
        d = rdata(4, 8)
        assert_close(-nml(d, dtype), -tch(d, dtype), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_mul(self, dtype):
        a, b = rdata(4, 8), rdata(4, 8)
        assert_close(
            nml(a, dtype) * nml(b, dtype), tch(a, dtype) * tch(b, dtype), dtype
        )
        assert_close(nml(a, dtype) * 2.0, tch(a, dtype) * 2.0, dtype)
        assert_close(2.0 * nml(a, dtype), 2.0 * tch(a, dtype), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_matmul(self, dtype):
        a = np.random.uniform(0.1, 0.5, (4, 8)).astype(np.float32)
        b = np.random.uniform(0.1, 0.5, (8, 4)).astype(np.float32)
        assert_close(
            nml(a, dtype) @ nml(b, dtype),
            tch(a, dtype) @ tch(b, dtype),
            dtype,
            atol=ATOL_BWD[dtype],
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_pow(self, dtype):
        d = rpos(4, 8)
        assert_close(nml(d, dtype) ** 2, tch(d, dtype) ** 2, dtype)
        assert_close(nml(d, dtype) ** 0.5, tch(d, dtype) ** 0.5, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_truediv(self, dtype):
        a, b = rdata(4, 8), rdata(4, 8)
        assert_close(
            nml(a, dtype) / nml(b, dtype), tch(a, dtype) / tch(b, dtype), dtype
        )
        assert_close(nml(a, dtype) / 2.0, tch(a, dtype) / 2.0, dtype)
        assert_close(1.0 / nml(a, dtype), 1.0 / tch(a, dtype), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_abs_dunder(self, dtype):
        d = np.random.uniform(-0.9, 0.9, (4, 8)).astype(np.float32)
        assert_close(abs(nml(d, dtype)), abs(tch(d, dtype)), dtype)


class TestMathDundersBackward:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_add(self, dtype):
        a, b = rdata(4, 8), rdata(4, 8)
        xn, xt = nml(a, dtype, True), tch(a, dtype, True)
        yn, yt = nml(b, dtype, True), tch(b, dtype, True)
        bwd(xn + yn)
        bwd(xt + yt)
        assert_grad(xn.grad, xt.grad, dtype, label='add dx')
        assert_grad(yn.grad, yt.grad, dtype, label='add dy')

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_sub(self, dtype):
        a, b = rdata(4, 8), rdata(4, 8)
        xn, xt = nml(a, dtype, True), tch(a, dtype, True)
        yn, yt = nml(b, dtype, True), tch(b, dtype, True)
        bwd(xn - yn)
        bwd(xt - yt)
        assert_grad(xn.grad, xt.grad, dtype, label='sub dx')
        assert_grad(yn.grad, yt.grad, dtype, label='sub dy')

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_neg(self, dtype):
        d = rdata(4, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(-xn)
        bwd(-xt)
        assert_grad(xn.grad, xt.grad, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_mul(self, dtype):
        a, b = rdata(4, 8), rdata(4, 8)
        xn, xt = nml(a, dtype, True), tch(a, dtype, True)
        yn, yt = nml(b, dtype, True), tch(b, dtype, True)
        bwd(xn * yn)
        bwd(xt * yt)
        assert_grad(xn.grad, xt.grad, dtype, label='mul dx')
        assert_grad(yn.grad, yt.grad, dtype, label='mul dy')

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_matmul(self, dtype):
        a = np.random.uniform(0.1, 0.5, (4, 8)).astype(np.float32)
        b = np.random.uniform(0.1, 0.5, (8, 4)).astype(np.float32)
        xn, xt = nml(a, dtype, True), tch(a, dtype, True)
        yn, yt = nml(b, dtype, True), tch(b, dtype, True)
        bwd(xn @ yn)
        bwd(xt @ yt)
        assert_grad(
            xn.grad, xt.grad, dtype, atol=ATOL_BWD[dtype], label='matmul dx'
        )
        assert_grad(
            yn.grad, yt.grad, dtype, atol=ATOL_BWD[dtype], label='matmul dy'
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_pow(self, dtype):
        d = rpos(4, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(xn**2)
        bwd(xt**2)
        assert_grad(xn.grad, xt.grad, dtype, label='pow int')
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(xn**0.5)
        bwd(xt**0.5)
        assert_grad(xn.grad, xt.grad, dtype, label='pow float')

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_truediv(self, dtype):
        a, b = rdata(4, 8), rdata(4, 8)
        xn, xt = nml(a, dtype, True), tch(a, dtype, True)
        yn, yt = nml(b, dtype, True), tch(b, dtype, True)
        bwd(xn / yn)
        bwd(xt / yt)
        assert_grad(xn.grad, xt.grad, dtype, label='div dx')
        assert_grad(yn.grad, yt.grad, dtype, label='div dy')

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_abs_dunder(self, dtype):
        d = rpos(4, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(abs(xn))
        bwd(abs(xt))
        assert_grad(xn.grad, xt.grad, dtype)


###############################################################################
# CLAMP: forward + backward
###############################################################################


class TestClampForward:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_minimum(self, dtype):
        a, b = rdata(4, 8), rdata(4, 8)
        assert_close(
            nml(a, dtype).minimum(nml(b, dtype)),
            torch.minimum(tch(a, dtype), tch(b, dtype)),
            dtype,
        )
        assert_close(
            nml(a, dtype).minimum(0.5), tch(a, dtype).clamp(max=0.5), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_maximum(self, dtype):
        a, b = rdata(4, 8), rdata(4, 8)
        assert_close(
            nml(a, dtype).maximum(nml(b, dtype)),
            torch.maximum(tch(a, dtype), tch(b, dtype)),
            dtype,
        )
        assert_close(
            nml(a, dtype).maximum(0.5), tch(a, dtype).clamp(min=0.5), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_clamp(self, dtype):
        d = rdata(4, 8)
        assert_close(
            nml(d, dtype).clamp(0.2, 0.8), tch(d, dtype).clamp(0.2, 0.8), dtype
        )
        assert_close(
            nml(d, dtype).clamp(min_value=0.3),
            tch(d, dtype).clamp(min=0.3),
            dtype,
        )
        assert_close(
            nml(d, dtype).clamp(max_value=0.7),
            tch(d, dtype).clamp(max=0.7),
            dtype,
        )


class TestClampBackward:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_minimum(self, dtype):
        a, b = rdata(4, 8), rdata(4, 8)
        xn, xt = nml(a, dtype, True), tch(a, dtype, True)
        yn, yt = nml(b, dtype, True), tch(b, dtype, True)
        bwd(xn.minimum(yn))
        bwd(torch.minimum(xt, yt))
        assert_grad(xn.grad, xt.grad, dtype, label='min dx')
        assert_grad(yn.grad, yt.grad, dtype, label='min dy')

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_maximum(self, dtype):
        a, b = rdata(4, 8), rdata(4, 8)
        xn, xt = nml(a, dtype, True), tch(a, dtype, True)
        yn, yt = nml(b, dtype, True), tch(b, dtype, True)
        bwd(xn.maximum(yn))
        bwd(torch.maximum(xt, yt))
        assert_grad(xn.grad, xt.grad, dtype, label='max dx')
        assert_grad(yn.grad, yt.grad, dtype, label='max dy')

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_clamp(self, dtype):
        d = rdata(4, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(xn.clamp(0.2, 0.8))
        bwd(xt.clamp(0.2, 0.8))
        assert_grad(xn.grad, xt.grad, dtype)


###############################################################################
# MATH METHODS: forward + backward
###############################################################################


class TestMathForward:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_abs(self, dtype):
        d = np.random.uniform(-0.9, 0.9, (4, 8)).astype(np.float32)
        assert_close(nml(d, dtype).abs(), tch(d, dtype).abs(), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_exp(self, dtype):
        d = np.random.uniform(-1.0, 1.0, (4, 8)).astype(np.float32)
        assert_close(nml(d, dtype).exp(), tch(d, dtype).exp(), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_log(self, dtype):
        d = rpos(4, 8)
        assert_close(nml(d, dtype).log(), tch(d, dtype).log(), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_log2(self, dtype):
        d = rpos(4, 8)
        assert_close(nml(d, dtype).log2(), tch(d, dtype).log2(), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_log10(self, dtype):
        d = rpos(4, 8)
        assert_close(nml(d, dtype).log10(), tch(d, dtype).log10(), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_sqrt(self, dtype):
        d = rpos(4, 8)
        assert_close(nml(d, dtype).sqrt(), tch(d, dtype).sqrt(), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_rsqrt(self, dtype):
        d = rpos(4, 8)
        assert_close(nml(d, dtype).rsqrt(), tch(d, dtype).rsqrt(), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_sin(self, dtype):
        d = runit(4, 8)
        assert_close(nml(d, dtype).sin(), tch(d, dtype).sin(), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_asin(self, dtype):
        d = runit(4, 8)
        assert_close(nml(d, dtype).asin(), tch(d, dtype).asin(), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_sinh(self, dtype):
        d = np.random.uniform(-1.0, 1.0, (4, 8)).astype(np.float32)
        assert_close(nml(d, dtype).sinh(), tch(d, dtype).sinh(), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_asinh(self, dtype):
        d = runit(4, 8)
        assert_close(nml(d, dtype).asinh(), tch(d, dtype).asinh(), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_cos(self, dtype):
        d = runit(4, 8)
        assert_close(nml(d, dtype).cos(), tch(d, dtype).cos(), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_acos(self, dtype):
        d = runit(4, 8)
        assert_close(nml(d, dtype).acos(), tch(d, dtype).acos(), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_cosh(self, dtype):
        d = np.random.uniform(-1.0, 1.0, (4, 8)).astype(np.float32)
        assert_close(nml(d, dtype).cosh(), tch(d, dtype).cosh(), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_acosh(self, dtype):
        d = racosh(4, 8)
        assert_close(nml(d, dtype).acosh(), tch(d, dtype).acosh(), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_tan(self, dtype):
        d = runit(4, 8)
        assert_close(nml(d, dtype).tan(), tch(d, dtype).tan(), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_tanh(self, dtype):
        d = runit(4, 8)
        assert_close(nml(d, dtype).tanh(), tch(d, dtype).tanh(), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_atan(self, dtype):
        d = runit(4, 8)
        assert_close(nml(d, dtype).atan(), tch(d, dtype).atan(), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_atanh(self, dtype):
        d = runit(4, 8)
        assert_close(nml(d, dtype).atanh(), tch(d, dtype).atanh(), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_atan2(self, dtype):
        a, b = rdata(4, 8), rdata(4, 8)
        assert_close(
            nml(a, dtype).atan2(nml(b, dtype)),
            torch.atan2(tch(b, dtype), tch(a, dtype)),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_sign(self, dtype):
        d = np.random.uniform(-0.9, 0.9, (4, 8)).astype(np.float32)
        assert_close(nml(d, dtype).sign(), tch(d, dtype).sign(), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_copysign(self, dtype):
        a = rdata(4, 8)
        b = np.random.uniform(-0.9, 0.9, (4, 8)).astype(np.float32)
        assert_close(
            nml(a, dtype).copysign(nml(b, dtype)),
            torch.copysign(tch(a, dtype), tch(b, dtype)),
            dtype,
        )


class TestMathBackward:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_abs(self, dtype):
        d = rpos(4, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(xn.abs())
        bwd(xt.abs())
        assert_grad(xn.grad, xt.grad, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_exp(self, dtype):
        d = np.random.uniform(-1.0, 1.0, (4, 8)).astype(np.float32)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(xn.exp())
        bwd(xt.exp())
        assert_grad(xn.grad, xt.grad, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_log(self, dtype):
        d = rpos(4, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(xn.log())
        bwd(xt.log())
        assert_grad(xn.grad, xt.grad, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_sqrt(self, dtype):
        d = rpos(4, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(xn.sqrt())
        bwd(xt.sqrt())
        assert_grad(xn.grad, xt.grad, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_sin(self, dtype):
        d = runit(4, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(xn.sin())
        bwd(xt.sin())
        assert_grad(xn.grad, xt.grad, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_cos(self, dtype):
        d = runit(4, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(xn.cos())
        bwd(xt.cos())
        assert_grad(xn.grad, xt.grad, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_tanh(self, dtype):
        d = runit(4, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(xn.tanh())
        bwd(xt.tanh())
        assert_grad(xn.grad, xt.grad, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_atan2(self, dtype):
        a, b = rdata(4, 8), rdata(4, 8)
        xn, xt = nml(a, dtype, True), tch(a, dtype, True)
        yn, yt = nml(b, dtype, True), tch(b, dtype, True)
        bwd(xn.atan2(yn))
        bwd(torch.atan2(xt, yt))
        assert_grad(xn.grad, xt.grad, dtype, label='atan2 dx')
        assert_grad(yn.grad, yt.grad, dtype, label='atan2 dy')


###############################################################################
# REDUCTIONS: forward + backward
###############################################################################


class TestReductionsForward:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_min(self, dtype):
        d = rdata(4, 8)
        assert_close(nml(d, dtype).min(), tch(d, dtype).min(), dtype)
        assert_close(
            nml(d, dtype).min(dim=0).values,
            tch(d, dtype).min(dim=0).values,
            dtype,
        )
        assert_close(
            nml(d, dtype).min(dim=-1).values,
            tch(d, dtype).min(dim=-1).values,
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_amin(self, dtype):
        d = rdata(2, 4, 8)
        assert_close(nml(d, dtype).amin(), tch(d, dtype).amin(), dtype)
        assert_close(
            nml(d, dtype).amin(dim=0), tch(d, dtype).amin(dim=0), dtype
        )
        assert_close(
            nml(d, dtype).amin(dim=-1), tch(d, dtype).amin(dim=-1), dtype
        )
        assert_close(
            nml(d, dtype).amin(dim=1, keepdim=True),
            tch(d, dtype).amin(1, keepdim=True),
            dtype,
        )
        assert_close(
            nml(d, dtype).amin(dim=(0, 1)),
            tch(d, dtype).amin(dim=(0, 1)),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_max(self, dtype):
        d = rdata(4, 8)
        assert_close(nml(d, dtype).max(), tch(d, dtype).max(), dtype)
        assert_close(
            nml(d, dtype).max(dim=0).values,
            tch(d, dtype).max(dim=0).values,
            dtype,
        )
        assert_close(
            nml(d, dtype).max(dim=-1).values,
            tch(d, dtype).max(dim=-1).values,
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_amax(self, dtype):
        d = rdata(2, 4, 8)
        assert_close(nml(d, dtype).amax(), tch(d, dtype).amax(), dtype)
        assert_close(
            nml(d, dtype).amax(dim=0), tch(d, dtype).amax(dim=0), dtype
        )
        assert_close(
            nml(d, dtype).amax(dim=-1), tch(d, dtype).amax(dim=-1), dtype
        )
        assert_close(
            nml(d, dtype).amax(dim=1, keepdim=True),
            tch(d, dtype).amax(1, keepdim=True),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_argmin(self, dtype):
        d = rdata(4, 8)
        assert_close(nml(d, dtype).argmin(), tch(d, dtype).argmin(), dtype)
        assert_close(
            nml(d, dtype).argmin(dim=0), tch(d, dtype).argmin(dim=0), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_argmax(self, dtype):
        d = rdata(4, 8)
        assert_close(nml(d, dtype).argmax(), tch(d, dtype).argmax(), dtype)
        assert_close(
            nml(d, dtype).argmax(dim=0), tch(d, dtype).argmax(dim=0), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_mean(self, dtype):
        d = rdata(2, 4, 8)
        assert_close(nml(d, dtype).mean(), tch(d, dtype).mean(), dtype)
        assert_close(
            nml(d, dtype).mean(dim=0), tch(d, dtype).mean(dim=0), dtype
        )
        assert_close(
            nml(d, dtype).mean(dim=-1), tch(d, dtype).mean(dim=-1), dtype
        )
        assert_close(
            nml(d, dtype).mean(dim=1, keepdim=True),
            tch(d, dtype).mean(1, keepdim=True),
            dtype,
        )
        assert_close(
            nml(d, dtype).mean(dim=(0, 2)),
            tch(d, dtype).mean(dim=(0, 2)),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_sum(self, dtype):
        d = rdata(2, 4, 8)
        assert_close(nml(d, dtype).sum(), tch(d, dtype).sum(), dtype)
        assert_close(nml(d, dtype).sum(dim=0), tch(d, dtype).sum(dim=0), dtype)
        assert_close(
            nml(d, dtype).sum(dim=-1), tch(d, dtype).sum(dim=-1), dtype
        )
        assert_close(
            nml(d, dtype).sum(dim=1, keepdim=True),
            tch(d, dtype).sum(1, keepdim=True),
            dtype,
        )
        assert_close(
            nml(d, dtype).sum(dim=(0, 2)), tch(d, dtype).sum(dim=(0, 2)), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_cumsum(self, dtype):
        d = rdata(4, 8)
        assert_close(
            nml(d, dtype).cumsum(dim=0), tch(d, dtype).cumsum(dim=0), dtype
        )
        assert_close(
            nml(d, dtype).cumsum(dim=1), tch(d, dtype).cumsum(dim=1), dtype
        )
        assert_close(
            nml(d, dtype).cumsum(dim=-1), tch(d, dtype).cumsum(dim=-1), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_prod(self, dtype):
        d = np.random.uniform(0.5, 1.5, (2, 4, 4)).astype(np.float32)
        assert_close(nml(d, dtype).prod(), tch(d, dtype).prod(), dtype)
        assert_close(
            nml(d, dtype).prod(dim=0), tch(d, dtype).prod(dim=0), dtype
        )
        assert_close(
            nml(d, dtype).prod(dim=-1), tch(d, dtype).prod(dim=-1), dtype
        )
        assert_close(
            nml(d, dtype).prod(dim=1, keepdim=True),
            tch(d, dtype).prod(1, keepdim=True),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_std(self, dtype):
        d = rdata(2, 4, 8)
        assert_close(nml(d, dtype).std(), tch(d, dtype).std(), dtype)
        assert_close(nml(d, dtype).std(dim=0), tch(d, dtype).std(dim=0), dtype)
        assert_close(
            nml(d, dtype).std(dim=-1), tch(d, dtype).std(dim=-1), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_norm(self, dtype):
        d = rpos(4, 8)
        assert_close(
            nml(d, dtype).norm('fro'), tch(d, dtype).norm('fro'), dtype
        )
        assert_close(nml(d, dtype).norm('l1'), tch(d, dtype).norm(1), dtype)


class TestReductionsBackward:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_min(self, dtype):
        d = rdata(4, 8)
        for dim in [0, 1]:
            xn, xt = nml(d, dtype, True), tch(d, dtype, True)
            bwd(xn.min(dim=dim).values)
            bwd(xt.min(dim=dim).values)
            assert_grad(xn.grad, xt.grad, dtype, label=f'min dim={dim}')

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_amin(self, dtype):
        d = rdata(2, 4, 8)
        for dim in [0, 1, -1]:
            xn, xt = nml(d, dtype, True), tch(d, dtype, True)
            bwd(xn.amin(dim=dim))
            bwd(xt.amin(dim=dim))
            assert_grad(xn.grad, xt.grad, dtype, label=f'amin dim={dim}')

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_mean(self, dtype):
        d = rdata(2, 4, 8)
        for dim in [None, 0, 1, -1]:
            xn, xt = nml(d, dtype, True), tch(d, dtype, True)
            bwd(xn.mean(dim=dim))
            bwd(xt.mean(dim=dim))
            assert_grad(xn.grad, xt.grad, dtype, label=f'mean dim={dim}')

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_sum(self, dtype):
        d = rdata(2, 4, 8)
        for dim in [None, 0, 1, -1]:
            xn, xt = nml(d, dtype, True), tch(d, dtype, True)
            bwd(xn.sum(dim=dim))
            bwd(xt.sum(dim=dim))
            assert_grad(xn.grad, xt.grad, dtype, label=f'sum dim={dim}')

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_cumsum(self, dtype):
        d = rdata(4, 8)
        for dim in [0, 1, -1]:
            xn, xt = nml(d, dtype, True), tch(d, dtype, True)
            bwd(xn.cumsum(dim=dim))
            bwd(xt.cumsum(dim=dim))
            assert_grad(xn.grad, xt.grad, dtype, label=f'cumsum dim={dim}')

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_norm(self, dtype):
        d = rpos(4, 8)
        for p, tp in [('fro', 'fro'), ('l1', 1)]:
            xn, xt = nml(d, dtype, True), tch(d, dtype, True)
            bwd(xn.norm(p))
            bwd(xt.norm(tp))
            assert_grad(xn.grad, xt.grad, dtype, label=f'norm {p}')


###############################################################################
# SORTING
###############################################################################


class TestSortingForward:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_sort(self, dtype):
        d = rdata(4, 8)
        for dim in [0, 1, -1]:
            nv, _ = nml(d, dtype).sort(dim=dim)
            tv, _ = tch(d, dtype).sort(dim=dim)
            assert_close(nv, tv, dtype, label=f'sort dim={dim}')
        nv, _ = nml(d, dtype).sort(dim=0, descending=True)
        tv, _ = tch(d, dtype).sort(dim=0, descending=True)
        assert_close(nv, tv, dtype, label='sort desc')

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_argsort(self, dtype):
        d = np.arange(32, dtype=np.float32).reshape(4, 8)
        np.random.shuffle(d.flat)
        assert_close(
            nml(d, dtype).argsort(dim=0),
            tch(d, dtype).argsort(dim=0).float(),
            dtype,
        )
        assert_close(
            nml(d, dtype).argsort(dim=1),
            tch(d, dtype).argsort(dim=1).float(),
            dtype,
        )


class TestSortingBackward:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_sort(self, dtype):
        d = rdata(4, 8)
        for dim in [0, 1]:
            xn, xt = nml(d, dtype, True), tch(d, dtype, True)
            nv, _ = xn.sort(dim=dim)
            tv, _ = xt.sort(dim=dim)
            bwd(nv)
            bwd(tv)
            assert_grad(xn.grad, xt.grad, dtype, label=f'sort dim={dim}')


###############################################################################
# INDEXING: forward + backward
###############################################################################


class TestIndexingForward:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_scatter_add(self, dtype):
        src = rdata(4, 8)
        ix = np.random.randint(0, 4, (4, 8)).astype(np.int32)
        base = np.zeros((4, 8), dtype=np.float32)
        assert_close(
            nml(base, dtype).scatter_add(0, idx_nml(ix), nml(src, dtype)),
            tch(base, dtype).scatter_add(0, idx_tch(ix), tch(src, dtype)),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_masked_fill(self, dtype):
        d = rdata(4, 8)
        mask = (np.random.rand(4, 8) > 0.5).astype(np.float32)
        mn = nectarml.Tensor(mask).to(DEVICE, dtype=nml_typing.float32)
        mt = torch.tensor(mask, dtype=torch.bool, device=DEVICE)
        assert_close(
            nml(d, dtype).masked_fill(mn, -1.0),
            tch(d, dtype).masked_fill(mt, -1.0),
            dtype,
        )


class TestIndexingBackward:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_scatter_add(self, dtype):
        src = rdata(4, 8)
        ix = np.random.randint(0, 4, (4, 8)).astype(np.int32)
        base = np.zeros((4, 8), dtype=np.float32)
        xn, xt = nml(base, dtype, True), tch(base, dtype, True)
        sn, st = nml(src, dtype, True), tch(src, dtype, True)
        bwd(xn.scatter_add(0, idx_nml(ix), sn))
        bwd(xt.scatter_add(0, idx_tch(ix), st))
        assert_grad(xn.grad, xt.grad, dtype, label='scatter_add base')
        assert_grad(sn.grad, st.grad, dtype, label='scatter_add src')

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_masked_fill(self, dtype):
        np.random.seed(42)
        d = rdata(4, 8)
        mask = (np.random.rand(4, 8) > 0.5).astype(np.float32)
        mn = nectarml.Tensor(mask).to(DEVICE, dtype=nml_typing.float32)
        mt = torch.tensor(mask, dtype=torch.bool, device=DEVICE)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(xn.masked_fill(mn, -1.0))
        bwd(xt.masked_fill(mt, -1.0))
        assert_grad(xn.grad, xt.grad, dtype)


###############################################################################
# SHAPE OPS: forward + backward
###############################################################################


class TestGetSetItem:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_getitem_forward(self, dtype):
        d = rdata(4, 8)
        x, xt = nml(d, dtype), tch(d, dtype)
        assert_close(x[0], xt[0], dtype)
        assert_close(x[1:3], xt[1:3], dtype)
        assert_close(x[1:3, 2:5], xt[1:3, 2:5], dtype)
        assert_close(x[::2], xt[::2], dtype)
        assert_close(x[-1], xt[-1], dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_getitem_backward(self, dtype):
        d = rdata(4, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(xn[1:3])
        bwd(xt[1:3])
        assert_grad(xn.grad, xt.grad, dtype, atol=ATOL_GRAD[dtype])

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_setitem_forward(self, dtype):
        d = rdata(4, 8)
        xn, xt = nml(d, dtype), tch(d, dtype)
        xn[0] = 0.0
        xt[0] = 0.0
        assert_close(xn, xt, dtype)
        xn[1:3] = 1.0
        xt[1:3] = 1.0
        assert_close(xn, xt, dtype)


class TestReshaping:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_reshape(self, dtype):
        d = rdata(2, 4, 8)
        x, xt = nml(d, dtype), tch(d, dtype)
        assert_close(x.reshape(8, 8), xt.reshape(8, 8), dtype)
        assert_close(x.reshape(-1), xt.reshape(-1), dtype)
        assert_close(x.reshape(4, 16), xt.reshape(4, 16), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_reshape_backward(self, dtype):
        d = rdata(2, 4, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(xn.reshape(8, 8))
        bwd(xt.reshape(8, 8))
        assert_grad(xn.grad, xt.grad, dtype, atol=ATOL_GRAD[dtype])

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_view(self, dtype):
        d = rdata(2, 4, 8)
        x, xt = nml(d, dtype), tch(d, dtype)
        assert_close(x.view(8, 8), xt.view(8, 8), dtype)
        assert_close(x.view(-1, 8), xt.view(-1, 8), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_flatten(self, dtype):
        d = rdata(2, 4, 8)
        x, xt = nml(d, dtype), tch(d, dtype)
        assert_close(x.flatten(), xt.flatten(), dtype)
        assert_close(x.flatten(1, 2), xt.flatten(1, 2), dtype)
        assert_close(x.flatten(0, 1), xt.flatten(0, 1), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_squeeze(self, dtype):
        d = rdata(1, 4, 1, 8)
        x, xt = nml(d, dtype), tch(d, dtype)
        assert_close(x.squeeze(), xt.squeeze(), dtype)
        assert_close(x.squeeze(0), xt.squeeze(0), dtype)
        assert_close(x.squeeze(2), xt.squeeze(2), dtype)
        assert_close(x.squeeze(-2), xt.squeeze(-2), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_unsqueeze(self, dtype):
        d = rdata(4, 8)
        x, xt = nml(d, dtype), tch(d, dtype)
        assert_close(x.unsqueeze(0), xt.unsqueeze(0), dtype)
        assert_close(x.unsqueeze(1), xt.unsqueeze(1), dtype)
        assert_close(x.unsqueeze(-1), xt.unsqueeze(-1), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_permute(self, dtype):
        d = rdata(2, 4, 8)
        x, xt = nml(d, dtype), tch(d, dtype)
        assert_close(x.permute(2, 0, 1), xt.permute(2, 0, 1), dtype)
        assert_close(x.permute(1, 2, 0), xt.permute(1, 2, 0), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_permute_backward(self, dtype):
        d = rdata(2, 4, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(xn.permute(2, 0, 1))
        bwd(xt.permute(2, 0, 1))
        assert_grad(xn.grad, xt.grad, dtype, atol=ATOL_GRAD[dtype])

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_transpose(self, dtype):
        d = rdata(4, 8)
        x, xt = nml(d, dtype), tch(d, dtype)
        assert_close(x.transpose(0, 1), xt.transpose(0, 1), dtype)
        d3 = rdata(2, 4, 8)
        x, xt = nml(d3, dtype), tch(d3, dtype)
        assert_close(x.transpose(0, 2), xt.transpose(0, 2), dtype)
        assert_close(x.transpose(-2, -1), xt.transpose(-2, -1), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_swapdims(self, dtype):
        d = rdata(2, 4, 8)
        x, xt = nml(d, dtype), tch(d, dtype)
        assert_close(x.swapdims(0, 2), xt.swapdims(0, 2), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_expand(self, dtype):
        d = rdata(1, 4, 1, 8)
        x, xt = nml(d, dtype), tch(d, dtype)
        assert_close(x.expand((2, 4, 3, 8)), xt.expand((2, 4, 3, 8)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_expand_backward(self, dtype):
        d = rdata(1, 4, 1, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(xn.expand((2, 4, 3, 8)))
        bwd(xt.expand((2, 4, 3, 8)))
        assert_grad(xn.grad, xt.grad, dtype, atol=ATOL_GRAD[dtype])

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_broadcast_to(self, dtype):
        d = rdata(1, 8)
        x, xt = nml(d, dtype), tch(d, dtype)
        assert_close(x.broadcast_to((4, 8)), xt.broadcast_to((4, 8)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_unfold(self, dtype):
        d = rdata(4, 16)
        x, xt = nml(d, dtype), tch(d, dtype)
        assert_close(x.unfold(1, 4, 2), xt.unfold(1, 4, 2), dtype)
        assert_close(x.unfold(0, 2, 1), xt.unfold(0, 2, 1), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_flip(self, dtype):
        d = rdata(4, 8)
        x, xt = nml(d, dtype), tch(d, dtype)
        assert_close(x.flip(0), xt.flip([0]), dtype)
        assert_close(x.flip(1), xt.flip([1]), dtype)
        assert_close(x.flip(-1), xt.flip([-1]), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_flip_backward(self, dtype):
        d = rdata(4, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(xn.flip(0))
        bwd(xt.flip([0]))
        assert_grad(xn.grad, xt.grad, dtype, atol=ATOL_GRAD[dtype])


class TestCombination:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_select(self, dtype):
        d = rdata(4, 8)
        x, xt = nml(d, dtype), tch(d, dtype)
        assert_close(x.select(0, 1), xt.select(0, 1), dtype)
        assert_close(x.select(1, 3), xt.select(1, 3), dtype)
        assert_close(x.select(-1, 0), xt.select(-1, 0), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_select_backward(self, dtype):
        d = rdata(4, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(xn.select(0, 1))
        bwd(xt.select(0, 1))
        assert_grad(xn.grad, xt.grad, dtype, atol=ATOL_GRAD[dtype])

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_unstack(self, dtype):
        d = rdata(3, 4, 8)
        x, xt = nml(d, dtype), tch(d, dtype)
        for i, (n, t) in enumerate(zip(x.unstack(dim=0), xt.unbind(dim=0))):
            assert_close(n, t, dtype, label=f'unstack part={i}')

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_unbind(self, dtype):
        d = rdata(3, 4, 8)
        x, xt = nml(d, dtype), tch(d, dtype)
        for i, (n, t) in enumerate(zip(x.unbind(dim=1), xt.unbind(dim=1))):
            assert_close(n, t, dtype, label=f'unbind part={i}')

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_split(self, dtype):
        d = rdata(8, 4)
        x, xt = nml(d, dtype), tch(d, dtype)
        for i, (n, t) in enumerate(zip(x.split(2, dim=0), xt.split(2, dim=0))):
            assert_close(n, t, dtype, label=f'split part={i}')

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_chunk(self, dtype):
        d = rdata(8, 4)
        x, xt = nml(d, dtype), tch(d, dtype)
        for i, (n, t) in enumerate(zip(x.chunk(4, dim=0), xt.chunk(4, dim=0))):
            assert_close(n, t, dtype, label=f'chunk part={i}')


class TestAdvancedIndexing:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_gather(self, dtype):
        d = rdata(4, 8)
        ix0 = np.random.randint(0, 4, (4, 8)).astype(np.int32)
        ix1 = np.random.randint(0, 8, (4, 8)).astype(np.int32)
        x, xt = nml(d, dtype), tch(d, dtype)
        assert_close(
            x.gather(0, idx_nml(ix0)), xt.gather(0, idx_tch(ix0)), dtype
        )
        assert_close(
            x.gather(1, idx_nml(ix1)), xt.gather(1, idx_tch(ix1)), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_gather_backward(self, dtype):
        d = rdata(4, 8)
        ix = np.random.randint(0, 4, (4, 8)).astype(np.int32)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(xn.gather(0, idx_nml(ix)))
        bwd(xt.gather(0, idx_tch(ix)))
        assert_grad(xn.grad, xt.grad, dtype, atol=ATOL_GRAD[dtype])

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_scatter(self, dtype):
        src = rdata(4, 8)
        ix = np.random.randint(0, 4, (4, 8)).astype(np.int32)
        base = np.zeros((4, 8), dtype=np.float32)
        assert_close(
            nml(base, dtype).scatter(0, idx_nml(ix), nml(src, dtype)),
            tch(base, dtype).scatter(0, idx_tch(ix), tch(src, dtype)),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_index_select(self, dtype):
        d = rdata(6, 8)
        ix0 = np.array([0, 2, 4], dtype=np.int32)
        ix1 = np.array([1, 3, 5, 7], dtype=np.int32)
        x, xt = nml(d, dtype), tch(d, dtype)
        assert_close(
            x.index_select(0, idx_nml(ix0)),
            xt.index_select(0, idx_tch(ix0)),
            dtype,
        )
        assert_close(
            x.index_select(1, idx_nml(ix1)),
            xt.index_select(1, idx_tch(ix1)),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_index_select_backward(self, dtype):
        d = rdata(6, 8)
        ix = np.array([0, 2, 4], dtype=np.int32)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(xn.index_select(0, idx_nml(ix)))
        bwd(xt.index_select(0, idx_tch(ix)))
        assert_grad(xn.grad, xt.grad, dtype, atol=ATOL_GRAD[dtype])
