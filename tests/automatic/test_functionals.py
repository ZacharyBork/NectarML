import numpy as np
import pytest
import torch
import torch.nn.functional as TF
import nectarml
import nectarml.nn.functional as F
from nectarml import typing as nml_typing

###############################################################################
# Config
###############################################################################

DEVICE = 'cuda'
ATOL_F32 = 1e-4
ATOL_F16 = 1e-2
ATOL_LOOSE = {nml_typing.float32: 2e-3, nml_typing.float16: 5e-2}
DTYPES = [nml_typing.float32, nml_typing.float16]
DTYPE_IDS = ['float32', 'float16']
TMAP = {nml_typing.float32: torch.float32, nml_typing.float16: torch.float16}
REDUCTIONS = ['mean', 'sum']


@pytest.fixture(autouse=True)
def fixed_seed():
    np.random.seed(42)
    torch.manual_seed(42)


###############################################################################
# Helpers
###############################################################################


def nml(data, dtype, grad=False):
    return nectarml.Tensor(data, requires_grad=grad).to(DEVICE, dtype=dtype)


def tch(data, dtype, grad=False):
    return torch.tensor(
        data, dtype=TMAP[dtype], device=DEVICE, requires_grad=grad
    )


def rdata(*shape):
    return np.random.uniform(0.1, 0.9, shape).astype(np.float32)


def runit(*shape):
    return np.random.uniform(-0.9, 0.9, shape).astype(np.float32)


def rpos(*shape):
    return np.random.uniform(0.1, 2.0, shape).astype(np.float32)


def to_np(t):
    if isinstance(t, nectarml.Tensor):
        return t.detach().cpu().numpy().astype(np.float32)
    return t.detach().cpu().float().numpy()


def atol(dtype):
    return ATOL_F16 if dtype == nml_typing.float16 else ATOL_F32


def assert_close(nml_r, torch_r, dtype, tol=None, label=''):
    if tol is None:
        tol = atol(dtype)
    a, b = to_np(nml_r), to_np(torch_r)
    assert (
        a.shape == b.shape
    ), f'{label} shape mismatch: nml={a.shape} torch={b.shape}'
    assert np.allclose(a, b, atol=tol, equal_nan=False), (
        f'{label} max_diff={np.nanmax(np.abs(a-b)):.6f} tol={tol}\n'
        f'  nml:   {a.flat[:8]}\n  torch: {b.flat[:8]}'
    )


def assert_grad(nml_g, torch_g, dtype, tol=None, label=''):
    if tol is None:
        tol = ATOL_LOOSE[dtype]
    assert nml_g is not None, f'{label} nml grad is None'
    assert torch_g is not None, f'{label} torch grad is None'
    a, b = to_np(nml_g), to_np(torch_g)
    assert (
        a.shape == b.shape
    ), f'{label} grad shape: nml={a.shape} torch={b.shape}'
    assert np.allclose(a, b, atol=tol, equal_nan=False), (
        f'{label} grad max_diff={np.nanmax(np.abs(a-b)):.6f}\n'
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
# ACTIVATION: forward + backward
###############################################################################


class TestActivationForward:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_relu(self, dtype):
        d = runit(4, 8)
        assert_close(F.relu(nml(d, dtype)), TF.relu(tch(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_relu_inplace(self, dtype):
        d = runit(4, 8)
        xn, xt = nml(d, dtype), tch(d, dtype)
        F.relu_(xn)
        TF.relu_(xt)
        assert_close(xn, xt, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_leaky_relu(self, dtype):
        d = runit(4, 8)
        for slope in [0.01, 0.1, 0.2]:
            assert_close(
                F.leaky_relu(nml(d, dtype), slope),
                TF.leaky_relu(tch(d, dtype), slope),
                dtype,
                label=f'slope={slope}',
            )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_elu(self, dtype):
        d = runit(4, 8)
        for alpha in [1.0, 0.5, 2.0]:
            assert_close(
                F.elu(nml(d, dtype), alpha),
                TF.elu(tch(d, dtype), alpha),
                dtype,
                label=f'alpha={alpha}',
            )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_selu(self, dtype):
        d = runit(4, 8)
        assert_close(F.selu(nml(d, dtype)), TF.selu(tch(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_sigmoid(self, dtype):
        d = runit(4, 8)
        assert_close(
            F.sigmoid(nml(d, dtype)), torch.sigmoid(tch(d, dtype)), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_tanh(self, dtype):
        d = runit(4, 8)
        assert_close(F.tanh(nml(d, dtype)), torch.tanh(tch(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_softmax(self, dtype):
        d = runit(2, 8)
        for dim in [-1, 0, 1]:
            assert_close(
                F.softmax(nml(d, dtype), dim=dim),
                TF.softmax(tch(d, dtype), dim=dim),
                dtype,
                label=f'dim={dim}',
            )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_softmin(self, dtype):
        d = runit(2, 8)
        assert_close(
            F.softmin(nml(d, dtype), dim=-1),
            TF.softmin(tch(d, dtype), dim=-1),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_log_softmax(self, dtype):
        d = runit(2, 8)
        assert_close(
            F.log_softmax(nml(d, dtype), dim=-1),
            TF.log_softmax(tch(d, dtype), dim=-1),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_gelu(self, dtype):
        d = runit(4, 8)
        assert_close(
            F.gelu(nml(d, dtype)),
            TF.gelu(tch(d, dtype)),
            dtype,
            tol=ATOL_LOOSE[dtype],
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_silu(self, dtype):
        d = runit(4, 8)
        assert_close(F.silu(nml(d, dtype)), TF.silu(tch(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_swish(self, dtype):
        d = runit(4, 8)
        assert_close(F.swish(nml(d, dtype)), TF.silu(tch(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_softplus(self, dtype):
        d = runit(4, 8)
        assert_close(
            F.softplus(nml(d, dtype)), TF.softplus(tch(d, dtype)), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_mish(self, dtype):
        d = runit(4, 8)
        assert_close(F.mish(nml(d, dtype)), TF.mish(tch(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_hardtanh(self, dtype):
        d = runit(4, 8)
        assert_close(
            F.hardtanh(nml(d, dtype)), TF.hardtanh(tch(d, dtype)), dtype
        )
        assert_close(
            F.hardtanh(nml(d, dtype), -0.5, 0.5),
            TF.hardtanh(tch(d, dtype), -0.5, 0.5),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_hardsigmoid(self, dtype):
        d = runit(4, 8)
        assert_close(
            F.hardsigmoid(nml(d, dtype)), TF.hardsigmoid(tch(d, dtype)), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_hardswish(self, dtype):
        d = runit(4, 8)
        assert_close(
            F.hardswish(nml(d, dtype)), TF.hardswish(tch(d, dtype)), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_softsign(self, dtype):
        d = runit(4, 8)
        assert_close(
            F.softsign(nml(d, dtype)), TF.softsign(tch(d, dtype)), dtype
        )


class TestActivationBackward:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_relu(self, dtype):
        d = runit(4, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(F.relu(xn))
        bwd(TF.relu(xt))
        assert_grad(xn.grad, xt.grad, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_leaky_relu(self, dtype):
        d = runit(4, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(F.leaky_relu(xn, 0.1))
        bwd(TF.leaky_relu(xt, 0.1))
        assert_grad(xn.grad, xt.grad, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_sigmoid(self, dtype):
        d = runit(4, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(F.sigmoid(xn))
        bwd(torch.sigmoid(xt))
        assert_grad(xn.grad, xt.grad, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_tanh(self, dtype):
        d = runit(4, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(F.tanh(xn))
        bwd(torch.tanh(xt))
        assert_grad(xn.grad, xt.grad, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_softmax(self, dtype):
        d = runit(2, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(F.softmax(xn, dim=-1))
        bwd(TF.softmax(xt, dim=-1))
        assert_grad(xn.grad, xt.grad, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_gelu(self, dtype):
        d = runit(4, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(F.gelu(xn))
        bwd(TF.gelu(xt))
        assert_grad(xn.grad, xt.grad, dtype, tol=ATOL_LOOSE[dtype])

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_silu(self, dtype):
        d = runit(4, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(F.silu(xn))
        bwd(TF.silu(xt))
        assert_grad(xn.grad, xt.grad, dtype)


###############################################################################
# ATTENTION
###############################################################################


class TestAttention:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_sdpa_basic(self, dtype):
        B, H, S, D = 2, 4, 8, 16
        q = rdata(B, H, S, D)
        k = rdata(B, H, S, D)
        v = rdata(B, H, S, D)
        out_n, _ = F.scaled_dot_product_attention(
            nml(q, dtype), nml(k, dtype), nml(v, dtype), training=False
        )
        out_t = TF.scaled_dot_product_attention(
            tch(q, dtype), tch(k, dtype), tch(v, dtype)
        )
        assert_close(out_n, out_t, dtype, tol=ATOL_LOOSE[dtype])

    def test_sdpa_causal(self):
        dtype = nml_typing.float32
        B, H, S, D = 1, 2, 6, 8
        q = rdata(B, H, S, D)
        k = rdata(B, H, S, D)
        v = rdata(B, H, S, D)
        out_n, _ = F.scaled_dot_product_attention(
            nml(q, dtype),
            nml(k, dtype),
            nml(v, dtype),
            is_causal=True,
            training=False,
        )
        out_t = TF.scaled_dot_product_attention(
            tch(q, dtype), tch(k, dtype), tch(v, dtype), is_causal=True
        )
        assert_close(out_n, out_t, dtype, tol=ATOL_LOOSE[dtype])


###############################################################################
# COMBINATION
###############################################################################


class TestCombinationForward:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_cat(self, dtype):
        a, b, c = rdata(2, 4), rdata(3, 4), rdata(1, 4)
        nml_t = [nml(x, dtype) for x in [a, b, c]]
        tch_t = [tch(x, dtype) for x in [a, b, c]]
        assert_close(F.cat(nml_t, dim=0), torch.cat(tch_t, dim=0), dtype)
        a, b = rdata(2, 4), rdata(2, 6)
        assert_close(
            F.cat([nml(a, dtype), nml(b, dtype)], dim=1),
            torch.cat([tch(a, dtype), tch(b, dtype)], dim=1),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_concatenate(self, dtype):
        a, b = rdata(2, 4), rdata(3, 4)
        assert_close(
            F.concatenate([nml(a, dtype), nml(b, dtype)], dim=0),
            torch.cat([tch(a, dtype), tch(b, dtype)], dim=0),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_stack(self, dtype):
        a, b, c = rdata(4, 8), rdata(4, 8), rdata(4, 8)
        nml_t = [nml(x, dtype) for x in [a, b, c]]
        tch_t = [tch(x, dtype) for x in [a, b, c]]
        assert_close(F.stack(nml_t, dim=0), torch.stack(tch_t, dim=0), dtype)
        assert_close(F.stack(nml_t, dim=1), torch.stack(tch_t, dim=1), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_select(self, dtype):
        d = rdata(4, 8)
        xn, xt = nml(d, dtype), tch(d, dtype)
        assert_close(F.select(xn, 0, 1), xt.select(0, 1), dtype)
        assert_close(F.select(xn, 1, 3), xt.select(1, 3), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_unstack(self, dtype):
        d = rdata(3, 4, 8)
        nparts = F.unstack(nml(d, dtype), dim=0)
        tparts = tch(d, dtype).unbind(dim=0)
        for n, t in zip(nparts, tparts):
            assert_close(n, t, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_unbind(self, dtype):
        d = rdata(3, 4, 8)
        nparts = F.unbind(nml(d, dtype), dim=1)
        tparts = tch(d, dtype).unbind(dim=1)
        for n, t in zip(nparts, tparts):
            assert_close(n, t, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_split(self, dtype):
        d = rdata(8, 4)
        for sz in [2, 3]:
            nparts = F.split(nml(d, dtype), sz, dim=0)
            tparts = tch(d, dtype).split(sz, dim=0)
            for n, t in zip(nparts, tparts):
                assert_close(n, t, dtype, label=f'size={sz}')

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_chunk(self, dtype):
        d = rdata(8, 4)
        nparts = F.chunk(nml(d, dtype), 4, dim=0)
        tparts = tch(d, dtype).chunk(4, dim=0)
        for n, t in zip(nparts, tparts):
            assert_close(n, t, dtype)


class TestCombinationBackward:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_cat(self, dtype):
        a, b = rdata(2, 4), rdata(3, 4)
        an, at = nml(a, dtype, True), tch(a, dtype, True)
        bn, bt = nml(b, dtype, True), tch(b, dtype, True)
        bwd(F.cat([an, bn], dim=0))
        bwd(torch.cat([at, bt], dim=0))
        assert_grad(an.grad, at.grad, dtype, label='cat a')
        assert_grad(bn.grad, bt.grad, dtype, label='cat b')

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_stack(self, dtype):
        a, b = rdata(4, 8), rdata(4, 8)
        an, at = nml(a, dtype, True), tch(a, dtype, True)
        bn, bt = nml(b, dtype, True), tch(b, dtype, True)
        bwd(F.stack([an, bn], dim=0))
        bwd(torch.stack([at, bt], dim=0))
        assert_grad(an.grad, at.grad, dtype, label='stack a')
        assert_grad(bn.grad, bt.grad, dtype, label='stack b')


###############################################################################
# DROPOUT
###############################################################################


class TestDropout:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_dropout_eval(self, dtype):
        d = rdata(4, 8)
        out = F.dropout(nml(d, dtype), p=0.5, training=False)
        assert_close(out, tch(d, dtype), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_dropout_p0(self, dtype):
        d = rdata(4, 8)
        out = F.dropout(nml(d, dtype), p=0.0, training=True)
        assert_close(out, tch(d, dtype), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_dropout_stochastic_shape(self, dtype):
        d = rdata(4, 8)
        out = F.dropout(nml(d, dtype), p=0.5, training=True)
        assert to_np(out).shape == d.shape

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_dropout1d_eval(self, dtype):
        d = rdata(2, 4, 8)
        out = F.dropout1d(nml(d, dtype), p=0.0, training=True)
        assert_close(out, tch(d, dtype), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_dropout2d_eval(self, dtype):
        d = rdata(2, 4, 8, 8)
        out = F.dropout2d(nml(d, dtype), p=0.5, training=False)
        assert_close(out, tch(d, dtype), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_dropout3d_eval(self, dtype):
        d = rdata(2, 4, 4, 4, 4)
        out = F.dropout3d(nml(d, dtype), p=0.5, training=False)
        assert_close(out, tch(d, dtype), dtype)


###############################################################################
# INDEXING
###############################################################################


class TestIndexingForward:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_gather(self, dtype):
        d = rdata(4, 8)
        ix = np.random.randint(0, 4, (4, 8)).astype(np.int32)
        assert_close(
            F.gather(nml(d, dtype), 0, idx_nml(ix)),
            tch(d, dtype).gather(0, idx_tch(ix)),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_scatter(self, dtype):
        src = rdata(4, 8)
        ix = np.random.randint(0, 4, (4, 8)).astype(np.int32)
        base = np.zeros((4, 8), dtype=np.float32)
        assert_close(
            F.scatter(nml(base, dtype), 0, idx_nml(ix), nml(src, dtype)),
            tch(base, dtype).scatter(0, idx_tch(ix), tch(src, dtype)),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_scatter_add(self, dtype):
        src = rdata(4, 8)
        ix = np.random.randint(0, 4, (4, 8)).astype(np.int32)
        base = np.zeros((4, 8), dtype=np.float32)
        assert_close(
            F.scatter_add(nml(base, dtype), 0, idx_nml(ix), nml(src, dtype)),
            tch(base, dtype).scatter_add(0, idx_tch(ix), tch(src, dtype)),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_where(self, dtype):
        a = runit(4, 8)
        b = runit(4, 8)
        cond = (np.random.rand(4, 8) > 0.5).astype(np.float32)
        cond_n = nectarml.Tensor(cond).to(DEVICE, dtype=nml_typing.float32)
        cond_t = torch.tensor(cond.astype(bool), device=DEVICE)
        assert_close(
            F.where(cond_n, nml(a, dtype), nml(b, dtype)),
            torch.where(cond_t, tch(a, dtype), tch(b, dtype)),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_masked_fill(self, dtype):
        d = rdata(4, 8)
        mask = (np.random.rand(4, 8) > 0.5).astype(np.float32)
        mn = nectarml.Tensor(mask).to(DEVICE, dtype=nml_typing.float32)
        mt = torch.tensor(mask.astype(bool), device=DEVICE)
        assert_close(
            F.masked_fill(nml(d, dtype), mn, -1.0),
            tch(d, dtype).masked_fill(mt, -1.0),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_index_select(self, dtype):
        d = rdata(6, 8)
        ix = np.array([0, 2, 4], dtype=np.int32)
        assert_close(
            F.index_select(nml(d, dtype), 0, idx_nml(ix)),
            tch(d, dtype).index_select(0, idx_tch(ix)),
            dtype,
        )


###############################################################################
# INTERPOLATION
###############################################################################


class TestInterpolation:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_upsample_nearest_2x(self, dtype):
        d = rdata(1, 3, 4, 4)
        assert_close(
            F.upsample(nml(d, dtype), scale_factor=2, mode='nearest'),
            TF.interpolate(tch(d, dtype), scale_factor=2, mode='nearest'),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_upsample_nearest_size(self, dtype):
        d = rdata(1, 1, 3, 3)
        assert_close(
            F.upsample(nml(d, dtype), size=(6, 6), mode='nearest'),
            TF.interpolate(tch(d, dtype), size=(6, 6), mode='nearest'),
            dtype,
        )

    @pytest.mark.skip(
        reason=(
            'Bilinear coordinate mapping differs from PyTorch (~0.21). '
            'Nearest-neighbor matches correctly. Fix requires aligning '
            'grid sampling convention with PyTorch align_corners=False.'
        )
    )
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_upsample_bilinear(self, dtype):
        d = rdata(1, 3, 4, 4)
        out_n = F.upsample(
            nml(d, dtype), size=(8, 8), mode='bilinear', align_corners=False
        )
        out_t = TF.interpolate(
            tch(d, dtype).float(),
            size=(8, 8),
            mode='bilinear',
            align_corners=False,
        )
        assert_close(out_n, out_t, dtype, tol=ATOL_LOOSE[dtype])


###############################################################################
# LOSS FUNCTIONS
###############################################################################


class TestLossForward:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize('reduction', REDUCTIONS)
    def test_l1_loss(self, dtype, reduction):
        inp, tgt = rpos(4, 8), rpos(4, 8)
        assert_close(
            F.l1_loss(nml(inp, dtype), nml(tgt, dtype), reduction=reduction),
            TF.l1_loss(tch(inp, dtype), tch(tgt, dtype), reduction=reduction),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize('reduction', REDUCTIONS)
    def test_mae_loss(self, dtype, reduction):
        inp, tgt = rpos(4, 8), rpos(4, 8)
        assert_close(
            F.mae_loss(nml(inp, dtype), nml(tgt, dtype), reduction=reduction),
            TF.l1_loss(tch(inp, dtype), tch(tgt, dtype), reduction=reduction),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize('reduction', REDUCTIONS)
    def test_mse_loss(self, dtype, reduction):
        inp, tgt = rpos(4, 8), rpos(4, 8)
        assert_close(
            F.mse_loss(nml(inp, dtype), nml(tgt, dtype), reduction=reduction),
            TF.mse_loss(tch(inp, dtype), tch(tgt, dtype), reduction=reduction),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize('reduction', REDUCTIONS)
    def test_huber_loss(self, dtype, reduction):
        inp, tgt = runit(4, 8), runit(4, 8)
        assert_close(
            F.huber_loss(
                nml(inp, dtype),
                nml(tgt, dtype),
                delta=1.0,
                reduction=reduction,
            ),
            TF.huber_loss(
                tch(inp, dtype),
                tch(tgt, dtype),
                delta=1.0,
                reduction=reduction,
            ),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize('reduction', REDUCTIONS)
    def test_bce_loss(self, dtype, reduction):
        inp = np.random.uniform(0.05, 0.95, (4, 8)).astype(np.float32)
        tgt = np.random.uniform(0.05, 0.95, (4, 8)).astype(np.float32)
        assert_close(
            F.bce_loss(nml(inp, dtype), nml(tgt, dtype), reduction=reduction),
            TF.binary_cross_entropy(
                tch(inp, dtype), tch(tgt, dtype), reduction=reduction
            ),
            dtype,
            tol=ATOL_LOOSE[dtype],
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize('reduction', REDUCTIONS)
    def test_bce_with_logits_loss(self, dtype, reduction):
        inp = runit(4, 8)
        tgt = np.random.uniform(0.0, 1.0, (4, 8)).astype(np.float32)
        assert_close(
            F.bce_with_logits_loss(
                nml(inp, dtype), nml(tgt, dtype), reduction=reduction
            ),
            TF.binary_cross_entropy_with_logits(
                tch(inp, dtype), tch(tgt, dtype), reduction=reduction
            ),
            dtype,
            tol=ATOL_LOOSE[dtype],
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize('reduction', REDUCTIONS)
    def test_kl_divergence_loss(self, dtype, reduction):
        log_inp = np.log(np.random.dirichlet(np.ones(8), 4) + 1e-8).astype(
            np.float32
        )
        tgt = np.random.dirichlet(np.ones(8), 4).astype(np.float32)
        assert_close(
            F.kl_divergence_loss(
                nml(log_inp, dtype), nml(tgt, dtype), reduction=reduction
            ),
            TF.kl_div(
                tch(log_inp, dtype), tch(tgt, dtype), reduction=reduction
            ),
            dtype,
            tol=ATOL_LOOSE[dtype],
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize('reduction', ['mean', 'sum'])
    def test_triplet_margin_loss(self, dtype, reduction):
        a = np.random.uniform(0.0, 0.3, (4, 8)).astype(np.float32)
        p = np.random.uniform(0.0, 0.3, (4, 8)).astype(
            np.float32
        )
        n = np.random.uniform(0.7, 1.0, (4, 8)).astype(
            np.float32
        )
        assert_close(
            F.triplet_margin_loss(
                nml(a, dtype),
                nml(p, dtype),
                nml(n, dtype),
                margin=1.0,
                reduction=reduction,
            ),
            TF.triplet_margin_loss(
                tch(a, dtype),
                tch(p, dtype),
                tch(n, dtype),
                margin=1.0,
                reduction=reduction,
            ),
            dtype,
            tol=ATOL_LOOSE[dtype],
        )


class TestLossBackward:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_l1_loss(self, dtype):
        inp, tgt = rpos(4, 8), rpos(4, 8)
        xn, xt = nml(inp, dtype, True), tch(inp, dtype, True)
        bwd(F.l1_loss(xn, nml(tgt, dtype)))
        bwd(TF.l1_loss(xt, tch(tgt, dtype)))
        assert_grad(xn.grad, xt.grad, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_mse_loss(self, dtype):
        inp, tgt = rpos(4, 8), rpos(4, 8)
        xn, xt = nml(inp, dtype, True), tch(inp, dtype, True)
        bwd(F.mse_loss(xn, nml(tgt, dtype)))
        bwd(TF.mse_loss(xt, tch(tgt, dtype)))
        assert_grad(xn.grad, xt.grad, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_bce_loss(self, dtype):
        inp = np.random.uniform(0.05, 0.95, (4, 8)).astype(np.float32)
        tgt = np.random.uniform(0.05, 0.95, (4, 8)).astype(np.float32)
        xn, xt = nml(inp, dtype, True), tch(inp, dtype, True)
        bwd(F.bce_loss(xn, nml(tgt, dtype)))
        bwd(TF.binary_cross_entropy(xt, tch(tgt, dtype)))
        assert_grad(xn.grad, xt.grad, dtype, tol=ATOL_LOOSE[dtype])

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_huber_loss(self, dtype):
        inp, tgt = runit(4, 8), runit(4, 8)
        xn, xt = nml(inp, dtype, True), tch(inp, dtype, True)
        bwd(F.huber_loss(xn, nml(tgt, dtype), delta=1.0))
        bwd(TF.huber_loss(xt, tch(tgt, dtype), delta=1.0))
        assert_grad(xn.grad, xt.grad, dtype)


###############################################################################
# MATH FUNCTIONALS
###############################################################################


class TestMathFunctionals:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_add(self, dtype):
        a, b = rdata(4, 8), rdata(4, 8)
        assert_close(
            F.add(nml(a, dtype), nml(b, dtype)),
            tch(a, dtype) + tch(b, dtype),
            dtype,
        )
        assert_close(F.add(nml(a, dtype), 2.0), tch(a, dtype) + 2.0, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_subtract(self, dtype):
        a, b = rdata(4, 8), rdata(4, 8)
        assert_close(
            F.subtract(nml(a, dtype), nml(b, dtype)),
            tch(a, dtype) - tch(b, dtype),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_multiply(self, dtype):
        a, b = rdata(4, 8), rdata(4, 8)
        assert_close(
            F.multiply(nml(a, dtype), nml(b, dtype)),
            tch(a, dtype) * tch(b, dtype),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_pow(self, dtype):
        d = rpos(4, 8)
        assert_close(F.pow(nml(d, dtype), 2), tch(d, dtype) ** 2, dtype)
        assert_close(F.pow(nml(d, dtype), 0.5), tch(d, dtype) ** 0.5, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_matmul(self, dtype):
        a = np.random.uniform(0.1, 0.5, (4, 8)).astype(np.float32)
        b = np.random.uniform(0.1, 0.5, (8, 4)).astype(np.float32)
        assert_close(
            F.matmul(nml(a, dtype), nml(b, dtype)),
            tch(a, dtype) @ tch(b, dtype),
            dtype,
            tol=ATOL_LOOSE[dtype],
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_negate(self, dtype):
        d = rdata(4, 8)
        assert_close(F.negate(nml(d, dtype)), -tch(d, dtype), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_abs(self, dtype):
        d = runit(4, 8)
        assert_close(F.abs(nml(d, dtype)), tch(d, dtype).abs(), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_exp(self, dtype):
        d = np.random.uniform(-1.0, 1.0, (4, 8)).astype(np.float32)
        assert_close(F.exp(nml(d, dtype)), tch(d, dtype).exp(), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_sqrt(self, dtype):
        d = rpos(4, 8)
        assert_close(F.sqrt(nml(d, dtype)), tch(d, dtype).sqrt(), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_sigmoid_math(self, dtype):
        d = runit(4, 8)
        assert_close(
            F.sigmoid(nml(d, dtype)), torch.sigmoid(tch(d, dtype)), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_clamp(self, dtype):
        d = rdata(4, 8)
        assert_close(
            F.clamp(nml(d, dtype), 0.2, 0.8),
            tch(d, dtype).clamp(0.2, 0.8),
            dtype,
        )


###############################################################################
# NORMALIZATION
###############################################################################


class TestNormalizationForward:
    def _expected(self, x_np, axes, C, spatial_shape, eps=1e-5):
        mean = x_np.mean(axis=axes, keepdims=True)
        var = x_np.var(axis=axes, keepdims=True)
        return ((x_np - mean) / np.sqrt(var + eps)).astype(np.float32)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_batch_norm2d(self, dtype):
        d = rdata(4, 8, 6, 6)
        C = 8
        gn = nml(np.ones(C, dtype=np.float32), dtype)
        bn = nml(np.zeros(C, dtype=np.float32), dtype)

        out_n, _ = F.batch_norm2d(nml(d, dtype), gamma=gn, beta=bn)

        xt = tch(d, dtype)
        gx = torch.ones(C, dtype=TMAP[dtype], device=DEVICE)
        bx = torch.zeros(C, dtype=TMAP[dtype], device=DEVICE)
        out_t = torch.nn.functional.batch_norm(
            xt, None, None, weight=gx, bias=bx, training=True, eps=1e-5
        )

        assert_close(out_n, out_t, dtype, tol=ATOL_LOOSE[dtype])

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_instance_norm2d(self, dtype):
        d = rdata(2, 4, 8, 8)
        C = 4
        gn = nml(np.ones(C, dtype=np.float32), dtype)
        bn = nml(np.zeros(C, dtype=np.float32), dtype)

        out_n, _ = F.instance_norm2d(nml(d, dtype), gamma=gn, beta=bn)

        xt = tch(d, dtype)
        gx = torch.ones(C, dtype=TMAP[dtype], device=DEVICE)
        bx = torch.zeros(C, dtype=TMAP[dtype], device=DEVICE)
        out_t = torch.nn.functional.instance_norm(
            xt, weight=gx, bias=bx, eps=1e-5
        )

        assert_close(out_n, out_t, dtype, tol=ATOL_LOOSE[dtype])

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_group_norm(self, dtype):
        d = rdata(2, 8, 4, 4)
        C, G = 8, 4
        gn = nml(np.ones(C, dtype=np.float32), dtype)
        bn = nml(np.zeros(C, dtype=np.float32), dtype)
        gt = tch(np.ones(C, dtype=np.float32), nml_typing.float32)
        bt = tch(np.zeros(C, dtype=np.float32), nml_typing.float32)
        out_n, _ = F.group_norm(nml(d, dtype), G, gamma=gn, beta=bn)
        out_t = TF.group_norm(tch(d, dtype).float(), G, weight=gt, bias=bt)
        assert_close(out_n, out_t, dtype, tol=ATOL_LOOSE[dtype])

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_layer_norm(self, dtype):
        d = rdata(2, 4, 8)
        ns = [4, 8]
        gn = nml(np.ones((4, 8), dtype=np.float32), dtype)
        bn = nml(np.zeros((4, 8), dtype=np.float32), dtype)
        gt = tch(np.ones((4, 8), dtype=np.float32), dtype)
        bt = tch(np.zeros((4, 8), dtype=np.float32), dtype)
        out_n = F.layer_norm(nml(d, dtype), ns, gamma=gn, beta=bn)
        out_t = TF.layer_norm(tch(d, dtype), ns, weight=gt, bias=bt)
        assert_close(out_n, out_t, dtype, tol=ATOL_LOOSE[dtype])


class TestNormalizationBackward:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_layer_norm(self, dtype):
        d = rdata(2, 4, 8)
        ns = [4, 8]
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        gn = nml(np.ones((4, 8), dtype=np.float32), dtype, True)
        gt = tch(np.ones((4, 8), dtype=np.float32), dtype, True)
        bn = nml(np.zeros((4, 8), dtype=np.float32), dtype, True)
        bt = tch(np.zeros((4, 8), dtype=np.float32), dtype, True)
        bwd(F.layer_norm(xn, ns, gamma=gn, beta=bn))
        bwd(TF.layer_norm(xt, ns, weight=gt, bias=bt))
        assert_grad(xn.grad, xt.grad, dtype, tol=5e-3, label='ln dx')
        assert_grad(gn.grad, gt.grad, dtype, tol=5e-3, label='ln dgamma')


###############################################################################
# PADDING
###############################################################################


class TestPaddingForward:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_constant_1d(self, dtype):
        d = rdata(2, 4, 8)
        assert_close(
            F.pad(nml(d, dtype), pad=(1, 2), mode='constant', value=0.0),
            TF.pad(tch(d, dtype), pad=(1, 2), mode='constant', value=0.0),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_constant_2d(self, dtype):
        d = rdata(2, 4, 6, 6)
        assert_close(
            F.pad(nml(d, dtype), pad=(1, 1, 2, 2), mode='constant', value=0.0),
            TF.pad(
                tch(d, dtype), pad=(1, 1, 2, 2), mode='constant', value=0.0
            ),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_reflect(self, dtype):
        d = rdata(2, 4, 8, 8)
        assert_close(
            F.pad(nml(d, dtype), pad=(1, 1, 1, 1), mode='reflect'),
            TF.pad(tch(d, dtype), pad=(1, 1, 1, 1), mode='reflect'),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_replicate(self, dtype):
        d = rdata(2, 4, 8, 8)
        assert_close(
            F.pad(nml(d, dtype), pad=(2, 2, 2, 2), mode='replicate'),
            TF.pad(tch(d, dtype), pad=(2, 2, 2, 2), mode='replicate'),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_pad_backward(self, dtype):
        d = rdata(2, 4, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(F.pad(xn, pad=(1, 2), mode='constant', value=0.0))
        bwd(TF.pad(xt, pad=(1, 2), mode='constant', value=0.0))
        assert_grad(xn.grad, xt.grad, dtype)


###############################################################################
# POOLING
###############################################################################


class TestPoolingForward:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_avg_pool1d(self, dtype):
        d = rdata(2, 4, 16)
        for k, s in [(2, 2), (3, 1)]:
            assert_close(
                F.avg_pool1d(nml(d, dtype), kernel_size=k, stride=s),
                TF.avg_pool1d(tch(d, dtype), kernel_size=k, stride=s),
                dtype,
                label=f'k={k},s={s}',
            )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_avg_pool2d(self, dtype):
        d = rdata(2, 4, 16, 16)
        for k, s in [(2, 2), (3, 1)]:
            assert_close(
                F.avg_pool2d(nml(d, dtype), kernel_size=k, stride=s),
                TF.avg_pool2d(tch(d, dtype), kernel_size=k, stride=s),
                dtype,
                label=f'k={k},s={s}',
            )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_avg_pool2d_padding(self, dtype):
        d = rdata(2, 4, 8, 8)
        assert_close(
            F.avg_pool2d(nml(d, dtype), kernel_size=3, stride=1, padding=1),
            TF.avg_pool2d(tch(d, dtype), kernel_size=3, stride=1, padding=1),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_max_pool1d(self, dtype):
        d = rdata(2, 4, 16)
        assert_close(
            F.max_pool1d(nml(d, dtype), kernel_size=2, stride=2),
            TF.max_pool1d(tch(d, dtype), kernel_size=2, stride=2),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_max_pool2d(self, dtype):
        d = rdata(2, 4, 16, 16)
        for k, s in [(2, 2), (3, 1)]:
            assert_close(
                F.max_pool2d(nml(d, dtype), kernel_size=k, stride=s),
                TF.max_pool2d(tch(d, dtype), kernel_size=k, stride=s),
                dtype,
                label=f'k={k},s={s}',
            )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_max_pool2d_padding(self, dtype):
        d = rdata(2, 4, 8, 8)
        assert_close(
            F.max_pool2d(nml(d, dtype), kernel_size=3, stride=1, padding=1),
            TF.max_pool2d(tch(d, dtype), kernel_size=3, stride=1, padding=1),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_avg_pool3d(self, dtype):
        d = rdata(2, 4, 8, 8, 8)
        assert_close(
            F.avg_pool3d(nml(d, dtype), kernel_size=2, stride=2),
            TF.avg_pool3d(tch(d, dtype), kernel_size=2, stride=2),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_max_pool3d(self, dtype):
        d = rdata(2, 4, 8, 8, 8)
        assert_close(
            F.max_pool3d(nml(d, dtype), kernel_size=2, stride=2),
            TF.max_pool3d(tch(d, dtype), kernel_size=2, stride=2),
            dtype,
        )


class TestPoolingBackward:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_avg_pool2d(self, dtype):
        d = rdata(2, 4, 8, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(F.avg_pool2d(xn, kernel_size=2, stride=2))
        bwd(TF.avg_pool2d(xt, kernel_size=2, stride=2))
        assert_grad(xn.grad, xt.grad, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_max_pool2d(self, dtype):
        d = rdata(2, 4, 8, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(F.max_pool2d(xn, kernel_size=2, stride=2))
        bwd(TF.max_pool2d(xt, kernel_size=2, stride=2))
        assert_grad(xn.grad, xt.grad, dtype)


###############################################################################
# REDUCTIONS
###############################################################################


class TestReductionFunctionalsForward:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_mean(self, dtype):
        d = rdata(2, 4, 8)
        assert_close(F.mean(nml(d, dtype)), tch(d, dtype).mean(), dtype)
        assert_close(
            F.mean(nml(d, dtype), dim=0), tch(d, dtype).mean(dim=0), dtype
        )
        assert_close(
            F.mean(nml(d, dtype), dim=-1), tch(d, dtype).mean(dim=-1), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_sum(self, dtype):
        d = rdata(2, 4, 8)
        assert_close(F.sum(nml(d, dtype)), tch(d, dtype).sum(), dtype)
        assert_close(
            F.sum(nml(d, dtype), dim=1), tch(d, dtype).sum(dim=1), dtype
        )
        assert_close(
            F.sum(nml(d, dtype), dim=-1), tch(d, dtype).sum(dim=-1), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_prod(self, dtype):
        d = np.random.uniform(0.95, 1.05, (2, 4, 4)).astype(np.float32)
        assert_close(
            F.prod(nml(d, dtype), dim=0),
            tch(d, dtype).prod(dim=0),
            dtype,
            tol=ATOL_LOOSE[dtype],
        )
        assert_close(
            F.prod(nml(d, dtype), dim=1),
            tch(d, dtype).prod(dim=1),
            dtype,
            tol=ATOL_LOOSE[dtype],
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_min(self, dtype):
        d = rdata(4, 8)
        xn, xt = nml(d, dtype), tch(d, dtype)
        assert_close(F.min(xn), xt.min(), dtype)
        assert_close(F.min(xn, dim=0).values, xt.min(dim=0).values, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_max(self, dtype):
        d = rdata(4, 8)
        xn, xt = nml(d, dtype), tch(d, dtype)
        assert_close(F.max(xn), xt.max(), dtype)
        assert_close(F.max(xn, dim=0).values, xt.max(dim=0).values, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_amin_amax(self, dtype):
        d = rdata(2, 4, 8)
        assert_close(
            F.amin(nml(d, dtype), dim=(0, 1)),
            tch(d, dtype).amin(dim=(0, 1)),
            dtype,
        )
        assert_close(
            F.amax(nml(d, dtype), dim=-1), tch(d, dtype).amax(dim=-1), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_argmin_argmax(self, dtype):
        d = rdata(4, 8)
        assert_close(F.argmin(nml(d, dtype)), tch(d, dtype).argmin(), dtype)
        assert_close(F.argmax(nml(d, dtype)), tch(d, dtype).argmax(), dtype)
        assert_close(
            F.argmin(nml(d, dtype), dim=0), tch(d, dtype).argmin(dim=0), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_norm(self, dtype):
        d = rpos(4, 8)
        assert_close(
            F.norm(nml(d, dtype), 'fro'), tch(d, dtype).norm('fro'), dtype
        )
        assert_close(F.norm(nml(d, dtype), 'l1'), tch(d, dtype).norm(1), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_cumsum(self, dtype):
        d = rdata(4, 8)
        assert_close(
            F.cumsum(nml(d, dtype), dim=0), tch(d, dtype).cumsum(dim=0), dtype
        )
        assert_close(
            F.cumsum(nml(d, dtype), dim=1), tch(d, dtype).cumsum(dim=1), dtype
        )


###############################################################################
# SHAPES
###############################################################################


class TestShapeFunctionals:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_reshape(self, dtype):
        d = rdata(2, 4, 8)
        assert_close(
            F.reshape(nml(d, dtype), (8, 8)),
            tch(d, dtype).reshape(8, 8),
            dtype,
        )
        assert_close(
            F.reshape(nml(d, dtype), (-1,)), tch(d, dtype).reshape(-1), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_view(self, dtype):
        d = rdata(2, 4, 8)
        assert_close(
            F.view(nml(d, dtype), (8, 8)), tch(d, dtype).view(8, 8), dtype
        )
        assert_close(
            F.view(nml(d, dtype), (-1, 8)), tch(d, dtype).view(-1, 8), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_flatten(self, dtype):
        d = rdata(2, 4, 8)
        assert_close(F.flatten(nml(d, dtype)), tch(d, dtype).flatten(), dtype)
        assert_close(
            F.flatten(nml(d, dtype), 1, 2), tch(d, dtype).flatten(1, 2), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_squeeze(self, dtype):
        d = rdata(1, 4, 1, 8)
        xn, xt = nml(d, dtype), tch(d, dtype)
        assert_close(F.squeeze(xn, dim=None), xt.squeeze(), dtype)
        assert_close(F.squeeze(xn, dim=0), xt.squeeze(0), dtype)
        assert_close(F.squeeze(xn, dim=2), xt.squeeze(2), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_unsqueeze(self, dtype):
        d = rdata(4, 8)
        xn, xt = nml(d, dtype), tch(d, dtype)
        assert_close(F.unsqueeze(xn, dim=0), xt.unsqueeze(0), dtype)
        assert_close(F.unsqueeze(xn, dim=-1), xt.unsqueeze(-1), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_transpose(self, dtype):
        d = rdata(4, 8)
        assert_close(
            F.transpose(nml(d, dtype), 0, 1),
            tch(d, dtype).transpose(0, 1),
            dtype,
        )
        d3 = rdata(2, 4, 8)
        assert_close(
            F.transpose(nml(d3, dtype), 0, 2),
            tch(d3, dtype).transpose(0, 2),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_swapdims(self, dtype):
        d = rdata(2, 4, 8)
        assert_close(
            F.swapdims(nml(d, dtype), 0, 2),
            tch(d, dtype).swapdims(0, 2),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_permute(self, dtype):
        d = rdata(2, 4, 8)
        assert_close(
            F.permute(nml(d, dtype), (2, 0, 1)),
            tch(d, dtype).permute(2, 0, 1),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_expand(self, dtype):
        d = rdata(1, 4, 1, 8)
        assert_close(
            F.expand(nml(d, dtype), (2, 4, 3, 8)),
            tch(d, dtype).expand(2, 4, 3, 8),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_broadcast_to(self, dtype):
        d = rdata(1, 8)
        assert_close(
            F.broadcast_to(nml(d, dtype), (4, 8)),
            tch(d, dtype).broadcast_to(4, 8),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_unfold(self, dtype):
        d = rdata(4, 16)
        assert_close(
            F.unfold(nml(d, dtype), 1, 4, 2),
            tch(d, dtype).unfold(1, 4, 2),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_flip(self, dtype):
        d = rdata(4, 8)
        assert_close(F.flip(nml(d, dtype), 0), tch(d, dtype).flip([0]), dtype)
        assert_close(F.flip(nml(d, dtype), 1), tch(d, dtype).flip([1]), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_reshape_backward(self, dtype):
        d = rdata(2, 4, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(F.reshape(xn, (8, 8)))
        bwd(xt.reshape(8, 8))
        assert_grad(xn.grad, xt.grad, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_permute_backward(self, dtype):
        d = rdata(2, 4, 8)
        xn, xt = nml(d, dtype, True), tch(d, dtype, True)
        bwd(F.permute(xn, (2, 0, 1)))
        bwd(xt.permute(2, 0, 1))
        assert_grad(xn.grad, xt.grad, dtype)
