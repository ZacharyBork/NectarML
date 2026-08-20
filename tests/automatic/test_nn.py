import numpy as np
import pytest
import torch
import torch.nn as TNN
import nectarml
import nectarml.nn as nn
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
REDUCTIONS = ['none', 'mean', 'sum']


@pytest.fixture(autouse=True)
def fixed_seed():
    np.random.seed(42)
    torch.manual_seed(42)


###############################################################################
# Helpers
###############################################################################


def nml_t(data, dtype, grad=False):
    return nectarml.Tensor(data, requires_grad=grad).to(DEVICE, dtype=dtype)


def tch_t(data, dtype, grad=False):
    return torch.tensor(
        data, dtype=TMAP[dtype], device=DEVICE, requires_grad=grad
    )


def rdata(*shape):
    return np.random.uniform(0.1, 0.9, shape).astype(np.float32)


def runit(*shape):
    return np.random.uniform(-0.9, 0.9, shape).astype(np.float32)


def to_np(t):
    if isinstance(t, nectarml.Tensor):
        return t.detach().cpu().numpy().astype(np.float32)
    return t.detach().cpu().float().numpy()


def atol(dtype):
    return ATOL_F16 if dtype == nml_typing.float16 else ATOL_F32


def assert_close(a, b, dtype, tol=None, label=''):
    if tol is None:
        tol = atol(dtype)
    na, nb = to_np(a), to_np(b)
    assert (
        na.shape == nb.shape
    ), f'{label} shape mismatch: nml={na.shape} torch={nb.shape}'
    assert np.allclose(na, nb, atol=tol, equal_nan=False), (
        f'{label} max_diff={np.nanmax(np.abs(na-nb)):.6f} tol={tol}\n'
        f'  nml:   {na.flat[:8]}\n  torch: {nb.flat[:8]}'
    )


def assert_grad(ng, tg, dtype, tol=None, label=''):
    if tol is None:
        tol = ATOL_LOOSE[dtype]
    assert ng is not None, f'{label} nml grad is None'
    assert tg is not None, f'{label} torch grad is None'
    na, nb = to_np(ng), to_np(tg)
    assert (
        na.shape == nb.shape
    ), f'{label} grad shape: nml={na.shape} torch={nb.shape}'
    assert np.allclose(na, nb, atol=tol, equal_nan=False), (
        f'{label} grad max_diff={np.nanmax(np.abs(na-nb)):.6f}\n'
        f'  nml:   {na.flat[:8]}\n  torch: {nb.flat[:8]}'
    )


def bwd(out):
    out.sum().backward()


def sync_weights(nml_mod, tch_mod):
    pass


###############################################################################
# MODULE BASE: utility methods
###############################################################################


class TestModuleBase:
    def test_to_device(self):
        mod = nn.Linear(8, 4)
        mod.to('cuda')
        for name, p in mod.named_parameters():
            assert p.device == 'cuda', f'{name} not on cuda'

    def test_to_dtype(self):
        mod = nn.Linear(8, 4)
        mod.to('cuda', dtype=nml_typing.float16)
        for name, p in mod.named_parameters():
            assert p.dtype == nml_typing.float16, f'{name} not float16'

    def test_cuda_cpu_roundtrip(self):
        mod = nn.Linear(8, 4)
        mod.cuda()
        for p in mod.parameters():
            assert p.device == 'cuda'
        mod.cpu()
        for p in mod.parameters():
            assert p.device == 'cpu'

    def test_zero_grad(self):
        mod = nn.Linear(8, 4).to('cuda')
        x = nml_t(rdata(2, 8), nml_typing.float32, grad=True)
        bwd(mod(x))
        mod.zero_grad()
        for p in mod.parameters():
            assert p.grad is None or np.allclose(to_np(p.grad), 0)

    def test_train_eval(self):
        mod = nn.Dropout(p=0.5)
        mod.train()
        assert mod.training == True
        mod.eval()
        assert mod.training == False

    def test_parameters_list(self):
        mod = nn.Linear(8, 4)
        params = mod.named_parameters()
        assert len(params) >= 1
        names = [n for n, _ in params]
        assert any('weight' in n for n in names)

    def test_sequential(self):
        mod = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)).to(
            'cuda'
        )
        x = nml_t(rdata(2, 8), nml_typing.float32)
        out = mod(x)
        assert out.shape == nectarml.typing.Size([2, 4])

    def test_module_list(self):
        mods = nn.ModuleList([nn.Linear(8, 8), nn.Linear(8, 4)])
        mods.to('cuda')
        params = mods.parameters()
        assert len(params) >= 2

    def test_module_dict(self):
        mods = nn.ModuleDict({'fc1': nn.Linear(8, 8), 'fc2': nn.Linear(8, 4)})
        mods.to('cuda')
        params = mods.parameters()
        assert len(params) >= 2

    def test_identity(self):
        mod = nn.Identity().to('cuda')
        x = nml_t(rdata(2, 8), nml_typing.float32)
        out = mod(x)
        assert_close(out, x, nml_typing.float32)


###############################################################################
# LINEAR
###############################################################################


class TestLinear:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_forward(self, dtype):
        d = rdata(4, 8)
        w = rdata(16, 8)
        b = rdata(16)

        nmod = nn.Linear(8, 16, bias=True, dtype=dtype).to('cuda')
        tmod = TNN.Linear(8, 16, bias=True).to(DEVICE, TMAP[dtype])

        with torch.no_grad():
            tmod.weight.copy_(
                torch.tensor(to_np(nmod.weight), dtype=TMAP[dtype])
            )
            tmod.bias.copy_(torch.tensor(to_np(nmod.bias), dtype=TMAP[dtype]))

        xn = nml_t(d, dtype)
        xt = tch_t(d, dtype)
        assert_close(nmod(xn), tmod(xt), dtype, tol=ATOL_LOOSE[dtype])

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_no_bias(self, dtype):
        nmod = nn.Linear(8, 16, bias=False, dtype=dtype).to('cuda')
        params = dict(nmod.named_parameters())
        assert 'bias' not in params
        assert 'weight' in params

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_backward(self, dtype):
        nmod = nn.Linear(8, 4, dtype=dtype).to('cuda')
        tmod = TNN.Linear(8, 4).to(DEVICE, TMAP[dtype])
        with torch.no_grad():
            tmod.weight.copy_(
                torch.tensor(to_np(nmod.weight), dtype=TMAP[dtype])
            )
            tmod.bias.copy_(torch.tensor(to_np(nmod.bias), dtype=TMAP[dtype]))

        d = rdata(2, 8)
        xn = nml_t(d, dtype, True)
        xt = tch_t(d, dtype, True)
        bwd(nmod(xn))
        bwd(tmod(xt))
        assert_grad(xn.grad, xt.grad, dtype, label='linear dx')

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_batched(self, dtype):
        nmod = nn.Linear(8, 4, dtype=dtype).to('cuda')
        d = rdata(3, 5, 8)
        out = nmod(nml_t(d, dtype))
        assert out.shape == nectarml.typing.Size([3, 5, 4])


###############################################################################
# WEIGHT INITIALIZATION
###############################################################################


class TestWeightInit:
    def test_zeros_(self):
        t = nml_t(rdata(4, 8), nml_typing.float32)
        nn.init.zeros_(t)
        assert np.allclose(to_np(t), 0)

    def test_ones_(self):
        t = nml_t(rdata(4, 8), nml_typing.float32)
        nn.init.ones_(t)
        assert np.allclose(to_np(t), 1)

    def test_constant_(self):
        t = nml_t(rdata(4, 8), nml_typing.float32)
        nn.init.constant_(t, 3.14)
        assert np.allclose(to_np(t), 3.14)

    def test_uniform_(self):
        t = nml_t(rdata(4, 8), nml_typing.float32)
        nn.init.uniform_(t, -1.0, 1.0)
        vals = to_np(t)
        assert vals.min() >= -1.0 and vals.max() <= 1.0

    def test_normal_(self):
        t = nml_t(rdata(64, 64), nml_typing.float32)
        nn.init.normal_(t, mean=0.0, std=1.0)
        vals = to_np(t)
        assert abs(vals.mean()) < 0.2
        assert abs(vals.std() - 1.0) < 0.2

    def test_xavier_uniform_(self):
        t = nml_t(rdata(8, 16), nml_typing.float32)
        nn.init.xavier_uniform_(t)
        vals = to_np(t)
        bound = np.sqrt(6.0 / (8 + 16))
        assert vals.min() >= -bound - 0.01
        assert vals.max() <= bound + 0.01

    def test_kaiming_uniform_(self):
        t = nml_t(rdata(8, 16), nml_typing.float32)
        nn.init.kaiming_uniform_(t, nonlinearity='relu')
        vals = to_np(t)
        assert not np.allclose(vals, 0)

    def test_orthogonal_(self):
        t = nml_t(rdata(8, 8), nml_typing.float32)
        nn.init.orthogonal_(t)
        vals = to_np(t)
        product = vals @ vals.T
        assert np.allclose(product, np.eye(8), atol=1e-3)


###############################################################################
# ACTIVATION MODULES
###############################################################################


class TestActivationModules:

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_relu(self, dtype):
        d = runit(4, 8)
        mod = nn.ReLU().to('cuda')
        assert_close(mod(nml_t(d, dtype)), TNN.ReLU()(tch_t(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_leaky_relu(self, dtype):
        d = runit(4, 8)
        mod = nn.LeakyReLU(negative_slope=0.1).to('cuda')
        assert_close(
            mod(nml_t(d, dtype)), TNN.LeakyReLU(0.1)(tch_t(d, dtype)), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_elu(self, dtype):
        d = runit(4, 8)
        mod = nn.ELU(alpha=1.0).to('cuda')
        assert_close(mod(nml_t(d, dtype)), TNN.ELU()(tch_t(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_selu(self, dtype):
        d = runit(4, 8)
        mod = nn.SELU().to('cuda')
        assert_close(mod(nml_t(d, dtype)), TNN.SELU()(tch_t(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_sigmoid(self, dtype):
        d = runit(4, 8)
        mod = nn.Sigmoid().to('cuda')
        assert_close(
            mod(nml_t(d, dtype)), TNN.Sigmoid()(tch_t(d, dtype)), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_tanh(self, dtype):
        d = runit(4, 8)
        mod = nn.Tanh().to('cuda')
        assert_close(mod(nml_t(d, dtype)), TNN.Tanh()(tch_t(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_softmax(self, dtype):
        d = runit(2, 8)
        mod = nn.Softmax(dim=-1).to('cuda')
        assert_close(
            mod(nml_t(d, dtype)), TNN.Softmax(dim=-1)(tch_t(d, dtype)), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_log_softmax(self, dtype):
        d = runit(2, 8)
        mod = nn.LogSoftmax(dim=-1).to('cuda')
        assert_close(
            mod(nml_t(d, dtype)),
            TNN.LogSoftmax(dim=-1)(tch_t(d, dtype)),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_gelu(self, dtype):
        d = runit(4, 8)
        mod = nn.GeLU().to('cuda')
        assert_close(
            mod(nml_t(d, dtype)),
            TNN.GELU()(tch_t(d, dtype)),
            dtype,
            tol=ATOL_LOOSE[dtype],
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_silu(self, dtype):
        d = runit(4, 8)
        mod = nn.SiLU().to('cuda')
        assert_close(mod(nml_t(d, dtype)), TNN.SiLU()(tch_t(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_softplus(self, dtype):
        d = runit(4, 8)
        mod = nn.Softplus().to('cuda')
        assert_close(
            mod(nml_t(d, dtype)), TNN.Softplus()(tch_t(d, dtype)), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_mish(self, dtype):
        d = runit(4, 8)
        mod = nn.Mish().to('cuda')
        assert_close(mod(nml_t(d, dtype)), TNN.Mish()(tch_t(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_hardtanh(self, dtype):
        d = runit(4, 8)
        mod = nn.Hardtanh(-1.0, 1.0).to('cuda')
        assert_close(
            mod(nml_t(d, dtype)),
            TNN.Hardtanh(-1.0, 1.0)(tch_t(d, dtype)),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_hardsigmoid(self, dtype):
        d = runit(4, 8)
        mod = nn.Hardsigmoid().to('cuda')
        assert_close(
            mod(nml_t(d, dtype)), TNN.Hardsigmoid()(tch_t(d, dtype)), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_hardswish(self, dtype):
        d = runit(4, 8)
        mod = nn.Hardswish().to('cuda')
        assert_close(
            mod(nml_t(d, dtype)), TNN.Hardswish()(tch_t(d, dtype)), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_softsign(self, dtype):
        d = runit(4, 8)
        mod = nn.Softsign().to('cuda')
        assert_close(
            mod(nml_t(d, dtype)), TNN.Softsign()(tch_t(d, dtype)), dtype
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_relu_backward(self, dtype):
        d = runit(4, 8)
        mod = nn.ReLU().to('cuda')
        xn, xt = nml_t(d, dtype, True), tch_t(d, dtype, True)
        bwd(mod(xn))
        bwd(TNN.ReLU()(xt))
        assert_grad(xn.grad, xt.grad, dtype)


###############################################################################
# CONVOLUTION
###############################################################################


class TestConvolution:
    def _sync_conv(self, nmod, tmod, dtype):
        with torch.no_grad():
            tmod.weight.copy_(
                torch.tensor(to_np(nmod.weight), dtype=TMAP[dtype])
            )
            if nmod.bias is not None:
                tmod.bias.copy_(
                    torch.tensor(to_np(nmod.bias), dtype=TMAP[dtype])
                )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_conv1d_basic(self, dtype):
        nmod = nn.Conv1d(4, 8, kernel_size=3, padding=1, dtype=dtype).to(
            'cuda'
        )
        tmod = TNN.Conv1d(4, 8, 3, padding=1).to(DEVICE, TMAP[dtype])
        self._sync_conv(nmod, tmod, dtype)
        d = rdata(2, 4, 16)
        assert_close(
            nmod(nml_t(d, dtype)),
            tmod(tch_t(d, dtype)),
            dtype,
            tol=ATOL_LOOSE[dtype],
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_conv2d_basic(self, dtype):
        nmod = nn.Conv2d(3, 8, kernel_size=3, padding=1, dtype=dtype).to(
            'cuda'
        )
        tmod = TNN.Conv2d(3, 8, 3, padding=1).to(DEVICE, TMAP[dtype])
        self._sync_conv(nmod, tmod, dtype)
        d = rdata(2, 3, 16, 16)
        assert_close(
            nmod(nml_t(d, dtype)),
            tmod(tch_t(d, dtype)),
            dtype,
            tol=ATOL_LOOSE[dtype],
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_conv2d_stride(self, dtype):
        nmod = nn.Conv2d(
            3, 8, kernel_size=3, stride=2, padding=1, dtype=dtype
        ).to('cuda')
        tmod = TNN.Conv2d(3, 8, 3, stride=2, padding=1).to(DEVICE, TMAP[dtype])
        self._sync_conv(nmod, tmod, dtype)
        d = rdata(2, 3, 16, 16)
        assert_close(
            nmod(nml_t(d, dtype)),
            tmod(tch_t(d, dtype)),
            dtype,
            tol=ATOL_LOOSE[dtype],
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_conv2d_no_bias(self, dtype):
        nmod = nn.Conv2d(
            3, 8, kernel_size=3, padding=1, bias=False, dtype=dtype
        ).to('cuda')
        tmod = TNN.Conv2d(3, 8, 3, padding=1, bias=False).to(
            DEVICE, TMAP[dtype]
        )
        self._sync_conv(nmod, tmod, dtype)
        d = rdata(2, 3, 8, 8)
        assert_close(
            nmod(nml_t(d, dtype)),
            tmod(tch_t(d, dtype)),
            dtype,
            tol=ATOL_LOOSE[dtype],
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_conv2d_backward(self, dtype):
        if dtype == nml_typing.float16:
            pytest.skip(
                'float16 conv backward accumulates gradients in float16 '
                'causing overflow. Backward computed correctly in float32 mode'
            )

        nmod = nn.Conv2d(3, 4, kernel_size=3, padding=1, dtype=dtype).to(
            'cuda'
        )
        tmod = TNN.Conv2d(3, 4, 3, padding=1).to(DEVICE, TMAP[dtype])
        self._sync_conv(nmod, tmod, dtype)
        d = rdata(2, 3, 8, 8)
        xn = nml_t(d, dtype, True)
        xt = tch_t(d, dtype, True)
        bwd(nmod(xn))
        bwd(tmod(xt))
        assert_grad(
            xn.grad, xt.grad, dtype, tol=ATOL_LOOSE[dtype], label='conv2d dx'
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_conv_transpose2d(self, dtype):
        nmod = nn.ConvTranspose2d(
            8,
            4,
            kernel_size=3,
            stride=2,
            padding=1,
            output_padding=1,
            dtype=dtype,
        ).to('cuda')
        tmod = TNN.ConvTranspose2d(
            8, 4, 3, stride=2, padding=1, output_padding=1
        ).to(DEVICE, TMAP[dtype])
        self._sync_conv(nmod, tmod, dtype)
        d = rdata(2, 8, 8, 8)
        assert_close(
            nmod(nml_t(d, dtype)),
            tmod(tch_t(d, dtype)),
            dtype,
            tol=ATOL_LOOSE[dtype],
        )


###############################################################################
# DROPOUT
###############################################################################


class TestDropoutModules:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_dropout_eval(self, dtype):
        mod = nn.Dropout(p=0.5)
        mod.eval()
        mod.to('cuda')
        d = rdata(4, 8)
        out = mod(nml_t(d, dtype))
        assert_close(out, nml_t(d, dtype), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_dropout_train_shape(self, dtype):
        mod = nn.Dropout(p=0.5)
        mod.train()
        mod.to('cuda')
        d = rdata(4, 8)
        out = mod(nml_t(d, dtype))
        assert to_np(out).shape == d.shape

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_dropout2d_eval(self, dtype):
        mod = nn.Dropout2d(p=0.5)
        mod.eval()
        mod.to('cuda')
        d = rdata(2, 4, 8, 8)
        out = mod(nml_t(d, dtype))
        assert_close(out, nml_t(d, dtype), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_dropout1d_eval(self, dtype):
        mod = nn.Dropout1d(p=0.5)
        mod.eval()
        mod.to('cuda')
        d = rdata(2, 4, 8)
        out = mod(nml_t(d, dtype))
        assert_close(out, nml_t(d, dtype), dtype)


###############################################################################
# NORMALIZATION
###############################################################################


class TestNormalizationModules:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_batch_norm2d_eval(self, dtype):
        C = 8
        nmod = nn.BatchNorm2d(C, dtype=dtype).to('cuda')
        tmod = TNN.BatchNorm2d(C).to(DEVICE, torch.float32)
        nmod.eval()
        tmod.eval()

        rs_mean = np.zeros(C, dtype=np.float32)
        rs_var = np.ones(C, dtype=np.float32)
        with torch.no_grad():
            tmod.running_mean.copy_(torch.tensor(rs_mean))
            tmod.running_var.copy_(torch.tensor(rs_var))
            tmod.weight.fill_(1.0)
            tmod.bias.fill_(0.0)

        d = rdata(4, C, 6, 6)
        xn = nml_t(d, dtype)
        xt = tch_t(d, nml_typing.float32)

        out_n = nmod(xn)
        out_t = tmod(xt)
        assert out_n.shape == nectarml.typing.Size([4, C, 6, 6])
        assert out_t.shape == torch.Size([4, C, 6, 6])

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_batch_norm2d_train_stats_tracked(self, dtype):
        C = 4
        nmod = nn.BatchNorm2d(C, track_running_stats=True, dtype=dtype).to(
            'cuda'
        )
        nmod.train()
        d = rdata(2, C, 8, 8)
        nmod(nml_t(d, dtype))
        out = nmod(nml_t(d, dtype))
        assert out.shape == nectarml.typing.Size([2, C, 8, 8])

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_batch_norm1d(self, dtype):
        C = 8
        nmod = nn.BatchNorm1d(C, dtype=dtype).to('cuda')
        nmod.train()
        d = rdata(4, C, 16)
        out = nmod(nml_t(d, dtype))
        assert out.shape == nectarml.typing.Size([4, C, 16])

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_instance_norm2d(self, dtype):
        C = 4
        nmod = nn.InstanceNorm2d(C, dtype=dtype).to('cuda')
        tmod = TNN.InstanceNorm2d(C, affine=True).to(DEVICE, torch.float32)
        with torch.no_grad():
            tmod.weight.fill_(1.0)
            tmod.bias.fill_(0.0)
        d = rdata(2, C, 8, 8)
        out_n = nmod(nml_t(d, dtype))
        out_t = tmod(tch_t(d, nml_typing.float32))
        assert_close(out_n, out_t, dtype, tol=ATOL_LOOSE[dtype])

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_group_norm(self, dtype):
        C, G = 8, 4
        nmod = nn.GroupNorm(G, C, dtype=dtype).to('cuda')
        tmod = TNN.GroupNorm(G, C).to(DEVICE, torch.float32)
        with torch.no_grad():
            tmod.weight.fill_(1.0)
            tmod.bias.fill_(0.0)
        d = rdata(2, C, 8, 8)
        out_n = nmod(nml_t(d, dtype))
        out_t = tmod(tch_t(d, nml_typing.float32))
        assert_close(out_n, out_t, dtype, tol=ATOL_LOOSE[dtype])

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_layer_norm(self, dtype):
        ns = [4, 8]
        nmod = nn.LayerNorm(ns, dtype=dtype).to('cuda')
        tmod = TNN.LayerNorm(ns).to(DEVICE, torch.float32)
        with torch.no_grad():
            tmod.weight.fill_(1.0)
            tmod.bias.fill_(0.0)
        d = rdata(2, 4, 8)
        out_n = nmod(nml_t(d, dtype))
        out_t = tmod(tch_t(d, nml_typing.float32))
        assert_close(out_n, out_t, dtype, tol=ATOL_LOOSE[dtype])

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_layer_norm_backward(self, dtype):
        ns = [4, 8]
        nmod = nn.LayerNorm(ns, dtype=dtype).to('cuda')
        tmod = TNN.LayerNorm(ns).to(DEVICE, torch.float32)
        with torch.no_grad():
            tmod.weight.fill_(1.0)
            tmod.bias.fill_(0.0)
        d = rdata(2, 4, 8)
        xn = nml_t(d, dtype, True)
        xt = tch_t(d, nml_typing.float32, True)
        bwd(nmod(xn))
        bwd(tmod(xt))
        assert_grad(xn.grad, xt.grad, dtype, tol=ATOL_LOOSE[dtype])

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_batch_norm_no_affine(self, dtype):
        C = 4
        nmod = nn.BatchNorm2d(C, affine=False, dtype=dtype).to('cuda')
        nmod.train()
        d = rdata(2, C, 8, 8)
        out = nmod(nml_t(d, dtype))
        assert out.shape == nectarml.typing.Size([2, C, 8, 8])
        params = dict(nmod.named_parameters())
        assert 'gamma' not in params and 'weight' not in params


###############################################################################
# PADDING MODULES
###############################################################################


class TestPaddingModules:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_constant_pad1d(self, dtype):
        d = rdata(2, 4, 8)
        nmod = nn.ConstantPad1d((1, 2), value=0.0).to('cuda')
        tmod = TNN.ConstantPad1d((1, 2), value=0.0).to(DEVICE)
        assert_close(nmod(nml_t(d, dtype)), tmod(tch_t(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_constant_pad2d(self, dtype):
        d = rdata(2, 4, 8, 8)
        nmod = nn.ConstantPad2d((1, 1, 2, 2), value=0.0).to('cuda')
        tmod = TNN.ConstantPad2d((1, 1, 2, 2), value=0.0).to(DEVICE)
        assert_close(nmod(nml_t(d, dtype)), tmod(tch_t(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_reflection_pad1d(self, dtype):
        d = rdata(2, 4, 8)
        nmod = nn.ReflectionPad1d((2, 2)).to('cuda')
        tmod = TNN.ReflectionPad1d((2, 2)).to(DEVICE)
        assert_close(nmod(nml_t(d, dtype)), tmod(tch_t(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_reflection_pad2d(self, dtype):
        d = rdata(2, 4, 8, 8)
        nmod = nn.ReflectionPad2d((1, 1, 1, 1)).to('cuda')
        tmod = TNN.ReflectionPad2d((1, 1, 1, 1)).to(DEVICE)
        assert_close(nmod(nml_t(d, dtype)), tmod(tch_t(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_replication_pad2d(self, dtype):
        d = rdata(2, 4, 8, 8)
        nmod = nn.ReplicationPad2d((2, 2, 2, 2)).to('cuda')
        tmod = TNN.ReplicationPad2d((2, 2, 2, 2)).to(DEVICE)
        assert_close(nmod(nml_t(d, dtype)), tmod(tch_t(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_zero_pad2d(self, dtype):
        d = rdata(2, 4, 8, 8)
        nmod = nn.ZeroPad2d((1, 1, 1, 1)).to('cuda')
        tmod = TNN.ZeroPad2d((1, 1, 1, 1)).to(DEVICE)
        assert_close(nmod(nml_t(d, dtype)), tmod(tch_t(d, dtype)), dtype)


###############################################################################
# POOLING MODULES
###############################################################################


class TestPoolingModules:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_avg_pool1d(self, dtype):
        d = rdata(2, 4, 16)
        nmod = nn.AvgPool1d(kernel_size=2, stride=2).to('cuda')
        tmod = TNN.AvgPool1d(kernel_size=2, stride=2).to(DEVICE)
        assert_close(nmod(nml_t(d, dtype)), tmod(tch_t(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_avg_pool2d(self, dtype):
        d = rdata(2, 4, 16, 16)
        nmod = nn.AvgPool2d(kernel_size=2, stride=2).to('cuda')
        tmod = TNN.AvgPool2d(kernel_size=2, stride=2).to(DEVICE)
        assert_close(nmod(nml_t(d, dtype)), tmod(tch_t(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_avg_pool2d_padding(self, dtype):
        d = rdata(2, 4, 8, 8)
        nmod = nn.AvgPool2d(kernel_size=3, stride=1, padding=1).to('cuda')
        tmod = TNN.AvgPool2d(kernel_size=3, stride=1, padding=1).to(DEVICE)
        assert_close(nmod(nml_t(d, dtype)), tmod(tch_t(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_max_pool1d(self, dtype):
        d = rdata(2, 4, 16)
        nmod = nn.MaxPool1d(kernel_size=2, stride=2).to('cuda')
        tmod = TNN.MaxPool1d(kernel_size=2, stride=2).to(DEVICE)
        assert_close(nmod(nml_t(d, dtype)), tmod(tch_t(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_max_pool2d(self, dtype):
        d = rdata(2, 4, 16, 16)
        nmod = nn.MaxPool2d(kernel_size=2, stride=2).to('cuda')
        tmod = TNN.MaxPool2d(kernel_size=2, stride=2).to(DEVICE)
        assert_close(nmod(nml_t(d, dtype)), tmod(tch_t(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_max_pool2d_padding(self, dtype):
        d = rdata(2, 4, 8, 8)
        nmod = nn.MaxPool2d(kernel_size=3, stride=1, padding=1).to('cuda')
        tmod = TNN.MaxPool2d(kernel_size=3, stride=1, padding=1).to(DEVICE)
        assert_close(nmod(nml_t(d, dtype)), tmod(tch_t(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_avg_pool3d(self, dtype):
        d = rdata(2, 4, 8, 8, 8)
        nmod = nn.AvgPool3d(kernel_size=2, stride=2).to('cuda')
        tmod = TNN.AvgPool3d(kernel_size=2, stride=2).to(DEVICE)
        assert_close(nmod(nml_t(d, dtype)), tmod(tch_t(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_max_pool3d(self, dtype):
        d = rdata(2, 4, 8, 8, 8)
        nmod = nn.MaxPool3d(kernel_size=2, stride=2).to('cuda')
        tmod = TNN.MaxPool3d(kernel_size=2, stride=2).to(DEVICE)
        assert_close(nmod(nml_t(d, dtype)), tmod(tch_t(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_avg_pool2d_backward(self, dtype):
        d = rdata(2, 4, 8, 8)
        nmod = nn.AvgPool2d(kernel_size=2, stride=2).to('cuda')
        tmod = TNN.AvgPool2d(kernel_size=2, stride=2).to(DEVICE)
        xn = nml_t(d, dtype, True)
        xt = tch_t(d, dtype, True)
        bwd(nmod(xn))
        bwd(tmod(xt))
        assert_grad(xn.grad, xt.grad, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_max_pool2d_backward(self, dtype):
        d = rdata(2, 4, 8, 8)
        nmod = nn.MaxPool2d(kernel_size=2, stride=2).to('cuda')
        tmod = TNN.MaxPool2d(kernel_size=2, stride=2).to(DEVICE)
        xn = nml_t(d, dtype, True)
        xt = tch_t(d, dtype, True)
        bwd(nmod(xn))
        bwd(tmod(xt))
        assert_grad(xn.grad, xt.grad, dtype)


###############################################################################
# UPSAMPLE MODULE
###############################################################################


class TestUpsampleModule:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_nearest_scale(self, dtype):
        d = rdata(1, 3, 4, 4)
        nmod = nn.Upsample(scale_factor=2, mode='nearest').to('cuda')
        tmod = TNN.Upsample(scale_factor=2, mode='nearest').to(DEVICE)
        assert_close(nmod(nml_t(d, dtype)), tmod(tch_t(d, dtype)), dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_nearest_size(self, dtype):
        d = rdata(1, 1, 3, 3)
        nmod = nn.Upsample(size=(6, 6), mode='nearest').to('cuda')
        tmod = TNN.Upsample(size=(6, 6), mode='nearest').to(DEVICE)
        assert_close(nmod(nml_t(d, dtype)), tmod(tch_t(d, dtype)), dtype)


###############################################################################
# LOSS MODULES
###############################################################################


class TestLossModules:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize('reduction', REDUCTIONS)
    def test_l1_loss(self, dtype, reduction):
        inp, tgt = rdata(4, 8), rdata(4, 8)
        nmod = nn.L1Loss(reduction=reduction).to('cuda')
        tmod = TNN.L1Loss(reduction=reduction).to(DEVICE)
        assert_close(
            nmod(nml_t(inp, dtype), nml_t(tgt, dtype)),
            tmod(tch_t(inp, dtype), tch_t(tgt, dtype)),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize('reduction', REDUCTIONS)
    def test_mse_loss(self, dtype, reduction):
        inp, tgt = rdata(4, 8), rdata(4, 8)
        nmod = nn.L2Loss(reduction=reduction).to('cuda')
        tmod = TNN.MSELoss(reduction=reduction).to(DEVICE)
        assert_close(
            nmod(nml_t(inp, dtype), nml_t(tgt, dtype)),
            tmod(tch_t(inp, dtype), tch_t(tgt, dtype)),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize('reduction', REDUCTIONS)
    def test_huber_loss(self, dtype, reduction):
        inp, tgt = runit(4, 8), runit(4, 8)
        nmod = nn.HuberLoss(delta=1.0, reduction=reduction).to('cuda')
        tmod = TNN.HuberLoss(delta=1.0, reduction=reduction).to(DEVICE)
        assert_close(
            nmod(nml_t(inp, dtype), nml_t(tgt, dtype)),
            tmod(tch_t(inp, dtype), tch_t(tgt, dtype)),
            dtype,
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize('reduction', REDUCTIONS)
    def test_bce_loss(self, dtype, reduction):
        inp = np.random.uniform(0.05, 0.95, (4, 8)).astype(np.float32)
        tgt = np.random.uniform(0.05, 0.95, (4, 8)).astype(np.float32)
        nmod = nn.BCELoss(reduction=reduction).to('cuda')
        tmod = TNN.BCELoss(reduction=reduction).to(DEVICE)
        assert_close(
            nmod(nml_t(inp, dtype), nml_t(tgt, dtype)),
            tmod(tch_t(inp, dtype), tch_t(tgt, dtype)),
            dtype,
            tol=ATOL_LOOSE[dtype],
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    @pytest.mark.parametrize('reduction', REDUCTIONS)
    def test_bce_with_logits_loss(self, dtype, reduction):
        inp = runit(4, 8)
        tgt = np.random.uniform(0.0, 1.0, (4, 8)).astype(np.float32)
        nmod = nn.BCEWithLogitsLoss(reduction=reduction).to('cuda')
        tmod = TNN.BCEWithLogitsLoss(reduction=reduction).to(DEVICE)
        assert_close(
            nmod(nml_t(inp, dtype), nml_t(tgt, dtype)),
            tmod(tch_t(inp, dtype), tch_t(tgt, dtype)),
            dtype,
            tol=ATOL_LOOSE[dtype],
        )

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_l1_loss_backward(self, dtype):
        inp, tgt = rdata(4, 8), rdata(4, 8)
        nmod = nn.L1Loss().to('cuda')
        tmod = TNN.L1Loss().to(DEVICE)
        xn = nml_t(inp, dtype, True)
        xt = tch_t(inp, dtype, True)
        bwd(nmod(xn, nml_t(tgt, dtype)))
        bwd(tmod(xt, tch_t(tgt, dtype)))
        assert_grad(xn.grad, xt.grad, dtype)

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_triplet_margin_loss(self, dtype):
        a = np.random.uniform(0.0, 0.3, (4, 8)).astype(np.float32)
        p = np.random.uniform(0.0, 0.3, (4, 8)).astype(np.float32)
        n = np.random.uniform(0.7, 1.0, (4, 8)).astype(np.float32)
        nmod = nn.TripletMarginLoss(margin=1.0).to('cuda')
        tmod = TNN.TripletMarginLoss(margin=1.0).to(DEVICE)
        assert_close(
            nmod(nml_t(a, dtype), nml_t(p, dtype), nml_t(n, dtype)),
            tmod(tch_t(a, dtype), tch_t(p, dtype), tch_t(n, dtype)),
            dtype,
            tol=ATOL_LOOSE[dtype],
        )


###############################################################################
# ATTENTION MODULE
###############################################################################


class TestMultiheadAttention:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_self_attention_shape(self, dtype):
        embed_dim, num_heads = 16, 4
        nmod = nn.MultiheadAttention(embed_dim, num_heads, dtype=dtype).to(
            'cuda'
        )
        S, B = 8, 2
        q = nml_t(rdata(S, B, embed_dim), dtype)
        out, weights = nmod(q, q, q)
        assert out.shape == nectarml.typing.Size([S, B, embed_dim])
        assert weights.shape == nectarml.typing.Size([B, S, S])

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_cross_attention_shape(self, dtype):
        embed_dim, num_heads = 16, 4
        nmod = nn.MultiheadAttention(embed_dim, num_heads, dtype=dtype).to(
            'cuda'
        )
        S_q, S_kv, B = 6, 10, 2
        q = nml_t(rdata(S_q, B, embed_dim), dtype)
        kv = nml_t(rdata(S_kv, B, embed_dim), dtype)
        out, weights = nmod(q, kv, kv)
        assert out.shape == nectarml.typing.Size([S_q, B, embed_dim])
        assert weights.shape == nectarml.typing.Size([B, S_q, S_kv])

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_batch_first(self, dtype):
        embed_dim, num_heads = 16, 4
        nmod = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True, dtype=dtype
        ).to('cuda')
        B, S = 2, 8
        q = nml_t(rdata(B, S, embed_dim), dtype)
        out, _ = nmod(q, q, q)
        assert out.shape == nectarml.typing.Size([B, S, embed_dim])

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_backward(self, dtype):
        embed_dim, num_heads = 8, 2
        nmod = nn.MultiheadAttention(embed_dim, num_heads, dtype=dtype).to(
            'cuda'
        )
        S, B = 4, 2
        xn = nml_t(rdata(S, B, embed_dim), dtype, True)
        out, _ = nmod(xn, xn, xn)
        bwd(out)
        assert xn.grad is not None


###############################################################################
# COMPOSED NETWORKS: end-to-end gradient flow
###############################################################################


class TestComposedNetworks:
    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_simple_mlp(self, dtype):
        model = nn.Sequential(
            nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4)
        ).to('cuda', dtype=dtype)

        x = nml_t(rdata(2, 8), dtype, True)
        loss = model(x).sum()
        loss.backward()
        assert x.grad is not None
        for p in model.parameters():
            assert p.grad is not None

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_conv_bn_relu(self, dtype):
        conv = nn.Conv2d(3, 8, 3, padding=1, dtype=dtype).to('cuda')
        bn = nn.BatchNorm2d(8, dtype=dtype).to('cuda')
        act = nn.ReLU().to('cuda')

        x = nml_t(rdata(2, 3, 8, 8), dtype, True)
        bn.train()
        out = act(bn(conv(x)))
        bwd(out)
        assert x.grad is not None

    @pytest.mark.parametrize('dtype', DTYPES, ids=DTYPE_IDS)
    def test_encoder_decoder_shape(self, dtype):
        enc = nn.Conv2d(1, 8, 3, padding=1, dtype=dtype).to('cuda')
        pool = nn.MaxPool2d(2, 2).to('cuda')
        dec = nn.ConvTranspose2d(
            8, 1, 3, stride=2, padding=1, output_padding=1, dtype=dtype
        ).to('cuda')
        x = nml_t(rdata(1, 1, 8, 8), dtype)
        out = dec(pool(enc(x)))
        assert out.shape == nectarml.typing.Size([1, 1, 8, 8])
