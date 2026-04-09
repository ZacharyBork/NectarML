import pytest

import numpy as np

import torch
import torch.nn.functional as torch_F

import nectarml
import nectarml.functional as F

### ACTIVATION FUNCTIONS ###

ACTIVATION_FUNCTIONS = [
    (F.relu,         torch_F.relu,         {}),
    (F.leaky_relu,   torch_F.leaky_relu,   {}),
    (F.elu,          torch_F.elu,          {}),
    (F.selu,         torch_F.selu,         {}),
    (F.sigmoid,      torch_F.sigmoid,      {}),
    (F.tanh,         torch_F.tanh,         {}),
    (F.softmax,      torch_F.softmax,      {'dim': -1}),
    (F.log_softmax,  torch_F.log_softmax,  {'dim': -1}),
    #(F.gelu,         torch_F.gelu,         {}),
    (F.silu,         torch_F.silu,         {}),
    (F.swish,        torch_F.silu,         {}),
    (F.softplus,     torch_F.softplus,     {}),
    (F.mish,         torch_F.mish,         {}),
    (F.hardtanh,     torch_F.hardtanh,     {}),
    #(F.hardsigmoid,  torch_F.hardsigmoid,  {}),
    (F.hardswish,    torch_F.hardswish,    {}),
    (F.softsign,     torch_F.softsign,     {}),
    (F.softmin,      torch_F.softmin,      {'dim': -1}),
]

@pytest.fixture
def sample_input():
    np.random.seed(42)
    data = np.random.randn(4, 16).astype(np.float32)
    return data

@pytest.mark.parametrize('nectarml_fn, torch_fn, kwargs', ACTIVATION_FUNCTIONS)
def test_activation(nectarml_fn, torch_fn, kwargs, sample_input):
    nectar_input = nectarml.Tensor(sample_input, requires_grad=True)
    torch_input  = torch.tensor(sample_input, requires_grad=True)
    
    nectar_out = nectarml_fn(nectar_input, **kwargs)
    torch_out  = torch_fn(torch_input, **kwargs)
    
    _nectar = nectar_out.detach().numpy()
    _torch  = torch_out.detach().numpy()
    assert np.allclose(_nectar, _torch, atol=1e-5), (
        f'{nectarml_fn.__name__} (forward): '
        f'max diff = {np.abs(_nectar - _torch).max()}')
    
    nectar_out.mean().backward()
    torch_out.mean().backward()
    
    nectar_grad = nectar_input.grad.numpy()
    torch_grad  = torch_input.grad.numpy()
    
    assert np.allclose(nectar_grad, torch_grad, atol=1e-5), (
        f'{nectarml_fn.__name__} (backward): '
        f'max diff = {np.abs(nectar_grad - torch_grad).max()}')

