import pytest

from typing import Any
from collections.abc import Generator

import numpy as np

### FIXTURES ###

@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng()

@pytest.fixture
def sample_input1d(
    rng: np.random.Generator, 
    shape: tuple[int, ...] = (1, 3, 16)
) -> np.ndarray:
    return rng.random(shape).astype(np.float32)

@pytest.fixture
def sample_input2d(
    rng: np.random.Generator, 
    shape: tuple[int, ...] = (1, 3, 16, 16)
) -> np.ndarray:
    return rng.random(shape).astype(np.float32)

### TRIGGERS ###

def pytest_addoption(parser: pytest.Parser) -> None:
    pass

def pytest_configure(config: pytest.Config):
    pass

def pytest_collection_modifyitems(config: pytest.Config, items: pytest.Item):
    pass

def _cleanup() -> None:
    pass

@pytest.fixture(scope="session", autouse=True)
def session_once(pytestconfig: pytest.Config) -> Generator[None, Any, None]:
    np.random.seed(42)
    yield
    _cleanup()

