"""Shared fixtures for the Phase 1 simulator tests."""

import numpy as np
import pytest

from powertoflops.config import DEFAULT


@pytest.fixture
def cfg():
    """The default (placeholder) configuration."""
    return DEFAULT


@pytest.fixture
def rng():
    """A seeded RNG for reproducible noise draws."""
    return np.random.default_rng(0)
