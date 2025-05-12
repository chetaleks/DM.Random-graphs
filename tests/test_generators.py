"""
Unit tests for функции из модуля src.generators.
"""

import numpy as np
import pytest

from src.generators import sample_h0, sample_h1


def test_sample_h0_shape_and_type():
    """sample_h0 возвращает numpy.ndarray правильной формы."""
    arr = sample_h0(10)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (10,)


def test_sample_h1_shape_and_type():
    """sample_h1 возвращает numpy.ndarray правильной формы."""
    arr = sample_h1(15, beta=2.0)
    assert isinstance(arr, np.ndarray)
    assert arr.shape == (15,)


def test_sample_statistics_normal():
    """Нормальное распределение: mean≈0, std≈1."""
    np.random.seed(0)
    x = sample_h0(100_000)
    assert abs(np.mean(x)) < 0.01
    assert abs(np.std(x) - 1) < 0.01


def test_sample_statistics_laplace():
    """Laplace(0, β): mean≈0, std≈β√2."""
    np.random.seed(1)
    y = sample_h1(100_000, beta=2.0)
    assert abs(np.mean(y)) < 0.01
    assert abs(np.std(y) - (2.0 * np.sqrt(2))) < 0.01


def test_sample_invalid_size_and_beta():
    """Отрицательные size или β приводят к ValueError."""
    with pytest.raises(ValueError):
        sample_h0(-1)
    with pytest.raises(ValueError):
        sample_h1(10, beta=-0.5)
