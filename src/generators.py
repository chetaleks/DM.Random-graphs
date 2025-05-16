"""
Модуль generators: функции-генераторы выборок для H0 и H1.
"""

import numpy as np
from scipy.stats import laplace, norm


def sample_h0(n: int) -> np.ndarray:
    """
    Генерация n независимых N(0,1)-выборок.
    """
    return norm.rvs(loc=0.0, scale=1.0, size=n)


def sample_h1(n: int, beta: float = np.sqrt(0.5)) -> np.ndarray:
    """
    Генерация n независимых Laplace(0,β)-выборок.
    """
    return laplace.rvs(loc=0.0, scale=beta, size=n)


def sample_H2(n, l=2 / (3**0.5)):
    """Генерация выборки из Exp(l) размера n."""
    return expon.rvs(loc=0, scale=l, size=n)


def sample_H3(n, b=1.0, alpha=3):
    """Генерация выборки из Pareto(a) размера n."""
    return pareto.rvs(b, loc=0, scale=alpha, size=n)
