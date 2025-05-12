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
