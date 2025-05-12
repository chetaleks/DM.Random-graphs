"""
Модуль simulator: Monte–Carlo-симуляции статистик по KNN/dist графам.
"""

import numpy as np
import pandas as pd

from .features import (
    compute_clique_number_1d,
    compute_edge_count,
    compute_triangles,
)
from .generators import sample_h0, sample_h1
from .graphs import build_distance_graph, build_knn_graph


def simulate_statistics(
    n: int, trials: int, graph_type: str, param: float, beta: float
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Для trials симуляций генерируем два набора выборок:
    H0 ~ N(0,1) и H1 ~ Laplace(0,β), строим граф указанного типа,
    вычисляем четыре характеристики и возвращаем два DataFrame:
    по одному для H0 и H1 (trials × 4).
    """
    stats0 = []
    stats1 = []
    for _ in range(trials):
        sample0 = sample_h0(n)
        sample1 = sample_h1(n, beta)

        if graph_type == "knn":
            graph0 = build_knn_graph(sample0, int(param))
            graph1 = build_knn_graph(sample1, int(param))
            # avg_clique не определён для knn-графа → NaN
            row0 = {
                "num_edges": compute_edge_count(graph0),
                "num_triangles": compute_triangles(graph0),
                "max_degree": max(dict(graph0.degree()).values()),
                "avg_clique": np.nan,
            }
            row1 = {
                "num_edges": compute_edge_count(graph1),
                "num_triangles": compute_triangles(graph1),
                "max_degree": max(dict(graph1.degree()).values()),
                "avg_clique": np.nan,
            }

        elif graph_type == "dist":
            graph0 = build_distance_graph(sample0, param)
            graph1 = build_distance_graph(sample1, param)
            row0 = {
                "num_edges": compute_edge_count(graph0),
                "num_triangles": compute_triangles(graph0),
                "max_degree": max(dict(graph0.degree()).values()),
                "avg_clique": compute_clique_number_1d(sample0, param),
            }
            row1 = {
                "num_edges": compute_edge_count(graph1),
                "num_triangles": compute_triangles(graph1),
                "max_degree": max(dict(graph1.degree()).values()),
                "avg_clique": compute_clique_number_1d(sample1, param),
            }

        else:
            raise ValueError(f"Unknown graph_type={graph_type!r}")

        stats0.append(row0)
        stats1.append(row1)

    return pd.DataFrame(stats0), pd.DataFrame(stats1)
