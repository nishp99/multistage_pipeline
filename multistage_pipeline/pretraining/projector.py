"""Population and realized geometry of a fixed-rank PCA projector."""

from functools import lru_cache
from typing import NamedTuple

import numpy as np
import jax.numpy as jnp
from jax import random
from scipy.linalg import eigh
from .traces import lower_order_traces, vector_vector_overlaps_from_moments


class Geometry(NamedTuple):
    initial: np.ndarray
    teacher: np.ndarray
    omega: np.ndarray
    traces: np.ndarray


def compact_geometry(moments, initial, teacher):
    mass, m1, m2, m3 = moments[:4]
    gram = np.array([[m2, m1 - m2], [m1 - m2, mass - 2 * m1 + m2]])
    omega = np.linalg.pinv(gram, rcond=1e-13) @ [m3, m2 - m3]
    return Geometry(
        np.asarray(initial), np.asarray(teacher), omega, gram[[0, 0, 1], [0, 1, 1]]
    )


@lru_cache(maxsize=2048)
def population_moments(alpha, signal_fraction=0.2, delta=0.1, quick=False):
    if not 0 < signal_fraction < 1 or alpha < signal_fraction or not 0 < delta < 1:
        raise ValueError("Require 0 < signal_fraction <= alpha and 0 < delta < 1.")
    grids = (
        dict(n_grid1d=256, n_grid2=32, n_grid3=20, n_grid3_check=10) if quick else {}
    )
    raw = lower_order_traces(float(alpha), 1 - signal_fraction, delta, **grids)
    return np.r_[signal_fraction, np.asarray(raw[:3])]


def population_geometry(moments, block_moments=None):
    f, m1, m2 = moments[:3]
    if block_moments is None:
        block_moments = [(1.0, 0.0), (0.0, 0.0), (f, 1 - f)]
    pairs = []
    for signal, noise in block_moments:
        total, first = vector_vector_overlaps_from_moments(
            signal, noise, m1 / f, m2 / f, 1 - f
        )
        pairs.append([first, total - first])
    return compact_geometry(moments, np.r_[pairs[1], pairs[2]], pairs[0])


def vectors(dimension, signal_fraction=0.2, seeds=(42, 43), balanced=False):
    rank = round(dimension * signal_fraction)
    teacher = np.zeros(dimension)
    teacher[:rank] = np.random.RandomState(seeds[0]).normal(size=rank)
    teacher *= np.sqrt(dimension) / np.linalg.norm(teacher)
    student = np.random.RandomState(seeds[1]).normal(size=dimension)
    if balanced:
        student -= teacher * (teacher @ student / dimension)
        for block in (student[:rank], student[rank:]):
            block *= np.sqrt(len(block)) / np.linalg.norm(block)
    else:
        student *= np.sqrt(dimension) / np.linalg.norm(student)
    return teacher, student


def block_moments(teacher, student, rank):
    d = len(student)
    return [
        (a[:rank] @ b[:rank] / d, a[rank:] @ b[rank:] / d)
        for a, b in ((teacher, teacher), (student, teacher), (student, student))
    ]


def gaussian_data(key, samples, dimension, rank, delta, dtype=np.float64):
    scale = jnp.where(jnp.arange(dimension) < rank, 1.0, jnp.sqrt(delta)).astype(dtype)
    return random.normal(key, (samples, dimension), dtype=dtype) * scale


def pca_basis(key, alpha, dimension, signal_fraction=0.2, delta=0.1, *, center=True):
    rank = round(signal_fraction * dimension)
    count = round(alpha * dimension)
    if count - int(center) < rank:
        raise ValueError("Too few samples for the requested PCA rank.")
    if isinstance(key, np.random.Generator):
        scale = np.r_[np.ones(rank), np.full(dimension - rank, np.sqrt(delta))]
        data = key.normal(size=(count, dimension)) * scale
    else:
        data = np.asarray(gaussian_data(key, count, dimension, rank, delta))
    if center:
        data = data - data.mean(axis=0)
    if count < dimension:
        eigen, left = eigh(
            data @ data.T / (count - int(center)),
            subset_by_index=(count - rank, count - 1),
            check_finite=False,
        )
        return (data.T @ left) / np.sqrt((count - int(center)) * eigen)
    return eigh(
        data.T @ data / (count - int(center)),
        subset_by_index=(dimension - rank, dimension - 1),
        check_finite=False,
    )[1]


def spectral_moments(basis):
    d, rank = basis.shape
    eigenvalues = np.linalg.eigvalsh(basis[:rank].T @ basis[:rank])
    return np.array([np.sum(eigenvalues**k) / d for k in range(6)])


def measured_spectral_state(basis, teacher, student):
    """Return contractions against P, C and C² in the retained coordinates."""
    d, rank = basis.shape
    t, w = basis.T @ teacher, basis.T @ student
    signal = basis[:rank].T @ basis[:rank]
    sw, st = signal @ w, signal @ t
    state = np.array([w @ t, sw @ t, sw @ st, w @ w, sw @ w, sw @ sw]) / d
    return state, np.array([t @ t, st @ t, st @ st]) / d


def measured_geometry(basis, teacher, student):
    s, t = measured_spectral_state(basis, teacher, student)
    return compact_geometry(
        spectral_moments(basis),
        [s[1], s[0] - s[1], s[4], s[3] - s[4]],
        [t[1], t[0] - t[1]],
    )
