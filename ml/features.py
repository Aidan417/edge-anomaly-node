"""Hand-crafted vibration statistics used by the baseline detector.

These are the features a condition-monitoring engineer would reach for
first. They cost almost nothing to compute on a microcontroller, so a
learned model has to beat them clearly to earn its place.
"""
import numpy as np


def rms(w: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean(w ** 2, axis=-1))


def kurtosis(w: np.ndarray) -> np.ndarray:
    c = w - w.mean(axis=-1, keepdims=True)
    m2 = np.mean(c ** 2, axis=-1)
    m4 = np.mean(c ** 4, axis=-1)
    return m4 / np.maximum(m2 ** 2, 1e-12)


def crest_factor(w: np.ndarray) -> np.ndarray:
    return np.max(np.abs(w), axis=-1) / np.maximum(rms(w), 1e-12)


def feature_table(w: np.ndarray) -> np.ndarray:
    """(n, 3) array of [rms, kurtosis, crest_factor] per window."""
    return np.stack([rms(w), kurtosis(w), crest_factor(w)], axis=-1)
