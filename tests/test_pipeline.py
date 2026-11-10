"""Unit tests that run without the dataset or a trained model."""
import numpy as np
import pytest

from ml.data import make_windows
from ml.features import crest_factor, feature_table, kurtosis, rms
from ml.model import build_autoencoder


def test_make_windows_shape_and_content():
    x = np.arange(1000, dtype=np.float32)
    w = make_windows(x, window=256, stride=128)
    assert w.shape == (6, 256)
    assert w[0, 0] == 0 and w[1, 0] == 128 and w[-1, -1] == 5 * 128 + 255


def test_rms_of_sine():
    t = np.arange(256) / 256
    w = np.sin(2 * np.pi * 4 * t)[None, :]
    assert rms(w)[0] == pytest.approx(1 / np.sqrt(2), rel=1e-2)


def test_kurtosis_gaussian_is_about_three():
    rng = np.random.default_rng(0)
    w = rng.normal(size=(50, 4096))
    assert kurtosis(w).mean() == pytest.approx(3.0, abs=0.1)


def test_crest_factor_impulse_is_large():
    w = np.zeros((1, 256), dtype=np.float32)
    w[0, 10] = 1.0
    assert crest_factor(w)[0] == pytest.approx(16.0, rel=1e-3)


def test_feature_table_shape():
    assert feature_table(np.random.rand(7, 256)).shape == (7, 3)


def test_autoencoder_shapes_and_size():
    m = build_autoencoder(256)
    assert m.input_shape == (None, 256, 1) and m.output_shape == (None, 256, 1)
    assert m.count_params() < 20_000, "model must stay small enough for an MCU"
    y = m.predict(np.zeros((2, 256, 1), dtype=np.float32), verbose=0)
    assert y.shape == (2, 256, 1)


def test_normalise_windows_is_zero_mean_unit_var():
    from ml.data import normalise_windows
    rng = np.random.default_rng(1)
    w = normalise_windows(rng.normal(3.0, 7.0, size=(5, 256)))
    assert np.allclose(w.mean(axis=1), 0, atol=1e-6)
    assert np.allclose(w.std(axis=1), 1, atol=1e-5)
