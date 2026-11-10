"""Loading and windowing of the CWRU bearing dataset.

Ten recordings are used, all from the drive-end accelerometer at 0 HP
load / ~1797 rpm:

    97.mat   healthy baseline                        (recorded at 48 kHz)
    105 / 169 / 209   inner-race fault, 0.007 / 0.014 / 0.021 in   (12 kHz)
    118 / 185 / 222   ball fault,       0.007 / 0.014 / 0.021 in   (12 kHz)
    130 / 197 / 234   outer-race fault, 0.007 / 0.014 / 0.021 in   (12 kHz)

The healthy baseline is recorded at 48 kHz, the fault files at 12 kHz
(the 1797 rpm shaft line lands at 29.95 Hz only under that reading, and
CWRU's other baseline files have exactly the 48 kHz record length). We
decimate the baseline by 4, then low-pass every recording at the same
4 kHz. Without the common filter the decimated baseline would have less
energy near 6 kHz than the native 12 kHz files, and a model could
separate healthy from faulty on that acquisition artefact alone.

The model only ever trains on healthy data; the fault recordings are
held back entirely for evaluation. That mirrors a real deployment, where
you have plenty of "machine running normally" data and no labelled
failures.

Two preprocessing variants are supported:

    abs   one global scale factor; absolute vibration amplitude is kept.
          Easy on this dataset (faulty bearings simply vibrate harder),
          but brittle in practice: amplitude changes with load, sensor
          mounting and gain.
    norm  each window is zero-mean / unit-variance. Only the *shape* of
          the vibration is left, so the detector has to learn what
          healthy vibration looks like rather than how loud it is.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, decimate, sosfiltfilt

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
FS = 12_000           # working sample rate, Hz
LOWPASS_HZ = 4_000    # common anti-alias band applied to EVERY recording

# label -> (file name, native sample rate)
NORMAL = ("97.mat", 48_000)
FAULTS = {
    "inner_007": ("105.mat", 12_000),
    "inner_014": ("169.mat", 12_000),
    "inner_021": ("209.mat", 12_000),
    "ball_007": ("118.mat", 12_000),
    "ball_014": ("185.mat", 12_000),
    "ball_021": ("222.mat", 12_000),
    "outer_007": ("130.mat", 12_000),
    "outer_014": ("197.mat", 12_000),
    "outer_021": ("234.mat", 12_000),
}
CONDITIONS = ["normal", *FAULTS]

WINDOW = 256          # samples per window (~21 ms at 12 kHz)
STRIDE = 64           # 75 % overlap for training windows (healthy data is only ~5 s)
VARIANTS = ("abs", "norm")


def load_signal(path: Path, native_fs: int, fs: int = FS) -> np.ndarray:
    """Return the drive-end accelerometer channel at `fs` Hz as float32."""
    mat = loadmat(path)
    keys = [k for k in mat if re.fullmatch(r"X\d+_DE_time", k)]
    if not keys:
        raise KeyError(f"no drive-end channel in {path.name}: {list(mat)}")
    x = mat[keys[0]].astype(np.float64).ravel()
    if native_fs != fs:
        q, r = divmod(native_fs, fs)
        if r:
            raise ValueError(f"{native_fs} Hz is not an integer multiple of {fs} Hz")
        x = decimate(x, q, zero_phase=True)      # anti-aliased, no phase shift
    sos = butter(8, LOWPASS_HZ, fs=fs, output="sos")
    x = sosfiltfilt(sos, x)                       # identical band for every file
    return x.astype(np.float32)


def make_windows(x: np.ndarray, window: int = WINDOW, stride: int = STRIDE) -> np.ndarray:
    """Slice a 1-D signal into overlapping windows, shape (n, window)."""
    n = (len(x) - window) // stride + 1
    idx = np.arange(window)[None, :] + stride * np.arange(n)[:, None]
    return x[idx]


def normalise_windows(w: np.ndarray) -> np.ndarray:
    """Per-window zero-mean, unit-variance (the `norm` variant)."""
    c = w - w.mean(axis=-1, keepdims=True)
    return c / np.maximum(c.std(axis=-1, keepdims=True), 1e-6)


@dataclass
class Dataset:
    variant: str
    train: np.ndarray            # healthy windows for training
    val: np.ndarray              # healthy windows for threshold selection
    test_normal: np.ndarray      # healthy windows never seen in training
    test_faults: dict[str, np.ndarray]
    scale: float                 # global divisor (abs variant), 1.0 for norm

    @property
    def test_x(self) -> np.ndarray:
        return np.concatenate([self.test_normal, *self.test_faults.values()])

    @property
    def test_y(self) -> np.ndarray:
        """1 = fault, 0 = healthy."""
        n_fault = sum(len(v) for v in self.test_faults.values())
        return np.concatenate([np.zeros(len(self.test_normal)), np.ones(n_fault)]).astype(int)

    @property
    def test_label(self) -> np.ndarray:
        """String label per test window (for per-fault reporting)."""
        labels = ["normal"] * len(self.test_normal)
        for name, v in self.test_faults.items():
            labels += [name] * len(v)
        return np.array(labels)


def load_dataset(variant: str = "norm", raw_dir: Path = RAW_DIR, window: int = WINDOW,
                 stride: int = STRIDE, seed: int = 0) -> Dataset:
    """Build the train / val / test split.

    The healthy recording is split *in time* (first 60 % train, next 20 %
    val, last 20 % test) rather than by shuffling windows, so overlapping
    windows never leak across the split.
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    normal = load_signal(raw_dir / NORMAL[0], NORMAL[1])
    n = len(normal)
    a, b = int(0.6 * n), int(0.8 * n)
    train = make_windows(normal[:a], window, stride)
    val = make_windows(normal[a:b], window, window)
    test_normal = make_windows(normal[b:], window, window)
    faults = {name: make_windows(load_signal(raw_dir / f, fs), window, window)
              for name, (f, fs) in FAULTS.items()}

    if variant == "abs":
        # One global scale from the training data only. A single divisor
        # keeps the on-device preprocessing to one multiply and preserves
        # absolute amplitude, which on this rig is a strong fault cue.
        scale = float(np.std(train)) * 4.0
        prep = lambda w: w / scale          # noqa: E731
    else:
        scale = 1.0
        prep = normalise_windows

    rng = np.random.default_rng(seed)
    train = prep(train)
    rng.shuffle(train)
    return Dataset(variant, train, prep(val), prep(test_normal),
                   {k: prep(v) for k, v in faults.items()}, scale)
