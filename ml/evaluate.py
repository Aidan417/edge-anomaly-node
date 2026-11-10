"""Evaluate the autoencoder as an anomaly detector and compare it with
hand-crafted single-feature baselines.

    python -m ml.evaluate --variant norm

Writes models/metrics_<variant>.json and figures under docs/figures/.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from tensorflow import keras

from ml.data import CONDITIONS, FAULTS, VARIANTS, load_dataset
from ml.features import feature_table
from ml.model import reconstruction_error

ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "models"
FIGS = ROOT / "docs" / "figures"
FEATURE_NAMES = ["rms", "kurtosis", "crest_factor"]


def threshold_from_normal(score_val: np.ndarray, pct: float = 99.0) -> float:
    """Alarm threshold: the given percentile of scores on healthy validation
    data, i.e. we accept ~1 % false alarms on windows we know are healthy."""
    return float(np.percentile(score_val, pct))


def detection_report(score: np.ndarray, y: np.ndarray, labels: np.ndarray, thr: float) -> dict:
    pred = (score > thr).astype(int)
    out = {
        "threshold": thr,
        "roc_auc": float(roc_auc_score(y, score)),
        "false_alarm_rate": float(pred[y == 0].mean()),
        "detection_rate": float(pred[y == 1].mean()),
        "per_condition": {},
    }
    for name in CONDITIONS:
        m = labels == name
        if m.any():
            out["per_condition"][name] = float(pred[m].mean())
    return out


def evaluate_variant(variant: str, make_figures: bool = True) -> dict:
    ds = load_dataset(variant)
    model = keras.models.load_model(MODELS / f"autoencoder_{variant}.keras")

    scores = {}
    ae_val = reconstruction_error(model, ds.val)
    scores["autoencoder"] = (ae_val, reconstruction_error(model, ds.test_x))
    f_val, f_test = feature_table(ds.val), feature_table(ds.test_x)
    for i, name in enumerate(FEATURE_NAMES):
        if variant == "norm" and name == "rms":
            continue            # rms is 1.0 for every normalised window
        scores[name] = (f_val[:, i], f_test[:, i])

    metrics = {"variant": variant, "n_test_windows": int(len(ds.test_y)),
               "params": int(model.count_params()), "detectors": {}}
    for name, (val, test) in scores.items():
        metrics["detectors"][name] = detection_report(test, ds.test_y, ds.test_label,
                                                       threshold_from_normal(val))
    MODELS.mkdir(exist_ok=True)
    (MODELS / f"metrics_{variant}.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))

    if make_figures:
        FIGS.mkdir(parents=True, exist_ok=True)
        ae_test = scores["autoencoder"][1]
        ae_thr = metrics["detectors"]["autoencoder"]["threshold"]

        # 1. reconstruction-error distribution per condition
        fig, ax = plt.subplots(figsize=(9, 4.5))
        for name in CONDITIONS:
            m = ds.test_label == name
            ax.hist(np.log10(ae_test[m]), bins=50, alpha=0.55, label=name.replace("_", " "))
        ax.axvline(np.log10(ae_thr), color="k", ls="--", label="alarm threshold")
        ax.set_xlabel("log10 reconstruction error")
        ax.set_ylabel("windows")
        ax.set_title(f"Autoencoder reconstruction error by condition ({variant})")
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(FIGS / f"error_hist_{variant}.png", dpi=130)
        plt.close(fig)

        # 2. ROC: autoencoder vs baselines
        fig, ax = plt.subplots(figsize=(5, 5))
        for name, (_, test) in scores.items():
            fpr, tpr, _ = roc_curve(ds.test_y, test)
            auc = metrics["detectors"][name]["roc_auc"]
            ax.plot(fpr, tpr, label=f"{name} (AUC {auc:.3f})")
        ax.plot([0, 1], [0, 1], "k:", lw=0.8)
        ax.set_xlabel("false alarm rate")
        ax.set_ylabel("detection rate")
        ax.set_title(f"ROC, all faults vs healthy ({variant})")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(FIGS / f"roc_{variant}.png", dpi=130)
        plt.close(fig)

        # 3. per-condition detection rate, grouped bars
        names = list(scores)
        conds = [c for c in CONDITIONS if c != "normal"]
        x = np.arange(len(conds))
        w = 0.8 / len(names)
        fig, ax = plt.subplots(figsize=(9, 4))
        for j, name in enumerate(names):
            rates = [metrics["detectors"][name]["per_condition"][c] for c in conds]
            ax.bar(x + (j - (len(names) - 1) / 2) * w, rates, w, label=name)
        ax.set_xticks(x)
        ax.set_xticklabels([c.replace("_", " ") for c in conds], rotation=30, ha="right")
        ax.set_ylabel("detection rate at 1 % false alarms")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Detection rate per fault type and size ({variant})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGS / f"per_condition_{variant}.png", dpi=130)
        plt.close(fig)

        # 4. example reconstructions
        show = ["normal", "inner_007", "ball_014", "outer_021"]
        fig, axes = plt.subplots(len(show), 1, figsize=(8, 7), sharex=True)
        for ax, name in zip(axes, show):
            wnd = ds.test_normal[0] if name == "normal" else ds.test_faults[name][0]
            rec = model.predict(wnd[None, :, None], verbose=0)[0, :, 0]
            ax.plot(wnd, lw=0.8, label="measured")
            ax.plot(rec, lw=0.8, label="reconstructed")
            ax.set_ylabel(name.replace("_", " "), rotation=0, ha="right", va="center")
        axes[0].legend(loc="upper right")
        axes[-1].set_xlabel("sample (12 kHz)")
        fig.suptitle(f"256-sample windows: measured vs reconstructed ({variant})")
        fig.tight_layout()
        fig.savefig(FIGS / f"reconstructions_{variant}.png", dpi=130)
        plt.close(fig)
    return metrics


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=VARIANTS, default="norm")
    a = p.parse_args()
    evaluate_variant(a.variant)
