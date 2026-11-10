"""Train the autoencoder on healthy vibration only.

    python -m ml.train --variant norm --epochs 300

Writes models/autoencoder_<variant>.keras and models/train_history_<variant>.json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from tensorflow import keras

from ml.data import VARIANTS, load_dataset
from ml.model import build_autoencoder

MODELS = Path(__file__).resolve().parents[1] / "models"


def main(variant: str = "norm", epochs: int = 300, batch: int = 64, seed: int = 0) -> Path:
    tf.random.set_seed(seed)
    np.random.seed(seed)
    ds = load_dataset(variant, seed=seed)
    model = build_autoencoder(ds.train.shape[1])
    model.summary()

    x_tr = ds.train[..., None]
    x_va = ds.val[..., None]
    hist = model.fit(
        x_tr, x_tr,
        validation_data=(x_va, x_va),
        epochs=epochs, batch_size=batch, shuffle=True, verbose=2,
        callbacks=[keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)],
    )
    MODELS.mkdir(exist_ok=True)
    out = MODELS / f"autoencoder_{variant}.keras"
    model.save(out)
    history = {k: [float(v) for v in vs] for k, vs in hist.history.items()}
    history["scale"] = ds.scale
    history["n_train_windows"] = int(len(ds.train))
    (MODELS / f"train_history_{variant}.json").write_text(json.dumps(history, indent=2))
    print(f"saved {out}  params={model.count_params()}  epochs={len(history['loss'])}")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=VARIANTS, default="norm")
    p.add_argument("--epochs", type=int, default=300)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    main(a.variant, a.epochs, a.batch, a.seed)
