"""A small 1-D convolutional autoencoder.

Sized for a microcontroller: ~5k parameters, no batch-norm (so the
quantised graph stays simple), ReLU only. The encoder compresses a
256-sample window to a 32-value code; the decoder reconstructs it.
Healthy windows reconstruct well; fault windows don't.
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

WINDOW = 256


def build_autoencoder(window: int = WINDOW) -> keras.Model:
    inp = keras.Input(shape=(window, 1), name="vibration")
    x = layers.Conv1D(8, 9, strides=2, padding="same", activation="relu")(inp)   # 128
    x = layers.Conv1D(16, 9, strides=2, padding="same", activation="relu")(x)    # 64
    x = layers.Conv1D(16, 9, strides=2, padding="same", activation="relu")(x)    # 32
    code = layers.Conv1D(1, 1, activation=None, name="code")(x)                  # 32 x 1

    x = layers.Conv1DTranspose(16, 9, strides=2, padding="same", activation="relu")(code)  # 64
    x = layers.Conv1DTranspose(16, 9, strides=2, padding="same", activation="relu")(x)     # 128
    x = layers.Conv1DTranspose(8, 9, strides=2, padding="same", activation="relu")(x)      # 256
    out = layers.Conv1D(1, 1, activation=None, name="reconstruction")(x)
    model = keras.Model(inp, out, name="vib_autoencoder")
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def reconstruction_error(model: keras.Model, x: "tf.Tensor | object", batch: int = 512):
    """Per-window mean-squared reconstruction error."""
    import numpy as np
    x = np.asarray(x, dtype="float32")[..., None]
    rec = model.predict(x, batch_size=batch, verbose=0)
    return np.mean((rec - x) ** 2, axis=(1, 2))
