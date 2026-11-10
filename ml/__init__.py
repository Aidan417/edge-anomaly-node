"""Edge anomaly node — ML pipeline (phase 1).

Trains a small 1D convolutional autoencoder on healthy bearing vibration,
evaluates it as an anomaly detector against real fault recordings, and
exports an int8 TFLite model small enough for a microcontroller.
"""
