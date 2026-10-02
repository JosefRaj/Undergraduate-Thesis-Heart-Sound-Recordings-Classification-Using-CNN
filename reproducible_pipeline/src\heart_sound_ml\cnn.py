from __future__ import annotations


def build_spectrogram_cnn(input_shape: tuple[int, int, int] = (128, 128, 1)):
    """Build the optional CNN without making TensorFlow a core dependency."""
    try:
        from tensorflow import keras
    except ImportError as exc:
        raise RuntimeError("Install the optional TensorFlow environment to build the CNN") from exc
    model = keras.Sequential(
        [
            keras.layers.Input(shape=input_shape),
            keras.layers.Conv2D(16, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(32, 3, activation="relu", padding="same"),
            keras.layers.MaxPooling2D(),
            keras.layers.Conv2D(64, 3, activation="relu", padding="same"),
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model
