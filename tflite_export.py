"""
TFLite export for mobile deployment demonstration.

This project does NOT attempt a direct conversion of DeepFace.
Instead, it creates a small untrained demo TensorFlow model and converts it
to TensorFlow Lite so you can demonstrate mobile compatibility locally.
"""

from __future__ import annotations

from pathlib import Path

DEMO_LABEL = "demonstration for mobile deployment"
DEFAULT_INPUT_SHAPE = (64, 64, 3)


def build_demo_model(input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE):
    # Small untrained model: Conv -> pooling -> GAP -> Dense.
    # Output is not used for accuracy evaluation.
    import tensorflow as tf

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = tf.keras.layers.Conv2D(8, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(2, 2))(x)
    x = tf.keras.layers.Conv2D(16, 3, activation="relu", padding="same")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(4, activation="softmax", name="demo_output")(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs, name="demo_tflite_model")


def export_tflite(
    out_path: Path = Path("models/model.tflite"),
    input_shape: tuple[int, int, int] = DEFAULT_INPUT_SHAPE,
) -> Path:
    import tensorflow as tf

    model = build_demo_model(input_shape=input_shape)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Keep export simple and runnable on CPU-only machines.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(tflite_model)

    print(f"[tflite_export] Wrote: {out_path.resolve()}")
    print(f"[tflite_export] Label: {DEMO_LABEL}")
    return out_path


def main() -> int:
    export_tflite()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

