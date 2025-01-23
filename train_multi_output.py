# train_multi_output.py
import os
import re
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def parse_filename(filename):
    base = os.path.splitext(filename)[0]  # "old_female_happy1"
    parts = base.split("_")  # ["old","female","happy1"]
    if len(parts) < 3:
        return None

    age_str = parts[0].lower()
    if age_str not in ["old", "young"]:
        return None
    # Convert: old=1.0, young=0.0
    age_val = 1.0 if age_str == "old" else 0.0

    gender_str = parts[1].lower()
    if gender_str not in ["female", "male"]:
        return None
    # female= [1,0], male=[0,1]
    if gender_str == "female":
        gender_oh = np.array([1., 0.], dtype=np.float32)
    else:
        gender_oh = np.array([0., 1.], dtype=np.float32)

    expr_str = re.sub(r'\d+', '', parts[2].lower())  # "happy1" => "happy"
    if expr_str not in ["happy", "sad"]:
        return None
    expr_oh = np.array([1., 0.]) if expr_str == "happy" else np.array([0., 1.])

    return age_val, gender_oh, expr_oh


def load_data(folder="data", target_size=(64, 64)):
    X, y_age, y_gender, y_expr = [], [], [], []
    exts = (".jpg", ".jpeg", ".png")

    for fname in os.listdir(folder):
        if not fname.lower().endswith(exts):
            continue

        parsed = parse_filename(fname)
        if parsed is None:
            print("Skipping", fname)
            continue
        age_val, g_oh, e_oh = parsed

        path = os.path.join(folder, fname)
        try:
            img = load_img(path, target_size=target_size)
            arr = img_to_array(img) / 255.0
        except Exception as exc:
            print(f"Skipping {fname} due to error: {exc}")
            continue

        X.append(arr)
        y_age.append(age_val)
        y_gender.append(g_oh)
        y_expr.append(e_oh)

    X = np.array(X, dtype=np.float32)
    y_age = np.array(y_age, dtype=np.float32).reshape((-1, 1))
    y_gender = np.array(y_gender, dtype=np.float32)
    y_expr = np.array(y_expr, dtype=np.float32)

    print(f"Loaded {len(X)} images from {folder}")
    return X, y_age, y_gender, y_expr


def create_model(input_shape=(64, 64, 3)):
    inp = Input(shape=input_shape)
    x = layers.Conv2D(16, (3, 3), activation='relu')(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, (3, 3), activation='relu')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)

    out_age = layers.Dense(1, name="out_age")(x)  # regression
    out_gender = layers.Dense(2, activation='softmax', name="out_gender")(x)  # female/male
    out_expr = layers.Dense(2, activation='softmax', name="out_expr")(x)  # happy/sad

    model = Model(inp, [out_age, out_gender, out_expr])
    return model


def main():
    # 1) Load data
    X, y_age, y_gen, y_expr = load_data("data", (64, 64))
    if len(X) == 0:
        print("No images found in 'data' folder. Aborting.")
        return

    # 2) Create model
    model = create_model((64, 64, 3))
    model.compile(
        optimizer='adam',
        loss={
            "out_age": "mse",
            "out_gender": "categorical_crossentropy",
            "out_expr": "categorical_crossentropy"
        }
    )
    model.summary()

    # 3) Train
    model.fit(
        X, {
            "out_age": y_age,
            "out_gender": y_gen,
            "out_expr": y_expr
        },
        epochs=5,
        batch_size=8
    )

    # 4) Save
    model.save("my_multiout.h5")
    print("Saved model to my_multiout.h5")


if __name__ == "__main__":
    main()
