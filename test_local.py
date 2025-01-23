import os
import re
import numpy as np
import pandas as pd
from tabulate import tabulate
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def parse_fileinfo(fname):
    base = os.path.splitext(fname)[0]
    parts = base.split("_")
    if len(parts)<3:
        return ("unknown","unknown","unknown")
    age_cat, gen_str, expr_str = parts[0], parts[1], parts[2]
    expr_str = re.sub(r'\d+','', expr_str.lower())
    return (age_cat.lower(), gen_str.lower(), expr_str)

def load_image(path):
    img = load_img(path, target_size=(64,64))
    arr = img_to_array(img)/255.0
    return np.expand_dims(arr,0)

def main():
    model = tf.keras.models.load_model("my_multiout.h5")
    folder = "data"
    results = []
    for fname in os.listdir(folder):
        if not fname.lower().endswith((".jpg",".jpeg",".png")):
            continue
        path = os.path.join(folder, fname)
        # parse ground truth from name
        true_agecat, true_gender, true_expr = parse_fileinfo(fname)

        # predict
        x_in = load_image(path)
        preds = model.predict(x_in)
        pred_age     = preds[0][0][0]
        pred_gender  = preds[1][0] # shape(2,)
        pred_expr    = preds[2][0] # shape(2,)

        # interpret
        guessed_agecat = "old" if pred_age>0.5 else "young"
        g_idx = np.argmax(pred_gender)
        guessed_gender = "female" if g_idx==0 else "male"
        e_idx = np.argmax(pred_expr)
        guessed_expr   = "happy" if e_idx==0 else "sad"

        results.append({
            "filename": fname,
            "true_agecat": true_agecat,
            "pred_agecat": guessed_agecat,
            "age_correct": (guessed_agecat == true_agecat),
            "true_gender": true_gender,
            "pred_gender": guessed_gender,
            "gender_correct": (guessed_gender == true_gender),
            "true_expr": true_expr,
            "pred_expr": guessed_expr,
            "expr_correct": (guessed_expr == true_expr),
            "numeric_age_est": float(pred_age)
        })

    df = pd.DataFrame(results)
    print(tabulate(df, headers='keys', tablefmt='psql'))
    age_acc  = df["age_correct"].mean()*100
    gen_acc  = df["gender_correct"].mean()*100
    expr_acc = df["expr_correct"].mean()*100
    print(f"\nAccuracy: Age={age_acc:.1f}%, Gender={gen_acc:.1f}%, Expr={expr_acc:.1f}%")

if __name__=="__main__":
    main()
