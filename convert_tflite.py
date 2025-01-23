# convert_to_tflite.py
import tensorflow as tf

def main():
    model = tf.keras.models.load_model("my_multiout.h5")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open("MyTFLiteApp/my_multi_out.tflite", "wb") as f:
        f.write(tflite_model)
    print("Saved my_multi_out.tflite")

if __name__=="__main__":
    main()
