import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

MODEL_PATH = r"C:\Users\allen\Downloads\potato\model_potato_disease.keras"

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

CLASS_NAMES = ["Early_Blight", "Healthy", "Late_Blight"]


def predict(img_path):
    img = image.load_img(img_path, target_size=(160, 160))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    img = tf.keras.applications.efficientnet.preprocess_input(img)

    preds = model.predict(img)

    return CLASS_NAMES[np.argmax(preds)]


# Example
print(predict(r"C:\Users\allen\Downloads\potato\testing\Healthy\image.jpg"))