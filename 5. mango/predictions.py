import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

MODEL_PATH = r"C:\Users\allen\Downloads\mango\model_mango_disease.keras"

CLASS_NAMES = [
    "Anthracnose",
    "Bacterial Canker",
    "Cutting Weevil",
    "Die Back",
    "Gall Midge",
    "Healthy",
    "Powdery Mildew",
    "Sooty Mould"
]

model = tf.keras.models.load_model(MODEL_PATH, compile=False)

def predict(img_path):
    img = image.load_img(img_path, target_size=(160, 160))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    preds = model.predict(img)
    return CLASS_NAMES[np.argmax(preds)]
