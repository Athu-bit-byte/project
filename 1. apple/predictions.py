import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

MODEL_PATH = r"C:\Users\allen\Downloads\Original Data\model_apple_disease.keras"

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# If you know class names, put them here (order matters)
CLASS_NAMES = None  # or ['Apple___Healthy', 'Apple___Scab', ...]

def predict(img_path):
    img = image.load_img(img_path, target_size=(160, 160))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    img = tf.keras.applications.efficientnet.preprocess_input(img)

    preds = model.predict(img)
    class_id = np.argmax(preds)

    if CLASS_NAMES:
        return CLASS_NAMES[class_id]
    return class_id
