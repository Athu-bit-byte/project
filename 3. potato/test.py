import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

MODEL_PATH = r"C:\Users\allen\Downloads\potato\model_potato_disease.keras"
TEST_DIR = r"C:\Users\allen\Downloads\potato\testing"

CLASS_NAMES = [
    "Early_Blight",
    "Healthy",
    "Late_Blight"
]

CLASS_NAMES.sort()

print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

IMG_SIZE = 224


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    img = tf.keras.applications.efficientnet.preprocess_input(img)

    preds = model.predict(img, verbose=0)
    return np.argmax(preds)


y_true = []
y_pred = []

print("Running predictions...")

for class_index, class_name in enumerate(CLASS_NAMES):
    class_path = os.path.join(TEST_DIR, class_name)

    for img_name in os.listdir(class_path):
        if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(class_path, img_name)

            pred = predict_image(img_path)

            y_true.append(class_index)
            y_pred.append(pred)


acc = accuracy_score(y_true, y_pred)
print(f"\nTest Accuracy: {acc * 100:.2f}%\n")

print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d",
            xticklabels=CLASS_NAMES,
            yticklabels=CLASS_NAMES)

plt.title("Confusion Matrix")
plt.savefig(r"C:\Users\allen\Downloads\potato\confusion_matrix.png")
plt.show()
