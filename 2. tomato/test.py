import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# --------------------------------------------------
# PATHS
# --------------------------------------------------
MODEL_PATH = r"C:\Users\allen\Downloads\tomato\model_tomato_disease.keras"
TEST_DIR = r"C:\Users\allen\Downloads\tomato\test"


# --------------------------------------------------
# CLASS NAMES (MUST MATCH TRAINING)
# --------------------------------------------------
CLASS_NAMES = [
    "bacterial_spot",
    "early_blight",
    "healthy",
    "late_blight",
    "leaf_mold",
    "mosaic_virus",
    "septoria_leaf_spot",
    "target_spot",
    "twospotted_spider_mite",
    "yellow_leaf_curl_virus"
]

CLASS_NAMES.sort()

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

# --------------------------------------------------
# LOAD & PREDICT ONE IMAGE
# --------------------------------------------------
IMG_SIZE = 224

def predict_image(img_path):
    img = image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)

    # IMPORTANT: same preprocessing as training
    img = tf.keras.applications.efficientnet.preprocess_input(img)

    preds = model.predict(img, verbose=0)
    return np.argmax(preds)


# --------------------------------------------------
# RUN PREDICTIONS ON TEST SET
# --------------------------------------------------
y_true = []
y_pred = []

print("Running predictions on test images...")

for class_index, class_name in enumerate(CLASS_NAMES):
    class_path = os.path.join(TEST_DIR, class_name)

    for img_name in os.listdir(class_path):
        if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(class_path, img_name)

            pred = predict_image(img_path)

            y_true.append(class_index)
            y_pred.append(pred)

# --------------------------------------------------
# METRICS
# --------------------------------------------------
acc = accuracy_score(y_true, y_pred)
print(f"\nTest Accuracy: {acc * 100:.2f}%\n")

print("Classification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()

plt.savefig("confusion_matrix.png")
plt.show()
