import pandas as pd
import numpy as np
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==============================================================================
# Step 3.2: Load Data and Preprocessing
# ==============================================================================

try:
    data = pd.read_csv('Crop_recommendation.csv')
except FileNotFoundError:
    print("Error: 'Crop_recommendation.csv' not found.")
    sys.exit()

feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
nitrogen_col = ['N']
other_feature_cols = ['P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']

# --- VALIDATION BLOCK ---
invalid_n = data[data[nitrogen_col[0]] < 0]
invalid_others = data[(data[other_feature_cols] <= 0).any(axis=1)]

if not invalid_n.empty or not invalid_others.empty:
    print("\n--- ERROR: INVALID DATA DETECTED ---")
    if not invalid_n.empty: print(f"Found {len(invalid_n)} rows where Nitrogen (N) is negative.")
    if not invalid_others.empty: print(f"Found {len(invalid_others)} rows where other factors are 0 or negative.")
    sys.exit()

X = data[feature_cols]
y = data['label']

le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(np.unique(y_encoded))

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# ==============================================================================
# Step 3.3: Hybrid Feature Extraction (RF Probabilities)
# ==============================================================================

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

X_train_rf_features = rf_model.predict_proba(X_train_scaled)
X_val_rf_features = rf_model.predict_proba(X_val_scaled)
X_test_rf_features = rf_model.predict_proba(X_test_scaled)

X_train_hybrid = np.hstack((X_train_scaled, X_train_rf_features))
X_val_hybrid = np.hstack((X_val_scaled, X_val_rf_features))
X_test_hybrid = np.hstack((X_test_scaled, X_test_rf_features))

y_train_one_hot = to_categorical(y_train, num_classes=num_classes)
y_val_one_hot = to_categorical(y_val, num_classes=num_classes)

# ==============================================================================
# Step 3.4: Train Hybrid ANN
# ==============================================================================

ann_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_hybrid.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax') 
])

ann_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("\nTraining the Hybrid RF+ANN Model...")
history = ann_model.fit(
    X_train_hybrid, y_train_one_hot,
    validation_data=(X_val_hybrid, y_val_one_hot),
    epochs=50, batch_size=32, verbose=1 
)

# ==============================================================================
# Step 3.5: Comprehensive Evaluation & Visualization
# ==============================================================================

# 1. Final Accuracy Result
hybrid_loss, hybrid_accuracy = ann_model.evaluate(X_test_hybrid, to_categorical(y_test, num_classes=num_classes), verbose=0)
print(f"\n--- Model Evaluation ---")
print(f"Hybrid RF+ANN Model Test Accuracy: {hybrid_accuracy:.4f}")

# 2. Plotting Training History (Loss and Accuracy)
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Hybrid Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Hybrid Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 3. Confusion Matrix
y_pred_prob = ann_model.predict(X_test_hybrid)
y_pred = np.argmax(y_pred_prob, axis=1)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
plt.title('Confusion Matrix: Hybrid RF+ANN')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# 4. Classification Report
print("\n--- Detailed Classification Report ---")
print(classification_report(y_test, y_pred, target_names=le.classes_))
import joblib

# Save scaler
joblib.dump(scaler, "scaler.pkl")

# Save Random Forest
joblib.dump(rf_model, "rf_model.pkl")

# Save label encoder
joblib.dump(le, "label_encoder.pkl")

# Save ANN model
ann_model.save("crop_ann_model.keras")
