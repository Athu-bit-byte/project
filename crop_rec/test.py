import pandas as pd
import numpy as np
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

# ==============================================================================
# 1. SETUP AND TRAINING
# ==============================================================================

try:
    # Load the dataset to calculate ranges and train the model
    data = pd.read_csv('Crop_recommendation.csv') 
except FileNotFoundError:
    print("Error: 'Crop_recommendation.csv' not found. Please ensure the file is in the same directory.")
    sys.exit(1)

feature_cols = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
X = data[feature_cols]
y = data['label']

# Dynamic range calculation for warnings (Sub-optimal = outer 10% edges)
ranges = {}
for col in feature_cols:
    c_min, c_max = data[col].min(), data[col].max()
    span = c_max - c_min
    ranges[col] = {
        'min': c_min, 
        'max': c_max,
        'low_threshold': c_min + (span * 0.10),
        'high_threshold': c_max - (span * 0.10)
    }

le = LabelEncoder()
y_encoded = le.fit_transform(y)
num_classes = len(np.unique(y_encoded))

# Split data (60% Train, 20% Val, 20% Test)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.25, random_state=42, stratify=y_train_full
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Hybrid Step: Train Random Forest to get probability features
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

X_train_rf_features = rf_model.predict_proba(X_train_scaled)
X_train_hybrid = np.hstack((X_train_scaled, X_train_rf_features))
y_train_one_hot = to_categorical(y_train, num_classes=num_classes)

# Define Hybrid ANN Architecture
ann_model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_hybrid.shape[1],)),
    Dense(64, activation='relu'),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])
ann_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
ann_model.fit(X_train_hybrid, y_train_one_hot, epochs=50, batch_size=32, verbose=0)

# ==============================================================================
# 2. VALIDATION WITH SUB-OPTIMAL WARNINGS
# ==============================================================================

def validate_and_warn(val, name, allow_zero=False):
    """Checks validity and issues warnings for extreme/sub-optimal values."""
    r = ranges[name]
    
    # Check if value is logically possible based on your rules
    if allow_zero:
        if val < 0:
            print(f"--- ERROR: {name} cannot be negative.")
            return False
    else:
        if val <= 0:
            print(f"--- ERROR: {name} must be greater than 0.")
            return False

    # Issue sub-optimal warnings for extreme edges of the dataset range
    if val < r['low_threshold']:
        print(f"   [!] WARNING: {name} is at a very LOW extreme. Conditions may be sub-optimal.")
    elif val > r['high_threshold']:
        print(f"   [!] WARNING: {name} is at a very HIGH extreme. Conditions may be sub-optimal.")
        
    return True

def get_user_input():
    print("\n" + "="*50)
    print(" CROP RECOMMENDATION SYSTEM - INPUT PANEL")
    print("="*50)
    
    inputs = {}
    try:
        # Nitrogen (Allows 0)
        val_n = float(input(f"Nitrogen N ({ranges['N']['min']:.1f}-{ranges['N']['max']:.1f}): "))
        if not validate_and_warn(val_n, 'N', allow_zero=True): return None
        inputs['N'] = val_n

        # Other features (Strictly > 0)
        for col in ['P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']:
            label = col.capitalize() if col != 'ph' else 'pH'
            prompt = f"{label} ({ranges[col]['min']:.1f}-{ranges[col]['max']:.1f}): "
            val = float(input(prompt))
            if not validate_and_warn(val, col): return None
            inputs[col] = val
            
        return inputs
    except ValueError:
        print("--- ERROR: Please enter numbers only.")
        return None

# ==============================================================================
# 3. INTERACTIVE TESTING
# ==============================================================================

while True:
    user_data = get_user_input()

    if user_data:
        input_df = pd.DataFrame([user_data])
        scaled_input = scaler.transform(input_df)
        rf_probs = rf_model.predict_proba(scaled_input)
        hybrid_input = np.hstack((scaled_input, rf_probs))
        
        preds = ann_model.predict(hybrid_input, verbose=0)
        idx = np.argmax(preds)
        
        crop = le.inverse_transform([idx])[0]
        confidence = preds[0][idx]
        
        print("\n" + "*"*50)
        print(f" RESULT: {crop.upper()}")
        print(f" CONFIDENCE: {confidence:.2%}")
        print("*"*50)
    
    if input("\nWould you like to test another sample? (y/n): ").lower() != 'y':
        break

print("\nSystem closed.")