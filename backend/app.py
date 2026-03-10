from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import sqlite3
import hashlib
import os
import uuid
import tempfile
import joblib
import requests
from datetime import datetime

import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image

app = Flask(__name__, template_folder='templates', static_folder='static')
app.secret_key = 'farmone_secret_key_123'
CORS(app, supports_credentials=True)

DB_PATH  = os.path.join(os.path.dirname(__file__), 'farmone.db')
BASE_DIR = os.path.dirname(os.path.dirname(__file__))   # mainpro/
CROP_REC_DIR = os.path.join(BASE_DIR, 'crop_rec')

# ═══════════════════════════════════════════════════════════════════════════════
# DISEASE DETECTION — Model registry
# ═══════════════════════════════════════════════════════════════════════════════

# Class names sorted alphabetically — must match how training split was created
CROP_CLASSES = {
    'Apple':     sorted(['Healthy', 'Cedar Apple Rust', 'Black Rot', 'Apple Scab']),
    'Tomato':    sorted(['bacterial_spot', 'early_blight', 'healthy', 'late_blight',
                         'leaf_mold', 'mosaic_virus', 'septoria_leaf_spot',
                         'target_spot', 'twospotted_spider_mite', 'yellow_leaf_curl_virus']),
    'Potato':    sorted(['Early_Blight', 'Healthy', 'Late_Blight']),
    'Rice':      sorted(['bacterial_leaf_blight', 'brown_spot', 'healthy', 'leaf_blast',
                         'leaf_scald', 'narrow_brown_spot', 'neck_blast', 'rice_hispa',
                         'sheath_blight', 'tungro']),
    'Mango':     sorted(['Anthracnose', 'Bacterial Canker', 'Cutting Weevil', 'Die Back',
                         'Gall Midge', 'Healthy', 'Powdery Mildew', 'Sooty Mould']),
    'Banana':    sorted(['Cordana', 'Healthy', 'Panama Disease', 'Yellow and Black Sigatoka']),
    'Sugarcane': sorted(['Healthy', 'Mosaic', 'RedRot', 'Rust', 'Yellow']),
    'Cotton':    sorted(['Bacterial Blight', 'Curl Virus', 'Fussarium Wilt', 'Healthy']),
    'Pumpkin':   sorted(['Bacterial Leaf Spot', 'Downy Mildew', 'Healthy Leaf',
                         'Mosaic Disease', 'Powdery Mildew']),
}

MODEL_PATHS = {
    'Apple':     os.path.join(BASE_DIR, '1. apple',     'model_apple_disease.keras'),
    'Tomato':    os.path.join(BASE_DIR, '2. tomato',    'model_tomato_disease.keras'),
    'Potato':    os.path.join(BASE_DIR, '3. potato',    'model_potato_disease.keras'),
    'Rice':      os.path.join(BASE_DIR, '4. rice',      'model_rice_disease.keras'),
    'Mango':     os.path.join(BASE_DIR, '5. mango',     'model_mango_disease.keras'),
    'Banana':    os.path.join(BASE_DIR, '6. banana',    'model_banana_disease.keras'),
    'Sugarcane': os.path.join(BASE_DIR, '7. sugarcane', 'model_sugarcane_disease.keras'),
    'Cotton':    os.path.join(BASE_DIR, '8. cotton',    'model_cotton_disease.keras'),
    'Pumpkin':   os.path.join(BASE_DIR, '9. pumpkin',   'model_pumpkin_disease.keras'),
}

_disease_model_cache = {}

def get_disease_model(crop):
    path = MODEL_PATHS.get(crop)
    if not path or not os.path.exists(path):
        raise FileNotFoundError(f"Model not found for crop: {crop}")
    if crop not in _disease_model_cache:
        print(f"[INFO] Loading disease model for {crop}...")
        _disease_model_cache[crop] = tf.keras.models.load_model(path, compile=False)
        print(f"[INFO] {crop} model loaded.")
    return _disease_model_cache[crop]

IMG_SIZE = 224

def predict_disease(crop, img_path):
    model = get_disease_model(crop)
    img = keras_image.load_img(img_path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = keras_image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = tf.keras.applications.efficientnet.preprocess_input(arr)
    preds = model.predict(arr, verbose=0)
    class_id   = int(np.argmax(preds))
    confidence = float(np.max(preds))
    class_name = CROP_CLASSES[crop][class_id]
    return class_name, confidence

# ═══════════════════════════════════════════════════════════════════════════════
# CROP RECOMMENDATION — Hybrid RF + ANN model
# ═══════════════════════════════════════════════════════════════════════════════

_rec_models = {}   # lazy loaded

def get_rec_models():
    if not _rec_models:
        print("[INFO] Loading crop recommendation models...")
        scaler_path = os.path.join(CROP_REC_DIR, 'scaler.pkl')
        rf_path     = os.path.join(CROP_REC_DIR, 'rf_model.pkl')
        le_path     = os.path.join(CROP_REC_DIR, 'label_encoder.pkl')
        ann_path    = os.path.join(CROP_REC_DIR, 'crop_ann_model.keras')

        _rec_models['scaler'] = joblib.load(scaler_path)
        _rec_models['rf']     = joblib.load(rf_path)
        _rec_models['le']     = joblib.load(le_path)
        _rec_models['ann'] = tf.keras.models.load_model(ann_path, compile=False)
        print("[INFO] Crop recommendation models loaded.")
    return _rec_models

FEATURE_COLS = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall']
FEATURE_RANGES = {
    'N':           (0, 140),
    'P':           (5, 145),
    'K':           (5, 205),
    'temperature': (8, 44),
    'humidity':    (14, 100),
    'ph':          (3.5, 10.0),
    'rainfall':    (20, 300)
}

def predict_crop(features: dict):
    """
    features: dict with keys N, P, K, temperature, humidity, ph, rainfall
    Returns top-5 list of {crop, confidence} sorted by confidence desc.
    """
    models = get_rec_models()
    arr = np.array([[features[c] for c in FEATURE_COLS]], dtype=float)

    scaled     = models['scaler'].transform(arr)
    rf_probs   = models['rf'].predict_proba(scaled)
    hybrid_inp = np.hstack((scaled, rf_probs))

    ann_preds  = models['ann'].predict(hybrid_inp, verbose=0)[0]  # shape: (num_classes,)
    le         = models['le']

    # Build top-5 and re-normalise so they sum to 1.0 (avoids 0% display for
    # lower-ranked crops when the softmax is very peaked)
    top_indices   = np.argsort(ann_preds)[::-1][:5]
    top_probs     = ann_preds[top_indices]
    top_probs_sum = top_probs.sum()
    scaled_probs  = top_probs / top_probs_sum if top_probs_sum > 0 else top_probs

    results = []
    for i, idx in enumerate(top_indices):
        results.append({
            'crop':       le.inverse_transform([idx])[0].capitalize(),
            'confidence': round(float(scaled_probs[i]), 4)
        })
    return results

# ═══════════════════════════════════════════════════════════════════════════════
# DATABASE
# ═══════════════════════════════════════════════════════════════════════════════

def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    with get_db() as conn:
        conn.executescript('''
            CREATE TABLE IF NOT EXISTS users (
                id         TEXT PRIMARY KEY,
                name       TEXT NOT NULL,
                email      TEXT NOT NULL UNIQUE,
                password   TEXT NOT NULL,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP
            );
            CREATE TABLE IF NOT EXISTS history (
                id       INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id  TEXT NOT NULL,
                type     TEXT NOT NULL,
                crop     TEXT NOT NULL,
                result   TEXT NOT NULL,
                date     TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users(id)
            );
        ''')
        conn.commit()

init_db()


def hash_password(pw):
    return hashlib.sha256(pw.encode()).hexdigest()

def prettify(name):
    return name.replace('_', ' ').title()

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — Disease Detection
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/predict', methods=['POST'])
def predict():
    crop = request.form.get('crop', '').strip()
    if crop not in MODEL_PATHS:
        return jsonify({'error': f'Unknown crop: {crop}'}), 400

    file = request.files.get('image')
    if not file:
        return jsonify({'error': 'No image uploaded'}), 400

    suffix = os.path.splitext(file.filename)[-1] or '.jpg'
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    try:
        class_name, confidence = predict_disease(crop, tmp_path)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        try: os.unlink(tmp_path)
        except: pass

    is_healthy = 'healthy' in class_name.lower()
    return jsonify({
        'disease':    prettify(class_name),
        'confidence': round(confidence, 4),
        'isHealthy':  is_healthy,
    }), 200

@app.route('/api/status', methods=['GET'])
def status():
    return jsonify({
        'online':        True,
        'models_loaded': list(_disease_model_cache.keys())
    }), 200


# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/crop-recommend', methods=['POST'])
def crop_recommend():
    data = request.get_json()
    try:
        features = {}
        for col in FEATURE_COLS:
            val = float(data[col])
            min_val, max_val = FEATURE_RANGES[col]
            if val < min_val or val > max_val:
                return jsonify({'error': f'{col} must be between {min_val} and {max_val}'}), 400
            features[col] = val
    except (KeyError, TypeError, ValueError) as e:
        return jsonify({'error': f'Missing or invalid field: {e}'}), 400

    try:
        results = predict_crop(features)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'results': results}), 200

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — News proxy (avoids browser CORS restriction on NewsAPI free tier)
# ═══════════════════════════════════════════════════════════════════════════════

NEWS_API_KEY = '238fb859f0034a4a90149ecd3954ebec'

@app.route('/api/news', methods=['GET'])
def news_proxy():
    query     = request.args.get('q', 'agriculture farming crop')
    page_size = request.args.get('pageSize', 10)
    try:
        url = 'https://newsapi.org/v2/everything'
        params = {
            'q':        query,
            'sortBy':   'publishedAt',
            'language': 'en',
            'pageSize': page_size,
            'apiKey':   NEWS_API_KEY,
        }
        resp = requests.get(url, params=params, timeout=10)
        data = resp.json()
        return jsonify(data), resp.status_code
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — UI Pages
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/tool')
def tool():
    return render_template('tool.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/signup')
def signup_page():
    return render_template('signup.html')

@app.route('/history')
def history_page():
    return render_template('history.html')

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — Auth
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/signup', methods=['POST'])
def signup():
    data     = request.get_json()
    name     = (data.get('name') or '').strip()
    email    = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    if not name or not email or len(password) < 6:
        return jsonify({'error': 'Please fill all fields. Password must be 6+ characters.'}), 400
    user_id = str(uuid.uuid4())
    try:
        with get_db() as conn:
            conn.execute('INSERT INTO users (id, name, email, password) VALUES (?, ?, ?, ?)',
                         (user_id, name, email, hash_password(password)))
            conn.commit()
    except sqlite3.IntegrityError:
        return jsonify({'error': 'An account with this email already exists.'}), 409
    session['user'] = {'id': user_id, 'name': name, 'email': email}
    return jsonify({'id': user_id, 'name': name, 'email': email}), 201


@app.route('/api/login', methods=['POST'])
def login():
    data     = request.get_json()
    email    = (data.get('email') or '').strip().lower()
    password = data.get('password') or ''
    if not email or not password:
        return jsonify({'error': 'Email and password are required.'}), 400
    with get_db() as conn:
        user = conn.execute(
            'SELECT id, name, email FROM users WHERE email = ? AND password = ?',
            (email, hash_password(password))
        ).fetchone()
    if not user:
        return jsonify({'error': 'Invalid email or password.'}), 401
    session['user'] = {'id': user['id'], 'name': user['name'], 'email': user['email']}
    return jsonify({'id': user['id'], 'name': user['name'], 'email': user['email']}), 200

@app.route('/api/me', methods=['GET'])
def me():
    if 'user' in session:
        return jsonify(session['user']), 200
    return jsonify({'error': 'Not logged in'}), 401

@app.route('/api/logout', methods=['POST'])
def logout():
    session.pop('user', None)
    return jsonify({'message': 'Logged out'}), 200

# ═══════════════════════════════════════════════════════════════════════════════
# ROUTES — History
# ═══════════════════════════════════════════════════════════════════════════════

@app.route('/api/history/<user_id>', methods=['GET'])
def get_history(user_id):
    with get_db() as conn:
        rows = conn.execute(
            'SELECT * FROM history WHERE user_id = ? ORDER BY id DESC', (user_id,)
        ).fetchall()
    return jsonify([dict(r) for r in rows]), 200


@app.route('/api/history', methods=['POST'])
def add_history():
    data    = request.get_json()
    user_id = data.get('user_id')
    htype   = data.get('type')
    crop    = data.get('crop')
    result  = data.get('result')
    date    = data.get('date') or datetime.now().strftime('%Y-%m-%d')
    if not all([user_id, htype, crop, result]):
        return jsonify({'error': 'Missing required fields.'}), 400
    with get_db() as conn:
        conn.execute(
            'INSERT INTO history (user_id, type, crop, result, date) VALUES (?, ?, ?, ?, ?)',
            (user_id, htype, crop, result, date)
        )
        conn.commit()
    return jsonify({'message': 'History entry saved.'}), 201


@app.route('/api/history/<user_id>', methods=['DELETE'])
def clear_history(user_id):
    with get_db() as conn:
        conn.execute('DELETE FROM history WHERE user_id = ?', (user_id,))
        conn.commit()
    return jsonify({'message': 'History cleared.'}), 200


if __name__ == '__main__':
    app.run(debug=True, port=5000)
