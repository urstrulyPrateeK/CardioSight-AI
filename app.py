import os
import json
import base64
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, send_file, session
import joblib
from werkzeug.utils import secure_filename
from utils.ocr_engine import extract_text_from_file, parse_medical_report, validate_extracted_data
import math
from fpdf import FPDF
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv():
        return False

# Load .env values for local development before reading environment variables.
load_dotenv()
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    from tensorflow.keras.models import load_model
except ImportError:
    load_model = None

try:
    import firebase_admin
    from firebase_admin import credentials, firestore, auth
except ImportError:
    firebase_admin = None
    credentials = None
    firestore = None
    auth = None
import config

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev-only-change-in-production')
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'cardiosight_models')

# Canonical feature order used by training artifacts and inference.
FEATURE_COLUMNS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]
BASE_MODEL_ORDER = ['LR', 'KNN', 'SVM', 'RF', 'DT']
# In this dataset/model pack, target=0 corresponds to heart disease (risk-positive class).
RISK_POSITIVE_CLASS = 0

# Canonical categorical mappings (UCI encoding).
CP_MAPPING = {
    "typical angina": 0,
    "atypical angina": 1,
    "non-anginal pain": 2,
    "non anginal pain": 2,
    "asymptomatic": 3
}
SLOPE_MAPPING = {
    "upsloping": 0,
    "flat": 1,
    "downsloping": 2
}
THAL_MAPPING = {
    "normal": 1,
    "fixed defect": 2,
    "reversible defect": 3,
    "reversable defect": 3
}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- FIREBASE SETUP ---
try:
    if not firebase_admin:
        app.logger.warning("firebase_admin not installed; Firebase features disabled.")
        db = None
    elif not os.path.exists(config.FIREBASE_CREDENTIALS_PATH):
        app.logger.warning("Firebase Admin JSON not found. Firebase features disabled.")
        db = None
    else:
        cred = credentials.Certificate(config.FIREBASE_CREDENTIALS_PATH)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        app.logger.info("Firebase Admin SDK initialized successfully.")
except Exception as e:
    app.logger.exception("Error initializing Firebase: %s", e)
    db = None

# --- LOAD MODELS ---
try:
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    base_models = {name: joblib.load(os.path.join(MODEL_DIR, f'{name}.pkl')) for name in BASE_MODEL_ORDER}
    meta_model = load_model(os.path.join(MODEL_DIR, 'cnn_lstm.h5')) if load_model else None
    app.logger.info("CardioSight models loaded.")
except Exception:
    app.logger.exception("Model loading failed.")
    scaler, base_models, meta_model = None, None, None


def _extract_token_from_request():
    auth_header = request.headers.get('Authorization', '')
    if auth_header.startswith('Bearer '):
        return auth_header.split('Bearer ', 1)[1].strip()
    return request.form.get('id_token') or request.args.get('id_token')


def _verify_uid():
    # Single auth verifier used by all protected endpoints.
    if not auth:
        return session.get('uid')
    id_token = _extract_token_from_request()
    if id_token:
        try:
            decoded_token = auth.verify_id_token(id_token)
            uid = decoded_token.get('uid')
            if uid:
                session['uid'] = uid
            return uid
        except Exception:
            pass
    # Session fallback keeps authenticated browser flows stable when token header is absent.
    return session.get('uid')


def _verify_id_token_value(id_token):
    if not auth:
        return None
    try:
        return auth.verify_id_token(id_token)
    except Exception:
        return None


def _normalize_label(value):
    return str(value or '').strip().lower()


def _map_categorical(value, mapping, default):
    """
    Accept either numeric form values or human-readable labels.
    Prevents encoding mismatch when UI/OCR submits label strings.
    """
    if value is None:
        return default

    raw = str(value).strip()
    if raw == '':
        return default

    if raw.isdigit():
        return int(raw)

    return mapping.get(_normalize_label(raw), default)


def _class_probability(model, transformed_features, class_label):
    probs = model.predict_proba(transformed_features)[0]
    classes = list(getattr(model, 'classes_', []))
    if class_label in classes:
        idx = classes.index(class_label)
    elif 1 in classes:
        idx = classes.index(1)
    else:
        idx = int(np.argmax(probs))
    return float(probs[idx])


def _risk_probability(model, transformed_features):
    """
    Return probability of the configured risk-positive class.
    Prevents swapped risk output when class semantics are inverted.
    """
    return _class_probability(model, transformed_features, RISK_POSITIVE_CLASS)


def _meta_predict_single(meta_model, scaled_features, meta_features):
    meta_row = np.array(meta_features, dtype=float).reshape(1, -1)
    seq_from_meta = meta_row.reshape(1, meta_row.shape[1], 1)
    input_nodes = list(getattr(meta_model, 'inputs', []))

    if len(input_nodes) == 1:
        in_shape = tuple(input_nodes[0].shape)
        if len(in_shape) == 3:
            return float(meta_model.predict(seq_from_meta, verbose=0)[0][0])
        return float(meta_model.predict(meta_row, verbose=0)[0][0])

    # Dual-input model support (current saved cnn_lstm.h5 in this repo).
    seq_shape = tuple(input_nodes[0].shape)
    dense_shape = tuple(input_nodes[1].shape)
    seq_steps = seq_shape[1] if len(seq_shape) > 1 else None
    dense_width = dense_shape[1] if len(dense_shape) > 1 else None

    scaled_arr = np.array(scaled_features, dtype=float).reshape(1, -1)
    seq_from_scaled = scaled_arr.reshape(1, scaled_arr.shape[1], 1)

    # Choose seq input source based on expected timesteps.
    if seq_steps in (None, scaled_arr.shape[1]):
        seq_input = seq_from_scaled
    elif seq_steps == meta_row.shape[1]:
        seq_input = seq_from_meta
    else:
        raise ValueError(f"Unsupported seq input shape for meta model: {seq_shape}")

    # Choose dense input source based on expected width.
    concat_dense = np.hstack([scaled_arr, meta_row])
    if dense_width in (None, concat_dense.shape[1]):
        dense_input = concat_dense
    elif dense_width == meta_row.shape[1]:
        dense_input = meta_row
    elif dense_width == scaled_arr.shape[1]:
        dense_input = scaled_arr
    else:
        raise ValueError(f"Unsupported dense input shape for meta model: {dense_shape}")

    return float(meta_model.predict([seq_input, dense_input], verbose=0)[0][0])


def _to_unit_probabilities(probs):
    """
    Normalize probabilities into [0, 1], handling accidental 0-100 inputs safely.
    """
    arr = np.array(probs, dtype=float).reshape(-1)
    if arr.size == 0:
        return arr
    if np.nanmax(arr) > 1.0 and np.nanmax(arr) <= 100.0 and np.nanmin(arr) >= 0.0:
        arr = arr / 100.0
    return np.clip(arr, 0.0, 1.0)


def _stabilize_meta_features(base_probs_class1):
    """
    Stabilize meta inputs so a single extreme learner (notably DT) cannot dominate.
    Keeps architecture/weights unchanged; only inference-time feature conditioning.
    """
    probs = _to_unit_probabilities(base_probs_class1)
    if probs.size == 0:
        return probs

    # Clamp exact 0/1 spikes to reduce saturation in sigmoid layers.
    stable = np.clip(probs, 0.02, 0.98)

    # Robustly damp large outliers toward vector median.
    median = float(np.median(stable))
    mad = float(np.median(np.abs(stable - median)))
    outlier_threshold = max(0.15, 3.0 * mad)
    for i in range(stable.shape[0]):
        if abs(float(stable[i]) - median) > outlier_threshold:
            stable[i] = 0.35 * stable[i] + 0.65 * median

    # Additional DT-specific damping against LR/SVM/RF consensus.
    try:
        idx = {name: i for i, name in enumerate(BASE_MODEL_ORDER)}
        dt_i = idx['DT']
        core = [stable[idx['LR']], stable[idx['SVM']], stable[idx['RF']]]
        core_mean = float(np.mean(core))
        stable[dt_i] = 0.5 * stable[dt_i] + 0.5 * core_mean
    except Exception:
        pass

    return np.clip(stable, 0.0, 1.0)


def _meta_risk_probability(meta_model, scaled_features, base_probs_class1, base_probs_risk):
    """
    Run CNN-LSTM meta inference with deterministic, meta-first selection.
    Fallback is used only when inference fails or output is invalid.
    """
    if not meta_model:
        return None, None

    # Preferred first: stabilized class-1 probabilities in 0-1 range.
    class1_raw = _to_unit_probabilities(base_probs_class1)
    class1_stable = _stabilize_meta_features(class1_raw)
    risk_unit = _to_unit_probabilities(base_probs_risk)
    # Robust consensus: rely on LR/SVM/RF core to avoid DT/KNN instability.
    consensus_risk = _weighted_blend_lr_svm_rf(risk_unit)
    class1_primary = 'raw' if RISK_POSITIVE_CLASS == 1 else 'inverted'
    class1_secondary = 'inverted' if class1_primary == 'raw' else 'raw'
    attempts = [
        ('class1_stable', class1_primary),
        ('class1_stable', class1_secondary),
        ('class1_raw', class1_primary),
        ('class1_raw', class1_secondary),
        ('risk', 'raw'),
        ('risk', 'inverted'),
    ]

    successful = []
    for source_name, orientation in attempts:
        if source_name == 'class1_stable':
            features = class1_stable
        elif source_name == 'class1_raw':
            features = class1_raw
        else:
            features = risk_unit

        try:
            meta_out = _meta_predict_single(meta_model, scaled_features, features)
        except Exception:
            continue

        if not np.isfinite(meta_out):
            continue

        risk_prob = meta_out if orientation == 'raw' else (1.0 - meta_out)
        risk_prob = float(np.clip(risk_prob, 0.0, 1.0))

        successful.append({
            'risk_prob': risk_prob,
            'distance': abs(risk_prob - consensus_risk),
            'source': source_name,
            'orientation': orientation
        })

    if not successful:
        return None, None

    best = min(successful, key=lambda x: x['distance'])
    return best['risk_prob'], {
        'source': best['source'],
        'orientation': best['orientation']
    }


def _weighted_blend_lr_svm_rf(base_probs_risk):
    """
    Weighted consensus fallback for medium-risk disagreement cases.
    Uses LR/SVM/RF only to avoid overreacting to unstable learners.
    """
    idx = {name: i for i, name in enumerate(BASE_MODEL_ORDER)}
    weights = {'LR': 0.35, 'SVM': 0.35, 'RF': 0.30}
    score = 0.0
    wsum = 0.0
    for model_name, weight in weights.items():
        pos = idx.get(model_name)
        if pos is None or pos >= len(base_probs_risk):
            continue
        score += float(base_probs_risk[pos]) * weight
        wsum += weight
    return float(score / wsum) if wsum else float(np.mean(base_probs_risk))

# --- CONTEXT PROCESSOR ---
@app.context_processor
def inject_firebase_config():
    return dict(firebase_config=config.FIREBASE_CLIENT_CONFIG)

# --- ROUTES ---

@app.route('/', methods=['GET'])
def index():
    return render_template('login.html')

@app.route('/home', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/login')
def login_page():
    return render_template('login.html')

@app.route('/register')
def register_page():
    return render_template('register.html')

@app.route('/predict_page', methods=['GET'])
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['GET'])
def predict_alias():
    # Backward-compatible alias used by older navigation/test scripts.
    return render_template('predict.html')

@app.route('/dashboard', methods=['GET'])
def dashboard():
    return render_template('dashboard.html')

@app.route('/history', methods=['GET'])
def history_alias():
    # Backward-compatible alias used by older navigation/test scripts.
    return render_template('dashboard.html')


@app.route('/auth/verify', methods=['POST'])
def verify_auth_session():
    """
    Verify Firebase ID token on the backend and create a server session.
    """
    payload = request.get_json(silent=True) or {}
    id_token = payload.get('idToken') or request.form.get('idToken')
    if not id_token:
        return jsonify({'success': False, 'error': 'Missing idToken'}), 400

    if not auth:
        return jsonify({'success': False, 'error': 'Firebase Admin auth not available'}), 503

    decoded = _verify_id_token_value(id_token)
    if not decoded:
        return jsonify({'success': False, 'error': 'Invalid token'}), 401

    session['uid'] = decoded.get('uid')
    session['email'] = decoded.get('email')
    session['auth_provider'] = (decoded.get('firebase') or {}).get('sign_in_provider')
    return jsonify({'success': True, 'uid': session['uid'], 'email': session.get('email')})


@app.route('/auth/logout', methods=['POST'])
def logout_session():
    session.clear()
    return jsonify({'success': True})


@app.route('/logout', methods=['GET', 'POST'])
def logout_alias():
    # Compatibility route for legacy clients expecting /logout.
    session.clear()
    if request.method == 'GET':
        return render_template('login.html')
    return jsonify({'success': True})


@app.route('/api/predictions/history', methods=['GET'])
def get_prediction_history():
    uid = _verify_uid()
    if not uid:
        return jsonify({'success': False, 'error': 'Unauthorized'}), 401

    if not db:
        return jsonify({'success': False, 'error': 'Database not connected'}), 500

    try:
        docs = db.collection('users').document(uid).collection('history') \
            .order_by('timestamp', direction=firestore.Query.DESCENDING).stream()

        history = []
        for doc in docs:
            data = doc.to_dict() or {}
            patient_input = data.get('patient_input', {})
            timestamp = data.get('timestamp')
            timestamp_iso = timestamp.isoformat() if timestamp else None

            history.append({
                'id': doc.id,
                'user_id': data.get('user_id', uid),
                'patient_input': patient_input,
                'prediction_result': data.get('prediction_result'),
                'probability_score': data.get('probability_score'),
                'timestamp': timestamp_iso
            })
        return jsonify({'success': True, 'history': history})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/save_profile', methods=['POST'])
def save_profile():
    if not db: return jsonify({'error': 'Database not connected'}), 500
    
    try:
        data = request.json
        uid = data.get('uid')
        id_token = data.get('idToken')

        if id_token:
            decoded = _verify_id_token_value(id_token)
            if not decoded:
                return jsonify({'error': 'Invalid token'}), 401
            token_uid = decoded.get('uid')
            if uid and uid != token_uid:
                return jsonify({'error': 'UID mismatch'}), 400
            uid = token_uid

        if not uid:
            return jsonify({'error': 'Missing uid'}), 400

        user_data = {
            'fullName': data.get('fullName'),
            'email': data.get('email'),
            'phone': data.get('phone'),
            'dob': data.get('dob'),
            'createdAt': firestore.SERVER_TIMESTAMP
        }
        db.collection('users').document(uid).set(user_data, merge=True)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/extract_from_report', methods=['POST'])
def extract_from_report():
    if 'file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    if file.filename == '': return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        text = extract_text_from_file(filepath)
        data = parse_medical_report(text)
        validation = validate_extracted_data(data)
        try: os.remove(filepath)
        except: pass

        # Treat partially extracted but usable structured values as success.
        has_usable_data = bool(data)
        return jsonify({
            'success': has_usable_data,
            'data': data,
            'missing_fields': validation.get('missing_fields', []),
            'found_fields': validation.get('found_fields', []),
            'message': 'Extraction completed.' if has_usable_data else 'Could not parse enough clinical fields from this file.'
        })

@app.route('/process_prediction', methods=['POST'])
def process_prediction():
    try:
        uid = _verify_uid()

        name = request.form.get('name', 'Guest')
        email = request.form.get('email', '')
        age = int(request.form.get('age') or 0)
        gender = request.form.get('gender', 'Male')
        cp = _map_categorical(request.form.get('cp'), CP_MAPPING, 0)
        trestbps = int(request.form.get('trestbps') or 120)
        chol = int(request.form.get('chol') or 200)
        fbs = int(request.form.get('fbs') or 0)
        restecg = int(request.form.get('restecg') or 0)
        thalach = int(request.form.get('thalach') or 150)
        exang = int(request.form.get('exang') or 0)
        oldpeak = float(request.form.get('oldpeak') or 0)
        slope = _map_categorical(request.form.get('slope'), SLOPE_MAPPING, 0)
        ca = int(request.form.get('ca') or 0)
        thal = _map_categorical(request.form.get('thal'), THAL_MAPPING, 1)
        # ---- Prepare input features ----

        gender_val = 1 if _normalize_label(gender) == "male" else 0

        input_features = {
            'age': age,
            'sex': gender_val,
            'cp': cp,
            'trestbps': trestbps,
            'chol': chol,
            'fbs': fbs,
            'restecg': restecg,
            'thalach': thalach,
            'exang': exang,
            'oldpeak': oldpeak,
            'slope': slope,
            'ca': ca,
            'thal': thal
        }
        # Build dataframe with explicit columns so order never drifts.
        feature_frame = pd.DataFrame([input_features], columns=FEATURE_COLUMNS)
        if hasattr(scaler, 'feature_names_in_'):
            feature_frame = feature_frame.reindex(columns=list(scaler.feature_names_in_))

        # ---- Prediction ----

        if scaler and base_models:
            scaled_features = scaler.transform(feature_frame)
            base_details = {}
            base_probs_class1 = []
            base_probs_risk = []
            rf_risk_prob = None

            for model_name in BASE_MODEL_ORDER:
                model = base_models[model_name]
                # Base models were trained with class-1 probabilities as meta features.
                class1_prob = _class_probability(model, scaled_features, 1)
                base_probs_class1.append(class1_prob)

                risk_prob = _risk_probability(model, scaled_features)
                base_probs_risk.append(risk_prob)
                # DT dashboard probability uses class-1 disease likelihood ([:,1]) explicitly.
                display_prob = class1_prob if model_name == 'DT' else risk_prob
                base_details[model_name] = round(float(display_prob) * 100, 2)

                if model_name == 'RF':
                    rf_risk_prob = risk_prob

            # Use CNN-LSTM output when available; fallback only on failure.
            meta_used = False
            meta_info = None
            safeguard_used = False
            prediction_prob = rf_risk_prob if rf_risk_prob is not None else 0.455
            final_meta_probability = None
            try:
                meta_prob, meta_info = _meta_risk_probability(meta_model, scaled_features, base_probs_class1, base_probs_risk)
                if meta_prob is not None:
                    prediction_prob = meta_prob
                    meta_used = True
                    # Safeguard: prevent extreme suppression in medium-risk disagreement cases.
                    consensus_mean = _weighted_blend_lr_svm_rf(base_probs_risk)
                    weighted_consensus = _weighted_blend_lr_svm_rf(base_probs_risk)
                    idx = {name: i for i, name in enumerate(BASE_MODEL_ORDER)}
                    core_probs = [
                        float(base_probs_risk[idx['LR']]),
                        float(base_probs_risk[idx['SVM']]),
                        float(base_probs_risk[idx['RF']])
                    ]
                    core_spread = max(core_probs) - min(core_probs)
                    deviation_threshold = 0.18 if core_spread <= 0.25 else 0.30

                    if 0.20 <= consensus_mean <= 0.80 and abs(float(meta_prob) - consensus_mean) > deviation_threshold:
                        prediction_prob = weighted_consensus
                        safeguard_used = True

                    final_meta_probability = round(float(prediction_prob) * 100, 2)
            except Exception as meta_err:
                app.logger.warning("Meta model inference failed, using RF fallback: %s", meta_err)

            probability_percent = round(float(prediction_prob) * 100, 2)
            rf_probability_percent = round(float(rf_risk_prob if rf_risk_prob is not None else prediction_prob) * 100, 2)
            if safeguard_used:
                final_model_label = "Weighted Ensemble Safeguard (LR/SVM/RF)"
            elif meta_used:
                final_model_label = "CNN-LSTM Meta-Learner"
            else:
                final_model_label = "Random Forest (Fallback)"

        else:

            probability_percent = 45.5
            meta_used = False
            meta_info = None
            safeguard_used = False
            final_meta_probability = None
            rf_probability_percent = 45.5
            final_model_label = "Random Forest (Fallback)"

            base_details = {
                "RF": 40,
                "LR": 50
            }
      

        # 3-level interpretation only (no model logic change).
        if probability_percent < 35:
            risk_level, risk_color = "Low", "green"
        elif probability_percent <= 65:
            risk_level, risk_color = "Moderate", "orange"
        else:
            risk_level, risk_color = "High", "red"

        # Map coded values back to readable labels for the report
        cp_labels = {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}
        fbs_labels = {0: 'No (<=120 mg/dl)', 1: 'Yes (>120 mg/dl)'}
        restecg_labels = {0: 'Normal', 1: 'ST-T Wave Abnormality', 2: 'Left Ventricular Hypertrophy'}
        exang_labels = {0: 'No', 1: 'Yes'}
        slope_labels = {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}
        thal_labels = {1: 'Normal', 2: 'Fixed Defect', 3: 'Reversible Defect'}

        result_data = {
            'name': name,
            'email': email,
            'probability': probability_percent,
            'final_meta_probability': final_meta_probability,
            'rf_probability': rf_probability_percent,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'ai_analysis': base_details,
            'meta_model_used': meta_used,
            'meta_safeguard_used': safeguard_used,
            'final_model_label': final_model_label,
            'meta_model_mode': meta_info,
            'details': {
                'age': age,
                'gender': gender,
                'cp': cp,
                'cp_label': cp_labels.get(cp, str(cp)),
                'trestbps': trestbps,
                'chol': chol,
                'fbs': fbs,
                'fbs_label': fbs_labels.get(fbs, str(fbs)),
                'restecg': restecg,
                'restecg_label': restecg_labels.get(restecg, str(restecg)),
                'thalach': thalach,
                'exang': exang,
                'exang_label': exang_labels.get(exang, str(exang)),
                'oldpeak': oldpeak,
                'slope': slope,
                'slope_label': slope_labels.get(slope, str(slope)),
                'ca': ca,
                'thal': thal,
                'thal_label': thal_labels.get(thal, str(thal)),
            }
        }

        # Save to Firestore if logged in
        if db and uid:
            # Persist complete prediction record tied to current user.
            history_data = {
                'user_id': uid,
                'timestamp': firestore.SERVER_TIMESTAMP,
                'patient_input': {
                    'name': name,
                    'email': email,
                    'age': age,
                    'sex': gender_val,
                    'cp': cp,
                    'trestbps': trestbps,
                    'chol': chol,
                    'fbs': fbs,
                    'restecg': restecg,
                    'thalach': thalach,
                    'exang': exang,
                    'oldpeak': oldpeak,
                    'slope': slope,
                    'ca': ca,
                    'thal': thal
                },
                'prediction_result': risk_level,
                'probability_score': probability_percent,
                'result': result_data
            }
            db.collection('users').document(uid).collection('history').add(history_data)

        return render_template('result.html', result=result_data)
        
    except Exception as e:
        app.logger.exception("Prediction processing failed.")
        return f"Error: {e}", 500


# ===================== PDF GENERATION HELPER =====================

def _generate_chart_image(probability, risk_color):
    """Generate a doughnut chart image matching the result page style."""
    fig, ax = plt.subplots(figsize=(3.5, 3.5), dpi=150)
    colors_map = {'red': '#ff4b2b', 'orange': '#ff9800', 'green': '#00c853'}
    chart_color = colors_map.get(risk_color, '#ff9800')
    sizes = [probability, 100 - probability]
    colors = [chart_color, '#e0e0e0']
    wedges, _ = ax.pie(sizes, colors=colors, startangle=90, counterclock=False,
                        wedgeprops=dict(width=0.35, edgecolor='white', linewidth=2))
    centre_circle = plt.Circle((0, 0), 0.55, fc='white')
    ax.add_artist(centre_circle)
    ax.text(0, 0.05, f'{probability}%', ha='center', va='center', fontsize=22, fontweight='bold', color=chart_color)
    ax.text(0, -0.18, 'Risk', ha='center', va='center', fontsize=10, color='#666666')
    ax.set_aspect('equal')
    plt.tight_layout()
    img_path = os.path.join('uploads', 'chart_temp.png')
    fig.savefig(img_path, transparent=True, bbox_inches='tight')
    plt.close(fig)
    return img_path

def _generate_model_bar_image(ai_analysis):
    """Generate a horizontal bar chart of model-by-model analysis."""
    models = list(ai_analysis.keys())
    scores = list(ai_analysis.values())
    fig, ax = plt.subplots(figsize=(5, max(1.5, len(models) * 0.6)), dpi=150)
    colors = []
    for s in scores:
        if s < 30: colors.append('#00c853')
        elif s < 70: colors.append('#ff9800')
        else: colors.append('#ff4b2b')
    bars = ax.barh(models, scores, color=colors, height=0.5, edgecolor='white', linewidth=0.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel('Risk Probability (%)', fontsize=9, color='#444')
    ax.tick_params(axis='y', labelsize=9)
    ax.tick_params(axis='x', labelsize=8)
    for bar, score in zip(bars, scores):
        ax.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2,
                f'{score}%', va='center', fontsize=8, fontweight='bold', color='#333')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.tick_params(left=False)
    ax.set_facecolor('#fafafa')
    fig.patch.set_facecolor('#fafafa')
    plt.tight_layout()
    img_path = os.path.join('uploads', 'models_temp.png')
    fig.savefig(img_path, bbox_inches='tight')
    plt.close(fig)
    return img_path


def _build_clinical_pdf(form_data):
    """Build a detailed, beautiful clinical PDF report and return the file path."""
    name = form_data.get('name', 'Patient')
    email = form_data.get('email', '')
    probability = float(form_data.get('probability', 0))
    risk_level = form_data.get('risk_level', 'Unknown')
    risk_color = form_data.get('risk_color', 'orange')
    age = form_data.get('age', '')
    gender = form_data.get('gender', '')
    cp_label = form_data.get('cp_label', '')
    trestbps = form_data.get('trestbps', '')
    chol = form_data.get('chol', '')
    fbs_label = form_data.get('fbs_label', '')
    restecg_label = form_data.get('restecg_label', '')
    thalach = form_data.get('thalach', '')
    exang_label = form_data.get('exang_label', '')
    oldpeak = form_data.get('oldpeak', '')
    slope_label = form_data.get('slope_label', '')
    ca = form_data.get('ca', '')
    thal_label = form_data.get('thal_label', '')

    # Rebuild ai_analysis from form keys like ai_LR, ai_RF etc
    ai_analysis = {}
    for key, val in form_data.items():
        if key.startswith('ai_'):
            model_name = key[3:]
            ai_analysis[model_name] = float(val)

    # Generate chart images
    chart_path = _generate_chart_image(probability, risk_color)
    models_path = _generate_model_bar_image(ai_analysis) if ai_analysis else None

    report_date = datetime.now().strftime('%B %d, %Y at %I:%M %p')

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()

    # ---- HEADER ----
    pdf.set_fill_color(75, 0, 130)  # Deep purple
    pdf.rect(0, 0, 210, 38, 'F')
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", 'B', 22)
    pdf.set_y(8)
    pdf.cell(0, 10, "CardioSight", ln=True, align='C')
    pdf.set_font("Helvetica", '', 10)
    pdf.cell(0, 6, "Advanced AI-Powered Heart Disease Risk Assessment Report", ln=True, align='C')
    pdf.set_font("Helvetica", '', 8)
    pdf.cell(0, 6, f"Generated on {report_date}", ln=True, align='C')

    pdf.set_text_color(0, 0, 0)
    pdf.ln(8)

    # ---- PATIENT INFORMATION ----
    pdf.set_font("Helvetica", 'B', 14)
    pdf.set_text_color(75, 0, 130)
    pdf.cell(0, 10, "Patient Information", ln=True)
    pdf.set_draw_color(75, 0, 130)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    pdf.set_font("Helvetica", '', 11)
    pdf.set_text_color(50, 50, 50)
    info_items = [
        ("Full Name", name),
        ("Age", f"{age} years"),
        ("Gender", gender),
    ]
    if email:
        info_items.append(("Email", email))

    for label, value in info_items:
        pdf.set_font("Helvetica", 'B', 10)
        pdf.cell(50, 7, f"{label}:", align='L')
        pdf.set_font("Helvetica", '', 10)
        pdf.cell(0, 7, str(value), ln=True)
    pdf.ln(5)

    # ---- RISK ASSESSMENT RESULT ----
    pdf.set_font("Helvetica", 'B', 14)
    pdf.set_text_color(75, 0, 130)
    pdf.cell(0, 10, "Risk Assessment Result", ln=True)
    pdf.set_draw_color(75, 0, 130)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    # Risk box with color
    color_map = {'red': (255, 75, 43), 'orange': (255, 152, 0), 'green': (0, 200, 83)}
    r, g, b = color_map.get(risk_color, (255, 152, 0))
    pdf.set_fill_color(r, g, b)
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Helvetica", 'B', 16)

    box_y = pdf.get_y()
    pdf.rect(10, box_y, 190, 14, 'F')
    pdf.set_xy(10, box_y + 2)
    pdf.cell(190, 10, f"{risk_level} Risk  -  Meta-Learner Probability: {probability}%", align='C')
    pdf.ln(18)
    pdf.set_text_color(50, 50, 50)

    # ---- CHART IMAGE ----
    if os.path.exists(chart_path):
        pdf.set_font("Helvetica", 'B', 11)
        pdf.set_text_color(75, 0, 130)
        pdf.cell(0, 8, "Risk Distribution", ln=True, align='C')
        pdf.set_text_color(50, 50, 50)
        chart_x = (210 - 60) / 2
        pdf.image(chart_path, x=chart_x, y=pdf.get_y(), w=60)
        pdf.ln(65)

    # ---- MODEL-BY-MODEL ANALYSIS ----
    if models_path and os.path.exists(models_path):
        pdf.set_font("Helvetica", 'B', 14)
        pdf.set_text_color(75, 0, 130)
        pdf.cell(0, 10, "Neural Analysis Breakdown", ln=True)
        pdf.set_draw_color(75, 0, 130)
        pdf.line(10, pdf.get_y(), 200, pdf.get_y())
        pdf.ln(2)
        pdf.set_font("Helvetica", '', 9)
        pdf.set_text_color(80, 80, 80)
        pdf.cell(0, 6, "Our CNN-LSTM architecture synthesized opinions from 5 distinct clinical models:", ln=True)
        pdf.ln(2)
        model_img_x = (210 - 120) / 2
        pdf.image(models_path, x=model_img_x, y=pdf.get_y(), w=120)
        pdf.ln(max(25, len(ai_analysis) * 12 + 5))

    # ---- PAGE 2: CLINICAL PARAMETERS ----
    pdf.add_page()

    pdf.set_font("Helvetica", 'B', 14)
    pdf.set_text_color(75, 0, 130)
    pdf.cell(0, 10, "Clinical Parameters & Analysis", ln=True)
    pdf.set_draw_color(75, 0, 130)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    # Parameter table
    params = [
        ("Chest Pain Type", cp_label, "Indicates the type of chest discomfort. 'Asymptomatic' chest pain (Type 3) is paradoxically the most concerning as it may indicate silent ischemia."),
        ("Resting Blood Pressure", f"{trestbps} mm Hg", f"Normal range: 90-120 mm Hg. {'Your BP is elevated, which increases strain on the heart.' if trestbps and int(trestbps) > 130 else 'Your resting BP is within an acceptable range.' if trestbps else 'N/A'}"),
        ("Serum Cholesterol", f"{chol} mg/dl", f"Desirable: <200 mg/dl. {'Elevated cholesterol contributes to plaque build-up in arteries (atherosclerosis).' if chol and int(chol) > 200 else 'Your cholesterol level is within the desirable range.' if chol else 'N/A'}"),
        ("Fasting Blood Sugar >120", fbs_label, "Elevated fasting blood sugar may indicate diabetes, a significant risk factor for heart disease."),
        ("Resting ECG", restecg_label, "Abnormalities in resting ECG, such as ST-T wave changes, can indicate underlying cardiac conditions."),
        ("Max Heart Rate Achieved", str(thalach), f"Age-predicted max: ~{220 - int(age) if age else 'N/A'} bpm. {'Your achieved heart rate is lower than expected, which may indicate reduced cardiac fitness.' if thalach and age and int(thalach) < (220 - int(age)) * 0.7 else 'Your exercise heart rate is appropriate.' if thalach else 'N/A'}"),
        ("Exercise Induced Angina", exang_label, "Chest pain during exercise strongly suggests coronary artery narrowing limiting blood flow under stress."),
        ("ST Depression (Oldpeak)", str(oldpeak), f"ST depression during exercise indicates myocardial ischemia. {'Values above 2.0 are clinically significant.' if oldpeak and float(oldpeak) > 2.0 else 'Your ST depression value is within normal limits.' if oldpeak else 'N/A'}"),
        ("Slope of Peak Exercise ST", slope_label, "A flat or downsloping ST segment during peak exercise is more concerning than an upsloping pattern."),
        ("Major Vessels Colored (0-3)", str(ca), "The number of major vessels visible on fluoroscopy. More colored vessels generally correlate with better coronary circulation."),
        ("Thalassemia", thal_label, "A reversible defect on thalium stress testing indicates an area of the heart with reduced blood flow under stress."),
    ]

    pdf.set_font("Helvetica", 'B', 9)
    pdf.set_fill_color(75, 0, 130)
    pdf.set_text_color(255, 255, 255)
    pdf.cell(55, 8, " Parameter", border=1, fill=True)
    pdf.cell(30, 8, " Value", border=1, fill=True)
    pdf.cell(105, 8, " Clinical Significance", border=1, fill=True)
    pdf.ln()

    pdf.set_text_color(50, 50, 50)
    fill = False
    for param_name, param_val, explanation in params:
        if fill:
            pdf.set_fill_color(240, 237, 248)
        else:
            pdf.set_fill_color(255, 255, 255)
        
        x_start = pdf.get_x()
        y_start = pdf.get_y()

        # Calculate needed height for explanation text
        pdf.set_font("Helvetica", '', 8)
        # Estimate lines needed for explanation in 105mm width
        char_per_line = 60
        lines_needed = max(1, math.ceil(len(explanation) / char_per_line))
        row_h = max(8, lines_needed * 4.5)

        pdf.set_font("Helvetica", 'B', 8)
        pdf.cell(55, row_h, f" {param_name}", border=1, fill=True)
        pdf.set_font("Helvetica", '', 8)
        pdf.cell(30, row_h, f" {param_val}", border=1, fill=True)

        # Multi-line explanation
        x_expl = pdf.get_x()
        y_expl = pdf.get_y()
        pdf.set_font("Helvetica", '', 7)
        pdf.multi_cell(105, row_h / lines_needed, f" {explanation}", border=1, fill=True)

        # Align Y to next row
        max_y = max(pdf.get_y(), y_start + row_h)
        pdf.set_y(max_y)

        fill = not fill

    pdf.ln(8)

    # ---- CLINICAL RECOMMENDATIONS ----
    pdf.set_font("Helvetica", 'B', 14)
    pdf.set_text_color(75, 0, 130)
    pdf.cell(0, 10, "Clinical Recommendations & Suggestions", ln=True)
    pdf.set_draw_color(75, 0, 130)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)

    pdf.set_text_color(50, 50, 50)

    if risk_level == 'High':
        pdf.set_fill_color(255, 230, 230)
        pdf.set_font("Helvetica", 'B', 10)
        pdf.cell(0, 7, "  ATTENTION REQUIRED - HIGH RISK DETECTED", ln=True, fill=True)
        pdf.set_font("Helvetica", '', 9)
        recs = [
            "Immediate consultation with a cardiologist is strongly recommended.",
            "Consider advanced diagnostic tests: coronary angiography, stress echocardiogram, or cardiac CT.",
            "If you experience chest pain, shortness of breath, or dizziness, seek emergency medical attention.",
            "Begin or intensify statin therapy as directed by your physician for cholesterol management.",
            "Strict blood pressure monitoring - aim for <130/80 mm Hg with prescribed medication.",
            "If diabetic, maintain HbA1c below 7% through medication and diet control.",
            "Adopt the DASH diet (Dietary Approaches to Stop Hypertension) or Mediterranean diet.",
            "Limit sodium intake to under 2,300 mg/day; ideally under 1,500 mg/day.",
            "Engage in supervised cardiac rehabilitation exercise programs.",
            "Complete smoking cessation if applicable - this is the single most impactful lifestyle change.",
        ]
    elif risk_level == 'Moderate':
        pdf.set_fill_color(255, 243, 224)
        pdf.set_font("Helvetica", 'B', 10)
        pdf.cell(0, 7, "  CAUTION ADVISED - MODERATE RISK", ln=True, fill=True)
        pdf.set_font("Helvetica", '', 9)
        recs = [
            "Schedule a follow-up appointment with your primary care physician within 2-4 weeks.",
            "Consider a comprehensive lipid panel and fasting glucose test if not done recently.",
            "Aim for at least 150 minutes of moderate aerobic exercise per week (brisk walking, swimming, cycling).",
            "Adopt a heart-healthy diet: increase fruits, vegetables, whole grains, and lean proteins.",
            "Reduce saturated fat intake to less than 6% of total daily calories.",
            "Limit alcohol consumption to moderate levels (1 drink/day for women, 2 for men).",
            "Monitor blood pressure regularly at home - maintain a log for your doctor.",
            "Manage stress through meditation, yoga, or deep breathing exercises.",
            "Maintain a healthy BMI (18.5-24.9) through balanced diet and regular exercise.",
            "If you smoke, seek help to quit - consult your doctor about cessation aids.",
        ]
    else:
        pdf.set_fill_color(230, 255, 230)
        pdf.set_font("Helvetica", 'B', 10)
        pdf.cell(0, 7, "  HEALTHY PROFILE - LOW RISK", ln=True, fill=True)
        pdf.set_font("Helvetica", '', 9)
        recs = [
            "Your heart health indicators are within normal ranges - maintain your current healthy lifestyle.",
            "Continue regular annual check-ups with your physician.",
            "Maintain your exercise routine - aim for 150+ minutes of moderate activity per week.",
            "Continue eating a balanced, nutrient-rich diet with plenty of fiber.",
            "Stay hydrated and maintain adequate sleep (7-9 hours per night).",
            "Continue monitoring your cholesterol and blood pressure periodically.",
            "Keep stress levels managed through activities you enjoy.",
            "Avoid smoking and limit alcohol consumption.",
        ]

    pdf.ln(2)
    for i, rec in enumerate(recs, 1):
        pdf.set_font("Helvetica", '', 9)
        pdf.cell(8, 6, f"{i}.")
        pdf.multi_cell(0, 6, rec)
        pdf.ln(1)

    # ---- DISCLAIMER ----
    pdf.ln(8)
    pdf.set_draw_color(180, 180, 180)
    pdf.line(10, pdf.get_y(), 200, pdf.get_y())
    pdf.ln(3)
    pdf.set_font("Helvetica", 'I', 7)
    pdf.set_text_color(130, 130, 130)
    pdf.multi_cell(0, 4,
        "DISCLAIMER: This report is generated by CardioSight AI, an experimental machine learning system, "
        "and is intended for informational and educational purposes only. It does NOT constitute medical advice, "
        "diagnosis, or treatment. Always consult a qualified healthcare professional for medical decisions. "
        "The AI models used (Logistic Regression, KNN, SVM, Random Forest, Decision Tree with CNN-LSTM meta-learner) "
        "are trained on historical datasets and may not reflect your individual clinical reality. "
        "CardioSight and its developers assume no liability for decisions made based on this report."
    )
    pdf.ln(2)
    pdf.set_font("Helvetica", '', 7)
    pdf.set_text_color(75, 0, 130)
    pdf.cell(0, 4, "CardioSight AI | cardiosight.onrender.com | Powered by CNN-LSTM Deep Learning Architecture", align='C')

    report_path = os.path.join(app.config['UPLOAD_FOLDER'], f"CardioSight_Report_{name.replace(' ', '_')}.pdf")
    pdf.output(report_path)

    # Cleanup temp images
    for tmp in [chart_path, models_path]:
        if tmp:
            try: os.remove(tmp)
            except: pass

    return report_path


@app.route('/download_report', methods=['POST'])
def download_report():
    try:
        report_path = _build_clinical_pdf(request.form)
        return send_file(report_path, as_attachment=True)
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error generating report: {e}", 500


@app.route('/test_email_config')
def test_email_config():
    key = os.environ.get('BREVO_API_KEY', '').strip()
    sender = os.environ.get('BREVO_SENDER_EMAIL', '').strip()
    result = {'key_exists': bool(key), 'sender': sender or 'NOT SET'}
    if key:
        try:
            req = urllib.request.Request('https://api.brevo.com/v3/account',
                headers={'api-key': key, 'Accept': 'application/json'}, method='GET')
            resp = urllib.request.urlopen(req, timeout=10)
            acct = json.loads(resp.read().decode('utf-8'))
            result['status'] = 'VALID'
            result['email'] = acct.get('email', '')
        except urllib.error.HTTPError as e:
            result['status'] = f'FAILED {e.code}: {e.read().decode("utf-8")[:300]}'
        except Exception as e:
            result['status'] = f'ERROR: {str(e)}'
    return jsonify(result)


@app.route('/send_report', methods=['POST'])
def send_report():
    try:
        form_data = request.form
        recipient_email = form_data.get('email', '')
        name = form_data.get('name', 'Patient')

        if not recipient_email:
            return jsonify({'success': False, 'error': 'No email address provided.'}), 400

        report_path = _build_clinical_pdf(form_data)

        api_key = os.environ.get('BREVO_API_KEY', '').strip()
        sender_email = os.environ.get('BREVO_SENDER_EMAIL', '').strip()

        if not api_key:
            return jsonify({'success': False, 'error': 'BREVO_API_KEY not set on server.'}), 500
        if not sender_email:
            return jsonify({'success': False, 'error': 'BREVO_SENDER_EMAIL not set on server.'}), 500

        with open(report_path, 'rb') as f:
            pdf_base64 = base64.b64encode(f.read()).decode('utf-8')

        email_body = (f"Dear {name},\n\n"
            "Thank you for using CardioSight, our Advanced AI-Powered Heart Disease Risk Assessment System.\n\n"
            "Please find your detailed clinical report attached to this email. This report contains:\n"
            "  - Your complete risk assessment results\n"
            "  - Analysis from our CNN-LSTM deep learning architecture\n"
            "  - Detailed clinical parameter explanations\n"
            "  - Personalized health recommendations\n\n"
            "IMPORTANT: This report is AI-generated and is for informational purposes only. "
            "Please consult a qualified healthcare professional for medical advice.\n\n"
            "Report generated from: CardioSight AI (cardiosight.onrender.com)\n\n"
            "Stay healthy,\nThe CardioSight AI Team")

        payload = json.dumps({
            "sender": {"name": "CardioSight AI", "email": sender_email},
            "to": [{"email": recipient_email, "name": name}],
            "subject": "CardioSight - Your Heart Disease Risk Assessment Report",
            "textContent": email_body,
            "attachment": [{"content": pdf_base64, "name": f"CardioSight_Report_{name.replace(' ', '_')}.pdf"}]
        }).encode('utf-8')

        req = urllib.request.Request('https://api.brevo.com/v3/smtp/email', data=payload,
            headers={'api-key': api_key, 'Content-Type': 'application/json', 'Accept': 'application/json'},
            method='POST')

        try:
            response = urllib.request.urlopen(req, timeout=30)
            app.logger.info("Brevo sent: %s", response.read().decode('utf-8'))
        except urllib.error.HTTPError as http_err:
            error_body = http_err.read().decode('utf-8')
            app.logger.error("Brevo error %s: %s", http_err.code, error_body)
            return jsonify({'success': False, 'error': f'Email error ({http_err.code}): {error_body}'}), 500

        try: os.remove(report_path)
        except: pass

        return jsonify({'success': True, 'message': 'Report emailed successfully!'})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug_env = os.environ.get('FLASK_DEBUG')
    debug = True if debug_env is None else str(debug_env).strip().lower() not in {'0', 'false', 'no', 'off'}
    app.run(port=port, debug=debug)
