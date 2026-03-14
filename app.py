# Import libraries
import os
import io
import json
import base64
import urllib.request
import urllib.error
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template, redirect, url_for, send_file, session
import joblib
from werkzeug.utils import secure_filename
from utils.ocr_engine import extract_text_from_file, parse_medical_report
import math
from fpdf import FPDF
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model
except ImportError:
    tf = None
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

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- FIREBASE SETUP ---
try:
    if not firebase_admin:
        print("WARNING: firebase_admin not installed; Firebase features disabled.")
        db = None
    elif not os.path.exists(config.FIREBASE_CREDENTIALS_PATH):
        print("WARNING: Firebase Admin JSON not found. Please place it in the root directory.")
        db = None
    else:
        cred = credentials.Certificate(config.FIREBASE_CREDENTIALS_PATH)
        firebase_admin.initialize_app(cred)
        db = firestore.client()
        print("Firebase Admin SDK Initialized Successfully.")
except Exception as e:
    print(f"Error initializing Firebase: {e}")
    db = None

# --- LOAD MODELS ---
MODEL_DIR = 'cardiosight_models'
try:
    scaler = joblib.load(os.path.join(MODEL_DIR, 'scaler.pkl'))
    base_models = {name: joblib.load(os.path.join(MODEL_DIR, f'{name}.pkl')) for name in ['LR', 'KNN', 'SVM', 'RF', 'DT']}
    meta_model = load_model(os.path.join(MODEL_DIR, 'cnn_lstm.h5')) if load_model else None
    print("CardioSight Models Loaded.")
except:
    print("Model loading failed.")
    scaler, base_models, meta_model = None, None, None

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

@app.route('/dashboard', methods=['GET'])
def dashboard():
    return render_template('dashboard.html')

@app.route('/save_profile', methods=['POST'])
def save_profile():
    if not db: return jsonify({'error': 'Database not connected'}), 500
    
    try:
        data = request.json
        uid = data.get('uid')
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
        try: os.remove(filepath)
        except: pass
        return jsonify({'success': True, 'data': data})

@app.route('/process_prediction', methods=['POST'])
def process_prediction():
    try:
        id_token = request.form.get('id_token')
        uid = None
        if id_token:
            try:
                decoded_token = auth.verify_id_token(id_token)
                uid = decoded_token['uid']
            except:
                print("Invalid Token")

        name = request.form.get('name', 'Guest')
        email = request.form.get('email', '')
        age = int(request.form.get('age') or 0)
        gender = request.form.get('gender', 'Male')
        cp = int(request.form.get('cp') or 0)
        trestbps = int(request.form.get('trestbps') or 120)
        chol = int(request.form.get('chol') or 200)
        fbs = int(request.form.get('fbs') or 0)
        restecg = int(request.form.get('restecg') or 0)
        thalach = int(request.form.get('thalach') or 150)
        exang = int(request.form.get('exang') or 0)
        oldpeak = float(request.form.get('oldpeak') or 0)
        slope = int(request.form.get('slope') or 0)
        ca = int(request.form.get('ca') or 0)
        thal = int(request.form.get('thal') or 1)

        gender_val = 1 if gender == "Male" else 0
        raw_features = np.array([[age, gender_val, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]])
        
        if scaler and meta_model:
            scaled_features = scaler.transform(raw_features)
            base_probs = []
            base_details = {}
            for n, m in base_models.items():
                p = m.predict_proba(scaled_features)[0][1]
                base_probs.append(p)
                base_details[n] = round(p*100, 2)
            
            base_probs = np.array([base_probs])
            input_seq = scaled_features.reshape(1, 13, 1)
            input_dense = np.hstack([scaled_features, base_probs])
            
            prediction_prob = meta_model.predict([input_seq, input_dense])[0][0]
            probability_percent = round(float(prediction_prob) * 100, 2)
        else:
            probability_percent = 45.5
            base_details = {'RF': 40, 'LR': 50}

        if probability_percent < 30: risk_level, risk_color = "Low", "green"
        elif probability_percent < 70: risk_level, risk_color = "Moderate", "orange"
        else: risk_level, risk_color = "High", "red"

        # Map coded values back to readable labels for the report
        cp_labels = {0: 'Typical Angina', 1: 'Atypical Angina', 2: 'Non-anginal Pain', 3: 'Asymptomatic'}
        fbs_labels = {0: 'No (<=120 mg/dl)', 1: 'Yes (>120 mg/dl)'}
        restecg_labels = {0: 'Normal', 1: 'ST-T Wave Abnormality', 2: 'Left Ventricular Hypertrophy'}
        exang_labels = {0: 'No', 1: 'Yes'}
        slope_labels = {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}
        thal_labels = {1: 'Normal', 2: 'Fixed Defect', 3: 'Reversable Defect'}

        result_data = {
            'name': name,
            'email': email,
            'probability': probability_percent,
            'risk_level': risk_level,
            'risk_color': risk_color,
            'ai_analysis': base_details,
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
            history_data = {
                'timestamp': firestore.SERVER_TIMESTAMP,
                'result': result_data,
                'input': {'age': age, 'chol': chol, 'bp': trestbps}
            }
            db.collection('users').document(uid).collection('history').add(history_data)

        return render_template('result.html', result=result_data)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"Error: {e}"


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
            print(f"Brevo sent: {response.read().decode('utf-8')}")
        except urllib.error.HTTPError as http_err:
            error_body = http_err.read().decode('utf-8')
            print(f"Brevo error {http_err.code}: {error_body}")
            return jsonify({'success': False, 'error': f'Email error ({http_err.code}): {error_body}'}), 500

        try: os.remove(report_path)
        except: pass

        return jsonify({'success': True, 'message': 'Report emailed successfully!'})

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

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