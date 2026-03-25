# CardioSight AI

Heart disease risk prediction web app built with Flask, Firebase auth/profile storage, OCR-based form assist, and an ensemble inference pipeline:

- Base models: `LR`, `KNN`, `SVM`, `RF`, `DT`
- Meta model: `CNN-LSTM` (`cardiosight_models/cnn_lstm.h5`)
- Final interpretation: `Low / Moderate / High` risk bands

## Local setup

1. Create and activate a Python virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Copy `.env.example` to `.env` and fill required values:
   - Firebase client config (`FIREBASE_*`)
   - Firebase Admin credentials path (`FIREBASE_CREDENTIALS_PATH`)
   - Optional Brevo email settings (`BREVO_*`)
4. Start backend:

```bash
python app.py
```

App default URL: `http://127.0.0.1:5000`

## Optional frontend command compatibility

This project uses Flask-rendered templates (no separate frontend build). For compatibility:

```bash
npm install
npm run dev
```

## Project structure

- `app.py` - Flask routes, inference pipeline, PDF/email report generation
- `templates/` - UI pages (`login`, `register`, `predict`, `result`, `dashboard`)
- `static/` - JS and CSS assets
- `cardiosight_models/` - trained model artifacts (`*.pkl`, `cnn_lstm.h5`, `scaler.pkl`)
- `dataset/heart_data_set.csv` - source dataset
- `utils/ocr_engine.py` - OCR + report parsing utilities

## Notes

- Inference uses fixed feature order:
  `age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal`
- No retraining is required to run inference.
- If Firebase credentials are not set, prediction still works but profile/history features are disabled.
