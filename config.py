import os
import json
import tempfile

# --- FIREBASE SERVER CONFIGURATION ---
# Supports file path OR JSON string via environment variable
FIREBASE_CREDENTIALS_PATH = os.environ.get("FIREBASE_CREDENTIALS_PATH", "")

# If FIREBASE_CREDENTIALS_JSON env var is set, write to temp file for SDK
_firebase_json = os.environ.get("FIREBASE_CREDENTIALS_JSON", "")
if _firebase_json and not FIREBASE_CREDENTIALS_PATH:
    try:
        cred_dict = json.loads(_firebase_json)
        _temp = tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False, dir=tempfile.gettempdir())
        json.dump(cred_dict, _temp)
        _temp.close()
        FIREBASE_CREDENTIALS_PATH = _temp.name
        print(f"Firebase credentials written to temp file: {FIREBASE_CREDENTIALS_PATH}")
    except Exception as e:
        print(f"Error parsing FIREBASE_CREDENTIALS_JSON: {e}")

# Fallback to local file for development
if not FIREBASE_CREDENTIALS_PATH:
    FIREBASE_CREDENTIALS_PATH = "cardiosight-d4414-firebase-adminsdk-fbsvc-75676d467e.json"

# --- FIREBASE CLIENT CONFIGURATION ---
FIREBASE_CLIENT_CONFIG = {
    "apiKey": os.environ.get("FIREBASE_API_KEY", ""),
    "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN", ""),
    "projectId": os.environ.get("FIREBASE_PROJECT_ID", ""),
    "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET", ""),
    "messagingSenderId": os.environ.get("FIREBASE_MESSAGING_SENDER_ID", ""),
    "appId": os.environ.get("FIREBASE_APP_ID", "")
}