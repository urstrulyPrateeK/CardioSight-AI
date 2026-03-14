import re
import os

# Try importing PyMuPDF for PDF text extraction
try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# Try importing pytesseract for image OCR
try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None


def extract_text_from_file(filepath):
    """
    Extracts text from a PDF or image file.
    PDFs: uses PyMuPDF for direct text extraction (fast, reliable).
    Images: uses Tesseract OCR.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.pdf':
        return extract_text_from_pdf(filepath)
    else:
        return extract_text_from_image(filepath)


def extract_text_from_pdf(pdf_path):
    """Extract text directly from PDF using PyMuPDF."""
    if not fitz:
        print("PyMuPDF not installed, cannot read PDFs")
        return ""
    try:
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        doc.close()
        print(f"PDF extracted text length: {len(text)}")
        print(f"PDF text preview: {text[:500]}")
        return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""


def extract_text_from_image(image_path):
    """Extracts text from an image file using Tesseract OCR."""
    if not pytesseract or not Image:
        print("pytesseract/Pillow not installed")
        return ""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print(f"Error in OCR: {e}")
        return ""


def parse_medical_report(text):
    """
    Parses extracted text to find heart disease parameters.
    Returns dict with keys matching the form fields.
    """
    data = {}
    if not text or not text.strip():
        print("WARNING: No text to parse")
        return data

    # Patient Name
    m = re.search(r'(?i)patient\s*name[:\s]+([A-Za-z\s]+)', text)
    if m:
        data['name'] = m.group(1).strip()

    # Age
    m = re.search(r'(?i)\bage[:\s]+(\d{1,3})', text)
    if m:
        data['age'] = m.group(1)

    # Gender
    if re.search(r'(?i)(?:sex|gender)[:\s]+(?:male|m\b)', text):
        data['gender'] = 'Male'
    elif re.search(r'(?i)(?:sex|gender)[:\s]+(?:female|f\b)', text):
        data['gender'] = 'Female'

    # Chest Pain Type
    m = re.search(r'(?i)chest\s*pain\s*(?:type)?[^:\n]*[:\s]+(.*)', text)
    if m:
        t = m.group(1).strip().lower()
        if 'typical' in t and 'atypical' not in t:
            data['cp'] = '0'
        elif 'atypical' in t:
            data['cp'] = '1'
        elif 'non' in t:
            data['cp'] = '2'
        elif 'asymptomatic' in t:
            data['cp'] = '3'

    # Resting Blood Pressure
    m = re.search(r'(?i)(?:resting\s*)?blood\s*pressure[^:\d]*[:\s]+(\d{2,3})', text)
    if m:
        data['trestbps'] = m.group(1)

    # Cholesterol
    m = re.search(r'(?i)cholest[eo]r?[ao]l[^:\d]*[:\s]+(\d{2,3})', text)
    if m:
        data['chol'] = m.group(1)

    # Fasting Blood Sugar
    m = re.search(r'(?i)fasting\s*blood\s*sugar[^:\n]*[:\s]+(.*)', text)
    if m:
        t = m.group(1).strip().lower()
        if 'true' in t or 'yes' in t:
            data['fbs'] = '1'
        else:
            data['fbs'] = '0'

    # Resting ECG
    m = re.search(r'(?i)(?:resting\s*)?ecg[^:\n]*[:\s]+(.*)', text)
    if m:
        t = m.group(1).strip().lower()
        if 'normal' in t:
            data['restecg'] = '0'
        elif 'st' in t or 'abnormal' in t:
            data['restecg'] = '1'
        elif 'hypertrophy' in t:
            data['restecg'] = '2'

    # Max Heart Rate
    m = re.search(r'(?i)(?:max(?:imum)?\s*)?heart\s*rate[^:\d]*[:\s]+(\d{2,3})', text)
    if m:
        data['thalach'] = m.group(1)

    # Exercise Induced Angina
    m = re.search(r'(?i)exercise\s*induced\s*angina[^:\n]*[:\s]+(.*)', text)
    if m:
        t = m.group(1).strip().lower()
        data['exang'] = '1' if ('yes' in t or 'true' in t) else '0'

    # ST Depression (Oldpeak)
    m = re.search(r'(?i)(?:st\s*depression|oldpeak)[^:\d]*[:\s]+([\d.]+)', text)
    if m:
        data['oldpeak'] = m.group(1)

    # Slope
    m = re.search(r'(?i)slope[^:\n]*[:\s]+(.*)', text)
    if m:
        t = m.group(1).strip().lower()
        if 'upslop' in t:
            data['slope'] = '0'
        elif 'flat' in t:
            data['slope'] = '1'
        elif 'downslop' in t:
            data['slope'] = '2'

    # Number of Major Vessels
    m = re.search(r'(?i)(?:number\s*of\s*)?major\s*vessels[^:\d]*[:\s]+(\d)', text)
    if m:
        data['ca'] = m.group(1)

    # Thalassemia
    m = re.search(r'(?i)thalass?emia[^:\n]*[:\s]+(.*)', text)
    if m:
        t = m.group(1).strip().lower()
        if 'normal' in t:
            data['thal'] = '1'
        elif 'fixed' in t:
            data['thal'] = '2'
        elif 'revers' in t:
            data['thal'] = '3'

    print(f"Parsed data: {data}")
    return data
