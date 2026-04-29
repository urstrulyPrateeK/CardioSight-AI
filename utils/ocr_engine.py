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
    PDFs: uses PyMuPDF for direct text extraction first,
          falls back to OCR on rendered page images for scanned PDFs.
    Images: uses Tesseract OCR.
    """
    ext = os.path.splitext(filepath)[1].lower()

    if ext == '.pdf':
        return extract_text_from_pdf(filepath)
    else:
        return extract_text_from_image(filepath)


def extract_text_from_pdf(pdf_path):
    """Extract text from PDF — tries digital text first, then OCR on rendered pages."""
    if not fitz:
        print("PyMuPDF not installed, cannot read PDFs")
        return ""
    try:
        doc = fitz.open(pdf_path)
        all_text = ""

        for page_num, page in enumerate(doc):
            # Try direct text extraction first (fast, works for digital PDFs)
            page_text = page.get_text()
            if page_text and page_text.strip():
                all_text += page_text + "\n"
            else:
                # Scanned PDF — render page to image and OCR it
                print(f"Page {page_num + 1}: No digital text, attempting OCR on rendered image...")
                ocr_text = _ocr_pdf_page(page)
                if ocr_text:
                    all_text += ocr_text + "\n"

        doc.close()

        # Final fallback: if still no text, try OCR on all pages regardless
        if not all_text.strip():
            print("No text found via direct extraction. Re-trying full OCR on all pages...")
            doc = fitz.open(pdf_path)
            for page in doc:
                ocr_text = _ocr_pdf_page(page)
                if ocr_text:
                    all_text += ocr_text + "\n"
            doc.close()

        print(f"PDF extracted text length: {len(all_text)}")
        if all_text.strip():
            print(f"PDF text preview: {all_text[:500]}")
        else:
            print("WARNING: No text could be extracted from PDF")

        return all_text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        import traceback
        traceback.print_exc()
        return ""


def _ocr_pdf_page(page):
    """Render a single PDF page to an image and run OCR on it."""
    if not pytesseract or not Image:
        print("pytesseract/Pillow not installed — cannot OCR scanned pages")
        return ""
    try:
        # Render page at 300 DPI for good OCR quality
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")

        import io
        img = Image.open(io.BytesIO(img_data))

        # Convert to grayscale for better OCR
        img = img.convert('L')

        text = pytesseract.image_to_string(img, config='--psm 6')
        return text
    except Exception as e:
        print(f"Error in PDF page OCR: {e}")
        return ""


def extract_text_from_image(image_path):
    """Extracts text from an image file using Tesseract OCR."""
    if not pytesseract or not Image:
        print("pytesseract/Pillow not installed")
        return ""
    try:
        img = Image.open(image_path)
        # Convert to grayscale for better OCR results
        img = img.convert('L')
        text = pytesseract.image_to_string(img, config='--psm 6')
        print(f"Image OCR text length: {len(text)}")
        if text.strip():
            print(f"Image OCR preview: {text[:500]}")
        return text
    except Exception as e:
        print(f"Error in OCR: {e}")
        import traceback
        traceback.print_exc()
        return ""


def parse_medical_report(text):
    """
    Parses extracted text to find heart disease parameters.
    Returns dict with keys matching the form fields.
    Handles diverse medical report formats with flexible regex.
    """
    data = {}
    if not text or not text.strip():
        print("WARNING: No text to parse")
        return data

    # Normalize text: fix common OCR artifacts
    text = text.replace('|', 'l').replace('—', '-').replace('–', '-')

    # Patient Name
    m = re.search(r'(?i)(?:patient\s*(?:name)?|name\s*of\s*patient)[:\s-]+([A-Za-z\s.]+?)(?:\n|$|\d)', text)
    if m:
        name = m.group(1).strip().rstrip('.')
        if len(name) > 1:
            data['name'] = name

    # Age — multiple patterns
    for pattern in [
        r'(?i)\bage[:\s/]+(\d{1,3})\s*(?:years?|yrs?|y)?',
        r'(?i)age[:\s]+(\d{1,3})',
        r'(?i)(\d{1,3})\s*(?:years?\s*old|yrs?\s*old)',
    ]:
        m = re.search(pattern, text)
        if m:
            age_val = int(m.group(1))
            if 1 <= age_val <= 120:
                data['age'] = str(age_val)
                break

    # Gender — multiple patterns
    for pattern in [
        r'(?i)(?:sex|gender)[:\s/]+\s*(male|female|m|f)\b',
        r'(?i)\b(male|female)\b',
    ]:
        m = re.search(pattern, text)
        if m:
            g = m.group(1).strip().lower()
            data['gender'] = 'Male' if g in ('male', 'm') else 'Female'
            break

    # Chest Pain Type
    m = re.search(r'(?i)chest\s*pain\s*(?:type)?[^:\n]*[:\s]+(.*?)(?:\n|$)', text)
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
        else:
            # Try numeric
            nm = re.search(r'(\d)', t)
            if nm and nm.group(1) in '0123':
                data['cp'] = nm.group(1)

    # Resting Blood Pressure — multiple patterns
    for pattern in [
        r'(?i)(?:resting\s*)?(?:blood\s*pressure|bp|b\.p\.?)[^:\d]*[:\s]+(\d{2,3})',
        r'(?i)(?:systolic|sbp)[^:\d]*[:\s]+(\d{2,3})',
        r'(?i)trestbps[:\s]+(\d{2,3})',
    ]:
        m = re.search(pattern, text)
        if m:
            bp_val = int(m.group(1))
            if 60 <= bp_val <= 250:
                data['trestbps'] = str(bp_val)
                break

    # Cholesterol — multiple patterns
    for pattern in [
        r'(?i)cholest[eo]r?[ao]l[^:\d]*[:\s]+(\d{2,3})',
        r'(?i)(?:serum\s*)?chol(?:esterol)?[^:\d]*[:\s]+(\d{2,3})',
        r'(?i)total\s*cholesterol[^:\d]*[:\s]+(\d{2,3})',
    ]:
        m = re.search(pattern, text)
        if m:
            chol_val = int(m.group(1))
            if 100 <= chol_val <= 600:
                data['chol'] = str(chol_val)
                break

    # Fasting Blood Sugar
    m = re.search(r'(?i)fasting\s*(?:blood\s*)?sugar[^:\n]*[:\s]+(.*?)(?:\n|$)', text)
    if m:
        t = m.group(1).strip().lower()
        if 'true' in t or 'yes' in t or '>120' in t.replace(' ', '') or '>  120' in t:
            data['fbs'] = '1'
        elif 'false' in t or 'no' in t or 'normal' in t:
            data['fbs'] = '0'
        else:
            # Try to parse numeric value
            nm = re.search(r'(\d+)', t)
            if nm:
                data['fbs'] = '1' if int(nm.group(1)) > 120 else '0'

    # Resting ECG
    m = re.search(r'(?i)(?:resting\s*)?(?:ecg|ekg|electrocardiogra)[^:\n]*[:\s]+(.*?)(?:\n|$)', text)
    if m:
        t = m.group(1).strip().lower()
        if 'normal' in t and 'abnormal' not in t:
            data['restecg'] = '0'
        elif 'st' in t or 'abnormal' in t or 'wave' in t:
            data['restecg'] = '1'
        elif 'hypertrophy' in t or 'lvh' in t:
            data['restecg'] = '2'

    # Max Heart Rate — multiple patterns
    for pattern in [
        r'(?i)(?:max(?:imum)?\s*)?heart\s*rate[^:\d]*[:\s]+(\d{2,3})',
        r'(?i)thalach[:\s]+(\d{2,3})',
        r'(?i)max\s*hr[:\s]+(\d{2,3})',
        r'(?i)peak\s*heart\s*rate[^:\d]*[:\s]+(\d{2,3})',
    ]:
        m = re.search(pattern, text)
        if m:
            hr_val = int(m.group(1))
            if 50 <= hr_val <= 250:
                data['thalach'] = str(hr_val)
                break

    # Exercise Induced Angina
    m = re.search(r'(?i)exercise\s*(?:induced\s*)?angina[^:\n]*[:\s]+(.*?)(?:\n|$)', text)
    if m:
        t = m.group(1).strip().lower()
        data['exang'] = '1' if ('yes' in t or 'true' in t or 'present' in t) else '0'

    # ST Depression (Oldpeak) — multiple patterns
    for pattern in [
        r'(?i)(?:st\s*depression|oldpeak|st\s*segment\s*depression)[^:\d]*[:\s]+([\d.]+)',
        r'(?i)oldpeak[:\s]+([\d.]+)',
    ]:
        m = re.search(pattern, text)
        if m:
            try:
                val = float(m.group(1))
                if 0 <= val <= 10:
                    data['oldpeak'] = m.group(1)
                    break
            except ValueError:
                pass

    # Slope
    m = re.search(r'(?i)slope[^:\n]*[:\s]+(.*?)(?:\n|$)', text)
    if m:
        t = m.group(1).strip().lower()
        if 'upslop' in t or 'up' in t:
            data['slope'] = '0'
        elif 'flat' in t:
            data['slope'] = '1'
        elif 'downslop' in t or 'down' in t:
            data['slope'] = '2'
        else:
            nm = re.search(r'(\d)', t)
            if nm and nm.group(1) in '012':
                data['slope'] = nm.group(1)

    # Number of Major Vessels
    for pattern in [
        r'(?i)(?:number\s*of\s*)?major\s*vessels[^:\d]*[:\s]+(\d)',
        r'(?i)\bca\b[:\s]+(\d)',
        r'(?i)vessels?\s*(?:colored|visible)[^:\d]*[:\s]+(\d)',
    ]:
        m = re.search(pattern, text)
        if m:
            ca_val = int(m.group(1))
            if 0 <= ca_val <= 4:
                data['ca'] = str(ca_val)
                break

    # Thalassemia
    m = re.search(r'(?i)thalass?emia[^:\n]*[:\s]+(.*?)(?:\n|$)', text)
    if m:
        t = m.group(1).strip().lower()
        if 'normal' in t:
            data['thal'] = '1'
        elif 'fixed' in t:
            data['thal'] = '2'
        elif 'revers' in t:
            data['thal'] = '3'
        else:
            nm = re.search(r'(\d)', t)
            if nm and nm.group(1) in '123':
                data['thal'] = nm.group(1)

    print(f"Parsed {len(data)} fields: {data}")
    return data
