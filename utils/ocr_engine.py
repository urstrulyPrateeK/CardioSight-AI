import os
import re

# Optional PDF text extraction engines
try:
    import pdfplumber
except ImportError:
    pdfplumber = None

try:
    import fitz  # PyMuPDF
except ImportError:
    fitz = None

# Optional OCR stack
try:
    import pytesseract
    from PIL import Image
except ImportError:
    pytesseract = None
    Image = None

if pytesseract:
    # Windows-friendly fallback when Tesseract is installed but PATH is not refreshed.
    default_cmd_candidates = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
    ]
    for cmd_path in default_cmd_candidates:
        if os.path.exists(cmd_path):
            pytesseract.pytesseract.tesseract_cmd = cmd_path
            break


FEATURE_FIELDS = [
    'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs',
    'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal'
]

FIELD_RANGES = {
    'trestbps': (80, 220),
    'chol': (100, 600),
    'thalach': (60, 220),
    'oldpeak': (0.0, 6.0),
}


def _clean_text(value):
    if not value:
        return ""
    txt = str(value).replace("\r", "\n")
    txt = re.sub(r"[ \t]+", " ", txt)
    txt = re.sub(r"\n{2,}", "\n", txt)
    return txt.strip()


def extract_text_from_file(filepath):
    """Extract text from uploaded PDF/image with multi-step fallback."""
    ext = os.path.splitext(filepath)[1].lower()
    if ext == '.pdf':
        return extract_text_from_pdf(filepath)
    return extract_text_from_image(filepath)


def extract_text_from_pdf(pdf_path):
    """
    PDF extraction order:
    1) pdfplumber text extraction
    2) PyMuPDF text extraction
    3) OCR fallback on rendered PDF pages (if pytesseract available)
    """
    text_parts = []

    if pdfplumber:
        try:
            with pdfplumber.open(pdf_path) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        text_parts.append(page_text)
        except Exception:
            pass

    if text_parts:
        return _clean_text("\n".join(text_parts))

    if fitz:
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                page_text = page.get_text() or ""
                if page_text.strip():
                    text_parts.append(page_text)
            doc.close()
        except Exception:
            pass

    if text_parts:
        return _clean_text("\n".join(text_parts))

    # Last resort: OCR each rendered PDF page.
    if fitz and pytesseract and Image:
        ocr_parts = []
        try:
            doc = fitz.open(pdf_path)
            for page in doc:
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
                mode = "RGBA" if pix.alpha else "RGB"
                img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
                ocr_text = pytesseract.image_to_string(img) or ""
                if ocr_text.strip():
                    ocr_parts.append(ocr_text)
            doc.close()
        except Exception:
            pass
        return _clean_text("\n".join(ocr_parts))

    return ""


def extract_text_from_image(image_path):
    """OCR text extraction for images using pytesseract."""
    if not (pytesseract and Image):
        return ""
    try:
        img = Image.open(image_path)
        return _clean_text(pytesseract.image_to_string(img))
    except Exception:
        return ""


def _normalize_key(raw_key):
    key = raw_key.lower().strip()
    key = re.sub(r"\([^)]*\)", " ", key)
    key = re.sub(r"[^a-z0-9]+", " ", key)
    return re.sub(r"\s+", " ", key).strip()


def _map_cp(value):
    v = value.lower()
    if re.search(r"\b0\b", v) or ("typical" in v and "atypical" not in v):
        return '0'
    if re.search(r"\b1\b", v) or "atypical" in v:
        return '1'
    if re.search(r"\b2\b", v) or "non-anginal" in v or "non anginal" in v:
        return '2'
    if re.search(r"\b3\b", v) or "asymptomatic" in v:
        return '3'
    return None


def _map_bool_to_binary(value):
    v = value.lower().strip()
    if re.search(r"\b(yes|true|positive)\b", v):
        return '1'
    if re.search(r"\b(no|false|negative)\b", v):
        return '0'
    num = re.search(r"[-+]?\d*\.?\d+", v)
    if num:
        try:
            n = float(num.group(0))
            return '1' if n > 0 else '0'
        except ValueError:
            return None
    return None


def _map_restecg(value):
    v = value.lower()
    if re.search(r"\b0\b", v) or "normal" in v:
        return '0'
    if re.search(r"\b1\b", v) or "st" in v or "abnormal" in v:
        return '1'
    if re.search(r"\b2\b", v) or "hypertrophy" in v or "lvh" in v:
        return '2'
    return None


def _map_slope(value):
    v = value.lower()
    if re.search(r"\b0\b", v) or "upslop" in v:
        return '0'
    if re.search(r"\b1\b", v) or "flat" in v:
        return '1'
    if re.search(r"\b2\b", v) or "downslop" in v:
        return '2'
    return None


def _map_thal(value):
    v = value.lower()
    if re.search(r"\b1\b", v) or "normal" in v:
        return '1'
    if re.search(r"\b2\b", v) or "fixed" in v:
        return '2'
    if re.search(r"\b3\b", v) or "reversible" in v or "reversable" in v:
        return '3'
    return None


def _extract_num(value, cast_type=float):
    match = re.search(r"[-+]?\d*\.?\d+", value or "")
    if not match:
        return None
    try:
        n = cast_type(match.group(0))
        if cast_type is int:
            return str(int(n))
        return str(float(n)).rstrip('0').rstrip('.') if '.' in str(float(n)) else str(float(n))
    except ValueError:
        return None


def _normalized_numeric_tokens(value):
    """
    Return OCR-tolerant numeric tokens.
    Handles common OCR artifacts like I30/l30/|30 -> 130.
    """
    if not value:
        return []
    tokens = re.findall(r"[Il|]?\d+(?:\.\d+)?", str(value))
    normalized = []
    for t in tokens:
        starts_with_ocr_prefix = t[0] in ("I", "l", "|")
        cleaned = t.replace("|", "1")
        if starts_with_ocr_prefix:
            tail = cleaned[1:]
            # Add both interpretations:
            # - OCR prefix is accidental marker (e.g., l220 -> 220)
            # - OCR prefix intended as 1 (e.g., I30 -> 130)
            if tail:
                normalized.append(tail)
            cleaned = "1" + tail
        normalized.append(cleaned)
    return normalized


def _pick_int_in_range(value, lower, upper):
    candidates = []
    for token in _normalized_numeric_tokens(value):
        if "." in token:
            continue
        try:
            n = int(token)
        except ValueError:
            continue
        if lower <= n <= upper:
            candidates.append(n)
    if not candidates:
        return None
    # Prefer 2-3 digit candidates (core use case for BP/chol/HR).
    preferred = [n for n in candidates if 2 <= len(str(abs(n))) <= 3]
    picked = preferred[0] if preferred else candidates[0]
    return str(picked)


def _pick_float_in_range(value, lower, upper):
    candidates = []
    for token in _normalized_numeric_tokens(value):
        try:
            n = float(token)
        except ValueError:
            continue
        if lower <= n <= upper:
            candidates.append(n)
    if not candidates:
        return None
    # Prefer decimal values for oldpeak; fallback to first range-safe value.
    decimal = [n for n in candidates if not n.is_integer()]
    picked = decimal[0] if decimal else candidates[0]
    return f"{picked:g}"


def _validate_numeric_field(field, value):
    if field not in FIELD_RANGES or value is None:
        return value
    low, high = FIELD_RANGES[field]
    try:
        n = float(value)
    except ValueError:
        return None
    if not (low <= n <= high):
        return None
    return value


def _extract_trestbps(value):
    return _validate_numeric_field('trestbps', _pick_int_in_range(value, 80, 220))


def _extract_chol(value):
    return _validate_numeric_field('chol', _pick_int_in_range(value, 100, 600))


def _extract_thalach(value):
    return _validate_numeric_field('thalach', _pick_int_in_range(value, 60, 220))


def _extract_oldpeak(value):
    return _validate_numeric_field('oldpeak', _pick_float_in_range(value, 0.0, 6.0))


def _apply_field_mapping(target, normalized_key, raw_value):
    value = (raw_value or "").strip()
    if not value:
        return

    def set_if_missing(field, mapped):
        if mapped is not None and field not in target:
            target[field] = mapped

    if any(k in normalized_key for k in ['patient name', 'name']):
        if 'name' not in target:
            target['name'] = value
        return

    if any(k in normalized_key for k in ['sex', 'gender']):
        v = value.lower()
        if re.search(r"\bfemale\b", v):
            target['gender'] = 'Female'
            target['sex'] = '0'
        elif re.search(r"\bmale\b", v):
            target['gender'] = 'Male'
            target['sex'] = '1'
        return

    if any(k in normalized_key for k in ['age']):
        set_if_missing('age', _extract_num(value, int))
        return

    if any(k in normalized_key for k in ['chest pain', 'cp']):
        set_if_missing('cp', _map_cp(value))
        return

    if any(k in normalized_key for k in ['resting blood pressure', 'blood pressure', 'trestbps']):
        set_if_missing('trestbps', _extract_trestbps(value))
        return

    if any(k in normalized_key for k in ['cholesterol', 'cholestrol', 'chol']):
        set_if_missing('chol', _extract_chol(value))
        return

    if any(k in normalized_key for k in ['fasting blood sugar', 'fbs']):
        set_if_missing('fbs', _map_bool_to_binary(value))
        return

    if any(k in normalized_key for k in ['rest ecg', 'resting ecg', 'restecg', 'ecg']):
        set_if_missing('restecg', _map_restecg(value))
        return

    if any(k in normalized_key for k in ['max heart rate', 'maximum heart rate', 'thalach', 'heart rate achieved']):
        set_if_missing('thalach', _extract_thalach(value))
        return

    if any(k in normalized_key for k in ['exercise induced angina', 'exang']):
        set_if_missing('exang', _map_bool_to_binary(value))
        return

    if any(k in normalized_key for k in ['oldpeak', 'st depression']):
        set_if_missing('oldpeak', _extract_oldpeak(value))
        return

    if any(k in normalized_key for k in ['slope']):
        set_if_missing('slope', _map_slope(value))
        return

    if any(k in normalized_key for k in ['major vessels', 'ca']):
        set_if_missing('ca', _extract_num(value, int))
        return

    if any(k in normalized_key for k in ['thalassemia', 'thal']):
        set_if_missing('thal', _map_thal(value))


def _parse_key_value_lines(text):
    parsed = {}
    for line in (text or "").splitlines():
        raw = line.strip()
        if not raw or len(raw) < 3:
            continue

        # Accept "label: value", "label = value", "label - value".
        # Keep numeric ranges intact (e.g., 80-220) by splitting on dash only if surrounded by spaces.
        parts = re.split(r"\s*[:=]\s*|\s+-\s+", raw, maxsplit=1)
        if len(parts) != 2:
            continue
        key, value = parts[0].strip(), parts[1].strip()
        if not key or not value:
            continue
        _apply_field_mapping(parsed, _normalize_key(key), value)
    return parsed


def _parse_labeled_lines(parsed, text):
    """
    Parse report-style lines where label and value are on same line without separators.
    Example: "Resting Blood Pressure 130 mm Hg"
    """
    lines = [re.sub(r"\s+", " ", ln.strip()) for ln in (text or "").splitlines() if ln.strip()]
    for line in lines:
        lower = line.lower()

        if 'cp' not in parsed:
            m = re.match(r"^chest pain type\s+(.+)$", lower, flags=re.IGNORECASE)
            if m:
                val = _map_cp(m.group(1))
                if val is not None:
                    parsed['cp'] = val

        if 'trestbps' not in parsed:
            m = re.match(r"^resting blood pressure\s+(.+)$", lower, flags=re.IGNORECASE)
            if m:
                val = _extract_trestbps(m.group(1))
                if val is not None:
                    parsed['trestbps'] = val

        if 'chol' not in parsed:
            m = re.match(r"^(serum\s+)?cholesterol\s+(.+)$", lower, flags=re.IGNORECASE)
            if m:
                val = _extract_chol(m.group(2))
                if val is not None:
                    parsed['chol'] = val

        if 'fbs' not in parsed:
            m = re.match(r"^fasting blood sugar.*?\s(yes|no|true|false)\b", lower, flags=re.IGNORECASE)
            if m:
                parsed['fbs'] = '1' if m.group(1) in ('yes', 'true') else '0'

        if 'restecg' not in parsed:
            m = re.match(r"^resting ecg\s+(.+)$", lower, flags=re.IGNORECASE)
            if m:
                val = _map_restecg(m.group(1))
                if val is not None:
                    parsed['restecg'] = val

        if 'thalach' not in parsed:
            m = re.match(r"^max heart rate achieved\s+(.+)$", lower, flags=re.IGNORECASE)
            if m:
                val = _extract_thalach(m.group(1))
                if val is not None:
                    parsed['thalach'] = val

        if 'exang' not in parsed:
            m = re.match(r"^exercise induced angina\s+(yes|no|true|false)\b", lower, flags=re.IGNORECASE)
            if m:
                parsed['exang'] = '1' if m.group(1) in ('yes', 'true') else '0'

        if 'oldpeak' not in parsed:
            m = re.match(r"^st depression \(oldpeak\)\s+(.+)$", lower, flags=re.IGNORECASE)
            if m:
                val = _extract_oldpeak(m.group(1))
                if val is not None:
                    parsed['oldpeak'] = val

        if 'slope' not in parsed:
            m = re.match(r"^slope of peak exercise st\s+(.+)$", lower, flags=re.IGNORECASE)
            if m:
                val = _map_slope(m.group(1))
                if val is not None:
                    parsed['slope'] = val

        if 'ca' not in parsed:
            m = re.match(r"^major vessels(?: colored)?(?: \(0-3\))?\s+(\d)\b", lower, flags=re.IGNORECASE)
            if m:
                parsed['ca'] = m.group(1)

        if 'thal' not in parsed:
            m = re.match(r"^thalassemia\s+(.+)$", lower, flags=re.IGNORECASE)
            if m:
                val = _map_thal(m.group(1))
                if val is not None:
                    parsed['thal'] = val


def _regex_fallback(parsed, text):
    lower = (text or "").lower()

    if 'age' not in parsed:
        m = re.search(r"\bage\s*[:=\-]?\s*(\d{1,3})\b", lower)
        if m:
            parsed['age'] = m.group(1)

    if 'gender' not in parsed:
        if re.search(r"\b(sex|gender)\b[^a-z]{0,10}\bmale\b", lower):
            parsed['gender'] = 'Male'
            parsed['sex'] = '1'
        elif re.search(r"\b(sex|gender)\b[^a-z]{0,10}\bfemale\b", lower):
            parsed['gender'] = 'Female'
            parsed['sex'] = '0'

    patterns = [
        ('cp', r"(chest\s*pain(?:\s*type)?|cp)[^\n:=-]{0,20}[:=\-]?\s*([^\n]+)", _map_cp),
        ('trestbps', r"(resting\s*blood\s*pressure|blood\s*pressure|trestbps)[^\n:=]{0,30}(?::|=)?\s*([^\n]+)", _extract_trestbps),
        ('chol', r"(cholesterol|cholestrol|chol)[^\n:=]{0,30}(?::|=)?\s*([^\n]+)", _extract_chol),
        ('fbs', r"(fasting\s*blood\s*sugar|fbs)[^\n:=-]{0,20}[:=\-]?\s*([^\n]+)", _map_bool_to_binary),
        ('restecg', r"(resting?\s*ecg|restecg|ecg)[^\n:=-]{0,20}[:=\-]?\s*([^\n]+)", _map_restecg),
        ('thalach', r"(max(?:imum)?\s*heart\s*rate|heart\s*rate\s*achieved|thalach)[^\n:=]{0,30}(?::|=)?\s*([^\n]+)", _extract_thalach),
        ('exang', r"(exercise\s*induced\s*angina|exang)[^\n:=-]{0,20}[:=\-]?\s*([^\n]+)", _map_bool_to_binary),
        ('oldpeak', r"(oldpeak|st\s*depression)[^\n:=]{0,30}(?::|=)?\s*([^\n]+)", _extract_oldpeak),
        ('slope', r"(slope)[^\n:=-]{0,20}[:=\-]?\s*([^\n]+)", _map_slope),
        ('ca', r"(major\s*vessels|ca)[^\n:=-]{0,20}[:=\-]?\s*(\d)", lambda x: _extract_num(x, int)),
        ('thal', r"(thalassemia|thal)[^\n:=-]{0,20}[:=\-]?\s*([^\n]+)", _map_thal),
    ]

    for field, pattern, mapper in patterns:
        if field in parsed:
            continue
        m = re.search(pattern, lower, flags=re.IGNORECASE)
        if not m:
            continue
        val = mapper(m.group(2))
        if val is not None:
            parsed[field] = val


def validate_extracted_data(data):
    """Return missing model fields for caller-side UX messaging."""
    missing = [field for field in FEATURE_FIELDS if field not in data]
    return {
        'missing_fields': missing,
        'found_fields': [field for field in FEATURE_FIELDS if field in data],
        'is_minimally_valid': len(data) >= 4
    }


def parse_medical_report(text):
    """
    Parse extracted report text into form-compatible keys used by prediction flow.
    """
    cleaned = _clean_text(text)
    if not cleaned:
        return {}

    parsed = {}
    _parse_labeled_lines(parsed, cleaned)
    parsed.update({k: v for k, v in _parse_key_value_lines(cleaned).items() if k not in parsed})
    _regex_fallback(parsed, cleaned)

    # Normalize edge cases to expected string payloads for HTML form fields.
    if 'gender' in parsed:
        parsed['gender'] = 'Male' if parsed['gender'].lower().startswith('m') else 'Female'
        parsed['sex'] = '1' if parsed['gender'] == 'Male' else '0'

    # Final numeric validation and cleanup.
    for field in ('trestbps', 'chol', 'thalach', 'oldpeak'):
        if field in parsed:
            parsed[field] = _validate_numeric_field(field, parsed[field])
            if parsed[field] is None:
                parsed.pop(field, None)

    return parsed
