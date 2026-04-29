# ============================================
# CardioSight - Production Dockerfile
# ============================================
FROM python:3.11-slim

# Install system dependencies (Tesseract OCR, image libs)
RUN apt-get update && apt-get install -y --no-install-recommends \
    tesseract-ocr \
    libtesseract-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application (includes pre-trained models)
COPY . .

# Ensure uploads directory exists
RUN mkdir -p uploads

# Expose the port Flask runs on
EXPOSE 5000

# Command to run the application with Gunicorn
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --timeout 300 --preload app:app
