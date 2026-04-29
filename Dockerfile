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

# Install TensorFlow CPU (separate layer for caching - large package)
RUN pip install --no-cache-dir tensorflow-cpu==2.15.1

# Copy the rest of the application
COPY . .

# Train models during build so they are baked into the image
RUN python train_cardiosight.py

# Expose the port Flask runs on
EXPOSE 5000

# Command to run the application with Gunicorn
CMD gunicorn --bind 0.0.0.0:${PORT:-5000} --workers 1 --timeout 300 --preload app:app
