# Base image
FROM python:3.9-slim

# Çalışma dizini
WORKDIR /app

# Gereksinimleri kopyala ve yükle
COPY ../../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# src klasörünü kopyala
COPY ../../src ./src

# Inference scriptini çalıştır
CMD ["python", "src/inference/run_inference.py"]