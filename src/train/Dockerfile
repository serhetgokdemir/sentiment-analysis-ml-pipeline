# Base image
FROM python:3.9-slim

# Work directory
WORKDIR /app

# Gereksinimleri kopyala ve yükle
COPY ../../requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the src folder
COPY ../../src ./src

# Run the train script
CMD ["python", "src/train/train.py"]