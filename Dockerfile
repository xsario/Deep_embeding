# Base image olarak Python 3.10 slim versiyonunu kullanıyoruz (daha hafif)
FROM python:3.10-slim

# Çalışma dizinini ayarla
WORKDIR /app

# Gerekli sistem kütüphanelerini yükle
# OpenCV normalde libgl1 ve libglib2.0'a ihtiyaç duyar (headless olsa bile bazen gerekebilir veya deepface'in diğer bağımlılıkları için)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Bağımlılık dosyasını kopyala
COPY requirements.txt .

# Bağımlılıkları yükle
# --no-cache-dir ile imaj boyutunu küçük tutuyoruz
RUN pip install --no-cache-dir -r requirements.txt

# deepface'in model ağırlıklarını indirmesi gerekebilir. 
# Normalde ilk çalıştırmada indirir ama burada da kalabilir.
# Home dizinini ayarlayalım ki modeller oraya insin.
ENV DEEPFACE_HOME=/app/.deepface

# Uygulama kodunu kopyala
COPY embeding.py .

# Çalıştırılacak komut
CMD ["python", "embeding.py"]
