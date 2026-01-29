import os
import cv2
import csv
import numpy as np
from deepface import DeepFace



# PHP tarafındaki müşteri görsellerinin kök dizini
# PHP tarafındaki müşteri görsellerinin kök dizini
# Environment variable 'TEST_ROOT' varsa onu kullan, yoksa default değeri kullan
TEST_ROOT = os.getenv("TEST_ROOT", "static/media/customer")

# Çıktı klasörü de parametrik olsun
OUTPUT_DIR = os.getenv("OUTPUT_DIR", "output")
OUTPUT_CSV = "facenet512_embeddings.csv"

MODEL_NAME = "Facenet512"
IMAGE_SIZE = 225
EMBEDDING_DIM = 512

os.makedirs(OUTPUT_DIR, exist_ok=True)
output_path = os.path.join(OUTPUT_DIR, OUTPUT_CSV)


def get_embedding(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE))

    reps = DeepFace.represent(
        img_path=img_rgb,
        model_name=MODEL_NAME,
        detector_backend="skip",
        enforce_detection=False,
        normalization="Facenet2018"
    )

    return np.array(reps[0]["embedding"], dtype=float)



rows = []
sample_counter = 1

if not os.path.exists(TEST_ROOT):
    print(f"Hata: {TEST_ROOT} dizini bulunamadı!")
    exit()

# customer_id klasörleri
customer_dirs = sorted([
    d for d in os.listdir(TEST_ROOT)
    if os.path.isdir(os.path.join(TEST_ROOT, d))
])

for customer_id in customer_dirs:
    person_dir = os.path.join(TEST_ROOT, customer_id)

    images = sorted([
        f for f in os.listdir(person_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])

    for img_name in images:
        img_path = os.path.join(person_dir, img_name)


        image_path = f"static/media/customer/{customer_id}/{img_name}"

        img = cv2.imread(img_path)
        if img is None:
            print(f"Okunamayan dosya: {img_path}")
            continue

        try:
            embedding = get_embedding(img)

            row = {
                "sample_id": f"s{sample_counter}",
                "person_id": customer_id,
                "image_path": image_path
            }

            for i in range(EMBEDDING_DIM):
                row[f"e{i}"] = embedding[i]

            rows.append(row)
            sample_counter += 1

            print(f"İşlendi: {image_path}")

        except Exception as e:
            print(f"Hata: {img_path} | {e}")


fieldnames = ["sample_id", "person_id", "image_path"] + [
    f"e{i}" for i in range(EMBEDDING_DIM)
]

with open(output_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Toplam sample sayısı : {len(rows)}")
print(f"Toplam kişi sayısı   : {len(customer_dirs)}")
print(f"CSV çıktısı          : {output_path}")
