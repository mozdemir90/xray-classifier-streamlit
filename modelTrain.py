import os
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib

# -----------------------
# AYARLAR
# -----------------------
DATA_DIR = "data"  # MedicalExpert klasörünün yolu
IMAGE_SIZE = (128, 128)

# -----------------------
# VERİ OKUMA
# -----------------------
X = []
y = []

# Tüm MedicalExpert klasörlerini dolaş
for expert_folder in os.listdir(DATA_DIR):
    expert_path = os.path.join(DATA_DIR, expert_folder)
    if not os.path.isdir(expert_path):
        continue

    # Alt klasörler (0Normal, 1Doubtful, ...)
    for class_folder in os.listdir(expert_path):
        class_path = os.path.join(expert_path, class_folder)
        if not os.path.isdir(class_path):
            continue

        label = class_folder  # Örnek: "0Normal"

        # Görselleri dolaş
        for filename in os.listdir(class_path):
            if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(class_path, filename)
                try:
                    img = Image.open(file_path).convert("L")  # Grayscale
                    img = img.resize(IMAGE_SIZE)
                    img_array = np.array(img) / 255.0
                    X.append(img_array.flatten())  # 2D → 1D
                    y.append(label)
                except Exception as e:
                    print(f"{file_path} yüklenemedi: {e}")


# -----------------------
# VERİ HAZIRLAMA
# -----------------------
X = np.array(X)
y = np.array(y)

le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Toplam görsel:", len(X))

X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# -----------------------
# MODEL EĞİTİMİ
# -----------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# -----------------------
# MODELİ KAYDET
# -----------------------
joblib.dump(clf, "model.pkl")
joblib.dump(le, "label_encoder.pkl")

print("✅ Model başarıyla eğitildi ve model.pkl olarak kaydedildi.")

