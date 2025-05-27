import joblib
import numpy as np
from PIL import Image
import os

# Modeli yükle
model = joblib.load("model.pkl")

# Görseli yükle
img_path = "Severe.png"  # test edeceğin X-ray görseli
image = Image.open(img_path).convert("L").resize((128, 128))
image_array = np.array(image).flatten() / 255.0

# Tahmin et
prediction = model.predict([image_array])
print(f"Tahmin edilen sınıf: {prediction[0]}")