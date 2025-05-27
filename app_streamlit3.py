import streamlit as st
import numpy as np
import pickle
from PIL import Image
import io
import joblib

# Başlık ve açıklama
st.set_page_config(page_title="Diz Röntgeni Tahmin Aracı", layout="centered")
st.title("🦵 Diz Röntgeni Derecelendirme")
st.markdown("Bu araç, yüklediğiniz diz röntgeni görüntüsünün **Osteoartrit şiddet derecesini** tahmin eder.")

# Sınıf isimleri
class_names = {
    0: "0 - Normal",
    1: "1 - Şüpheli (Doubtful)",
    2: "2 - Hafif (Mild)",
    3: "3 - Orta (Moderate)",
    4: "4 - Şiddetli (Severe)"
}

# Modeli yükle
 
@st.cache_resource
def load_model():
    return joblib.load("model.pkl")


model = load_model()

# Görsel yükleme
uploaded_file = st.file_uploader("📷 Röntgen görüntüsünü yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Gri tonlama
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    # Görseli modele uygun hale getir
    image_resized = image.resize((128, 128))
    image_array = np.array(image_resized).flatten().reshape(1, -1)

    # Tahmin
    prediction = model.predict(image_array)[0]
    probability = model.predict_proba(image_array)[0][prediction] * 100

    st.markdown("---")
    st.subheader("🔍 Tahmin Sonucu")
    st.success(f"**{class_names[prediction]}**")
    st.info(f"🎯 Güven skoru: **{probability:.2f}%**")
