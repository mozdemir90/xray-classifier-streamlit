import streamlit as st
import numpy as np
from PIL import Image
import joblib
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt

# Sayfa ayarları
st.set_page_config(
    page_title="X-ray Sınıflandırıcı",
    page_icon="static/icon.jpeg",  # Örnek ikon dosyası
    layout="centered"
)

@st.cache_resource
def load_model():
    """Modeli ve etiket kodlayıcısını yükle."""
    model = joblib.load("model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

def log_prediction(filename, prediction, confidence):
    """Tahmin sonuçlarını kaydet."""
    log_file = "logs.csv"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {"timestamp": now, "filename": filename, "prediction": prediction, "confidence": confidence}
    df = pd.DataFrame([row])

    if os.path.exists(log_file):
        df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df.to_csv(log_file, index=False)

def preprocess_image(uploaded_file):
    """Görüntüyü işleyin ve modele uygun formata getirin."""
    image = Image.open(uploaded_file).convert("L")
    resized = image.resize((128, 128))
    array = np.array(resized) / 255.0
    flat = array.flatten().reshape(1, -1)
    return image, flat

def display_prediction(probs, encoder, predicted_label, confidence):
    """Tahmin sonuçlarını ve güven dağılımını göster."""
    st.subheader("🔍 Tahmin Sonucu:")
    st.success(f"{predicted_label} ({confidence}%)")

    st.subheader("📊 Güven Dağılımı")
    fig, ax = plt.subplots()
    ax.bar(encoder.classes_, probs)
    ax.set_ylabel("Güven (%)")
    ax.set_ylim(0, 1)
    ax.set_xticklabels(encoder.classes_, rotation=45)
    st.pyplot(fig)

# Sidebar bilgilendirme
st.sidebar.title("Bilgilendirme")
st.sidebar.info(
    "Bu uygulama diz röntgeni görsellerinden hastalık seviyesini sınıflandırmak için eğitildi. "
    "Lütfen net ve doğrudan X-ray görüntüsü yükleyin."
)

st.title("X-ray Sınıflandırıcı")

# Görsel yükleme
uploaded_file = st.file_uploader("Bir diz X-ray görseli yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image, flat = preprocess_image(uploaded_file)
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    model, encoder = load_model()
    prediction = model.predict(flat)[0]
    probs = model.predict_proba(flat)[0]

    predicted_label = encoder.inverse_transform([prediction])[0]
    confidence = round(np.max(probs) * 100, 2)

    display_prediction(probs, encoder, predicted_label, confidence)

    log_prediction(uploaded_file.name, predicted_label, confidence)
    st.success("Tahmin kaydedildi.")
    st.markdown("---")  