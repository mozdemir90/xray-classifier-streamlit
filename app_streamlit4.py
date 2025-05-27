import streamlit as st
import numpy as np
from PIL import Image
import joblib
import os
import pandas as pd
import datetime
import matplotlib.pyplot as plt

st.set_page_config(
    page_title="X-ray Sınıflandırıcı",
    page_icon="static/icon.jpeg",  # örnek: PNG dosyası
    layout="centered"
)
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    encoder = joblib.load("label_encoder.pkl")
    return model, encoder

def log_prediction(filename, prediction, confidence):
    log_file = "logs.csv"
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    row = {
        "timestamp": now,
        "filename": filename,
        "prediction": prediction,
        "confidence": confidence
    }
    df = pd.DataFrame([row])
    if os.path.exists(log_file):
        df.to_csv(log_file, mode='a', header=False, index=False)
    else:
        df.to_csv(log_file, index=False)

st.sidebar.title("Bilgilendirme")
st.sidebar.info("""
Bu uygulama diz röntgeni görsellerinden hastalık seviyesini sınıflandırmak için eğitildi.  
Lütfen net ve doğrudan X-ray görüntüsü yükleyin.
""")

st.title("X-ray Sınıflandırıcı")

uploaded_file = st.file_uploader("Bir diz X-ray görseli yükleyin", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    resized = image.resize((128, 128))
    array = np.array(resized) / 255.0
    flat = array.flatten().reshape(1, -1)

    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    model, encoder = load_model()
    prediction = model.predict(flat)[0]
    probs = model.predict_proba(flat)[0]

    predicted_label = encoder.inverse_transform([prediction])[0]
    confidence = round(np.max(probs) * 100, 2)

    st.subheader("🔍 Tahmin Sonucu:")
    st.success(f"{predicted_label} ({confidence}%)")

    st.subheader("📊 Güven Dağılımı")
    fig, ax = plt.subplots()
    ax.bar(encoder.classes_, probs)
    ax.set_ylabel("Güven (%)")
    ax.set_ylim(0, 1)
    st.pyplot(fig)

    log_prediction(uploaded_file.name, predicted_label, confidence)
    st.success("Tahmin kaydedildi.")
    st.markdown("---")
    