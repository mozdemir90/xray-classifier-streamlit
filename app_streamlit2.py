import streamlit as st
from PIL import Image
import numpy as np
import joblib
import logging

# Loglama ayarları
logging.basicConfig(level=logging.INFO)

# Model ve boyut
MODEL_PATH = "model.pkl"
IMAGE_SIZE = (128, 128)

# Modeli yükle
model = joblib.load(MODEL_PATH)

# Başlık ve stil
st.markdown("<h1 style='color: blue;'>Diz Röntgeni Sınıflandırma (Kellgren-Lawrence Skoru)</h1>", unsafe_allow_html=True)
st.write("Görsel yükleyin ve modelin sınıflandırmasını görün.")

# Görsel yükleme
uploaded_file = st.file_uploader("Bir diz röntgeni görseli yükleyin (PNG veya JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")
    image = image.resize(IMAGE_SIZE)
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    # Görüntü işlemede seçenekler
    if st.checkbox("Histogram Eşitleme"):
        # Histogram eşitleme işlemleri
        pass

    if st.checkbox("Gürültü Azaltma"):
        # Gürültü azaltma işlemleri
        pass

    # Görseli numpy array'e çevir
    image_array = np.array(image).flatten().reshape(1, -1)

    # Tahmin yap
    prediction = model.predict(image_array)[0]
    st.success(f"Tahmin edilen sınıf: {prediction}")

    # Tahmin olasılıklarını göster
    probabilities = model.predict_proba(image_array)[0]
    st.write("Tahmin olasılıkları:")
    for i, prob in enumerate(probabilities):
        st.write(f"Sınıf {i}: {prob:.2f}")

    # Tahmin edilen sınıfı açıklama ekle
    sınıf_aciklamalari = {
        0: "Normal diz",
        1: "Hafif osteoartrit",
        2: "Orta osteoartrit",
        3: "İleri osteoartrit",
        4: "Çok ileri osteoartrit"
    }
    st.write(f"Sınıf {prediction}: {sınıf_aciklamalari[prediction]}")
else:
    st.warning("Lütfen bir görsel yükleyin.")

# Modeli yeniden eğitme
if st.button("Modeli Yeniden Eğit"):
    st.write("Model yeniden eğitiliyor...")
    # Model eğitimi kodu buraya eklenecek

# Sidebar bilgileri
st.sidebar.header("Hakkında")
st.sidebar.write("Bu uygulama, diz röntgeni görsellerini Kellgren-Lawrence skoruna göre sınıflandırmak için geliştirilmiştir.")
st.sidebar.write("Model, diz osteoartritinin derecelerini belirlemek için kullanılabilir.")
st.sidebar.header("Kullanım Talimatları")
st.sidebar.write("1. Bir diz röntgeni görseli yükleyin (PNG veya JPG formatında).")
st.sidebar.write("2. Model, yüklenen görseli analiz edecek ve tahmin edecektir.")
st.sidebar.write("3. Tahmin edilen sınıf ve olasılıkları göreceksiniz.")
st.sidebar.header("Geri Bildirim")
st.sidebar.write("Geri bildirimleriniz için teşekkür ederiz!")

logging.info("Uygulama başlatıldı.")