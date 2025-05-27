import streamlit as st
from PIL import Image
import numpy as np
import joblib
import io

# Model ve boyut
MODEL_PATH = "model.pkl"
IMAGE_SIZE = (128, 128)

# Modeli yükle
model = joblib.load(MODEL_PATH)

# Başlık
#st.title("Diz Röntgeni Sınıflandırma (Kellgren-Lawrence Skoru)")
st.markdown("<h1 style='color: blue;'>Diz Röntgeni Sınıflandırma (Kellgren-Lawrence Skoru)</h1>", unsafe_allow_html=True)
st.write("Görsel yükleyin ve modelin sınıflandırmasını görün.")

# Görsel yükleme
uploaded_file = st.file_uploader("Bir diz röntgeni görseli yükleyin (PNG veya JPG)", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("L")  # Gri tonlama
    image = image.resize(IMAGE_SIZE)
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

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
    if prediction == 0:
        st.write("Sınıf 0: Normal diz")
    elif prediction == 1:
        st.write("Sınıf 1: Hafif osteoartrit")
    elif prediction == 2:
        st.write("Sınıf 2: Orta osteoartrit")
    elif prediction == 3:
        st.write("Sınıf 3: İleri osteoartrit")
    elif prediction == 4:
        st.write("Sınıf 4: Çok ileri osteoartrit")
else:
    st.warning("Lütfen bir görsel yükleyin.")
# Uygulama sonu
st.write("Uygulama, diz röntgeni görsellerini Kellgren-Lawrence skoruna göre sınıflandırmak için eğitilmiş bir makine öğrenimi modelini kullanır.")
# Uygulama hakkında bilgi
st.sidebar.header("Hakkında")
st.sidebar.write("Bu uygulama, diz röntgeni görsellerini Kellgren-Lawrence skoruna göre sınıflandırmak için eğitilmiş bir makine öğrenimi modelini kullanır.")
st.sidebar.write("Model, diz osteoartritinin derecelerini belirlemek için kullanılabilir.")
# Uygulama hakkında bilgi
st.sidebar.header("Kullanım Talimatları")
st.sidebar.write("1. Bir diz röntgeni görseli yükleyin (PNG veya JPG formatında).")
st.sidebar.write("2. Model, yüklenen görseli analiz edecek ve tahmin edecektir.")
st.sidebar.write("3. Tahmin edilen sınıf ve olasılıkları göreceksiniz.")
# Uygulama hakkında bilgi
st.sidebar.header("Model Bilgisi")
st.sidebar.write("Model, diz osteoartritinin Kellgren-Lawrence skoruna göre sınıflandırılması için eğitilmiştir.")
st.sidebar.write("Model, çeşitli diz röntgeni görselleri üzerinde eğitim almıştır ve görsel özelliklerini kullanarak tahmin yapar.")
# Uygulama hakkında bilgi
st.sidebar.header("Geliştirici Bilgisi")
st.sidebar.write("Bu uygulama, diz osteoartritinin Kellgren-Lawrence skoruna göre sınıflandırılması için geliştirilmiştir.")
st.sidebar.write("Geliştirici: [Adınız]")
st.sidebar.write("İletişim: [E-posta adresiniz]")
# Uygulama hakkında bilgi
st.sidebar.header("Kaynaklar")
st.sidebar.write("1. [Kellgren-Lawrence Skoru](https://en.wikipedia.org/wiki/Kellgren%E2%80%93Lawrence_scale)")     
st.sidebar.write("2. [Diz Osteoartriti](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7081230/)")
st.sidebar.write("3. [Makine Öğrenimi](https://scikit-learn.org/stable/)")
# Uygulama hakkında bilgi
st.sidebar.header("Geri Bildirim")
st.sidebar.write("Uygulama hakkında geri bildirimde bulunmak isterseniz, lütfen iletişim bilgilerinizi bırakın.")
# Uygulama hakkında bilgi
st.sidebar.write("Uygulama, Streamlit kullanılarak geliştirilmiştir.")
# Uygulama hakkında bilgi
st.sidebar.write("Streamlit, hızlı ve etkileşimli web uygulamaları oluşturmak için kullanılan bir Python kütüphanesidir.")
# Uygulama hakkında bilgi
st.sidebar.write("Uygulama, makine öğrenimi modellerini web üzerinde kolayca kullanmak için tasarlanmıştır.")