# 🩻 X-ray Görüntü Sınıflandırıcı

Bu proje, diz röntgeni (X-ray) görüntülerini analiz ederek kireçlenme durumlarını sınıflandırmak için eğitilmiş bir makine öğrenmesi modeli ve Streamlit tabanlı bir web arayüzü sunar.

## 🚀 Demo

👉 [Canlı Uygulama (Streamlit Cloud)](https://xray-classifier-streamlit.streamlit.app)  
🔗 [Proje Reposu (GitHub)](https://github.com/mozdemir90/xray-classifier-streamlit)

## 🧠 Kullanılan Teknolojiler

- Python
- scikit-learn
- Streamlit
- Pillow
- NumPy
- Matplotlib
- Pandas

## ⚙️ Özellikler

- X-ray görüntüsünü yükleyerek sınıflandırma yapma
- Görüntü önizlemesi
- Tahmin sonucu ve modelin güven skoru (%)
- Tahmin loglama (isteğe bağlı olarak genişletilebilir)
- Basit ve kullanıcı dostu bir arayüz

## 📦 Kurulum

Aşağıdaki adımları izleyerek projeyi kendi bilgisayarınızda çalıştırabilirsiniz:

```bash
# 1. Repo'yu klonlayın
git clone https://github.com/mozdemir90/xray-classifier-streamlit.git
cd xray-classifier-streamlit

# 2. Ortam oluşturun ve gerekli paketleri kurun
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. Uygulamayı başlatın
streamlit run app.py

Notlar:
 * Model eğitim süreci bu repoya dahil değildir. Eğitim kodları başka bir klasörde tutulmuştur.
 * Bu proje yalnızca demo amaçlıdır ve tıbbi teşhis için kullanılmamalıdır.

Lisans:
* Bu proje açık kaynak olup MIT Lisansı ile lisanslanmıştır.