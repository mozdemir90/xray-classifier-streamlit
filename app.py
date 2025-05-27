from flask import Flask, request, jsonify
import joblib
import numpy as np
from PIL import Image
import io

app = Flask(__name__)

# Modeli yükle
model = joblib.load("model.pkl")

# Tahmin fonksiyonu
def predict_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("L").resize((128, 128))
    image_array = np.array(image).flatten() / 255.0
    prediction = model.predict([image_array])
    return int(prediction[0])

# Ana endpoint
@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "Dosya yüklenmedi"}), 400

    file = request.files["file"]
    prediction = predict_image(file.read())
    return jsonify({"tahmin": prediction})

if __name__ == "__main__":
    app.run(debug=True)
# Flask uygulamasını çalıştır
# Uygulama çalışırken terminalde hata ayıklama bilgilerini göster       