# train.py

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import joblib

# 1. Veri setini yükle (örnek olarak iris)
iris = load_iris()
X = pd.DataFrame(iris.data, columns=iris.feature_names)
y = iris.target

# 2. Modeli oluştur ve eğit
model = RandomForestClassifier()
model.fit(X, y)

# 3. Eğitilen modeli dosyaya kaydet
joblib.dump(model, "model.pkl")

print("Model başarıyla eğitildi ve 'model.pkl' olarak kaydedildi.")