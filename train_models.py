import pandas as pd
import numpy as np
import os
import joblib

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import make_pipeline

# 3 farklı model oluşturup karşılaştırmak için fonksiyon
def train_and_select_model(csv_path, model_name, save_folder="modeller"):
    df = pd.read_csv(csv_path)

    if "siralama" not in df.columns or df.shape[0] < 5:
        print(f"{model_name} - Veri seti eksik veya yetersiz.")
        return

    # log dönüşümü (feature engineering)
    df['log_siralama'] = np.log(df['siralama'])
    X = df[['log_siralama']]
    y = df.iloc[:, 1]  # ikinci sütun genelde net değeridir (tyt_net, ayt_net, vs.)

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X, y)
    pred_lr = lr.predict(X)
    mse_lr = mean_squared_error(y, pred_lr)
    r2_lr = r2_score(y, pred_lr)
    print(f"{model_name} doğruluk oranı: %{r2_lr*100:.2f}")

    # Polynomial Regression (degree=2)
    poly = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    poly.fit(X, y)
    pred_poly = poly.predict(X)
    mse_poly = mean_squared_error(y, pred_poly)
    r2_poly = r2_score(y, pred_poly)
    print(f"{model_name} doğruluk oranı: %{r2_poly*100:.2f}")

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)
    pred_rf = rf.predict(X)
    mse_rf = mean_squared_error(y, pred_rf)
    r2_rf = r2_score(y, pred_rf)
    print(f"{model_name} doğruluk oranı: %{r2_rf*100:.2f}")

    # Karşılaştırma ve en iyi modeli seçme
    results = {
        "Linear Regression": (lr, mse_lr, r2_lr),
        "Polynomial Regression": (poly, mse_poly, r2_poly),
        "Random Forest": (rf, mse_rf, r2_rf)
    }

    best_model_name = min(results, key=lambda name: results[name][1])
    best_model, best_mse, best_r2 = results[best_model_name]

    print(f"\n {model_name} için en iyi model: {best_model_name}")
    print(f"   - MSE: {best_mse:.2f} | R²: {best_r2:.4f}")

    os.makedirs(save_folder, exist_ok=True)
    joblib.dump(best_model, f"{save_folder}/{model_name}.pkl")
    


# Tüm modelleri sırayla eğitiyoruz
models_to_train = [
    ("veriler/tyt.csv", "tyt"),
    ("veriler/ayt_sayisal.csv", "ayt_sayisal"),
    ("veriler/ayt_sozel.csv", "ayt_sozel"),
    ("veriler/ayt_ea_processed.csv", "ayt_ea"),
]

for path, name in models_to_train:
    train_and_select_model(path, name)
