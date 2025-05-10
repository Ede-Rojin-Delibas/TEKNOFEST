import pandas as pd
import numpy as np
import os
import pickle
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV   

# Model ve veri seti tanımları
veri_kaynaklari = {
    "tyt": {
        "dosya": "veri_setleri/tyt.csv",
        "model": make_pipeline(
            StandardScaler(),
            RandomForestRegressor(n_estimators=100,
                                  max_depth=5, #derinliği sınırla
                                  min_samples_leaf=3, #overfitting'i azalt.
                                  random_state=42)
        )
    },
    "ayt_sayisal": {
        "dosya": "veri_setleri/ayt_sayisal.csv",
        "model": make_pipeline(
            StandardScaler(),
            RandomForestRegressor(n_estimators=200,
                                  max_depth=5, #derinliği sınırla
                                  min_samples_leaf=3, #overfitting'i azalt.
                                  random_state=42)
        )
    },
    "ayt_sozel": {
        "dosya": "veri_setleri/ayt_sozel.csv",
        "model": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2),
            RandomForestRegressor(  # PolynomialFeatures yerine
            n_estimators=50,    # Daha az ağaç
            max_depth=3,        # Daha sığ ağaçlar
            min_samples_leaf=2, # Daha az örnek
            random_state=42
            )
        ),
        "optimize":True #bu model için optimizasyon yapılacak
    },
    "ayt_ea": {
        "dosya": "veri_setleri/ayt_ea.csv",
        "model": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2),
            LinearRegression()
        )
    }
}

def optimize_sozel_model(X, y):
    """AYT Sözel için en iyi parametreleri bulan fonksiyon"""
    
    # Temel model pipeline
    pipeline = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(),
        RandomForestRegressor()
    )
    
    # Aranacak parametreler
    param_grid = {
        'randomforestregressor__n_estimators': [50, 100],
        'randomforestregressor__max_depth': [2, 3, 4],
        'randomforestregressor__min_samples_leaf': [1, 2, 3]
    }
    
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=3,
        scoring='neg_mean_absolute_error',  # MAE kullan
        verbose=1,
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    return grid_search.best_estimator_

def model_egit_ve_degerlendir(X, y, model, sinav_turu):
    # Veri seti çok küçükse cross-validation yerine tek split
    if len(X) < 10:
        test_size = 0.3
    else:
        test_size = 0.2
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Model eğitimi
    model.fit(X_train, y_train)
    
    # Test seti değerlendirmesi
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\nTest seti değerlendirmesi:")
    print(f"R² skoru: {r2:.4f}")
    print(f"Ortalama Mutlak Hata (MAE): {mae:.4f}")
    print(f"Kök Ortalama Kare Hata (RMSE): {rmse:.4f}")
    
    return model

# Ana döngü
if not os.path.exists("modeller"):
    os.makedirs("modeller")

for ad, bilgiler in veri_kaynaklari.items():
    try:
        print(f"\n{ad.upper()} modeli eğitiliyor...")
        
        # Veri yükleme
        veri = pd.read_csv(bilgiler["dosya"])
        print(f"\n{ad} Veri Seti Analizi:")
        print(f"Örnek sayısı: {len(veri)}")
        print(f"Toplam net aralığı: {veri['toplam_net'].min()} - {veri['toplam_net'].max()}")
        print(f"Sıralama aralığı: {veri['siralama'].min()} - {veri['siralama'].max()}")
        
        if "toplam_net" not in veri.columns:
            raise ValueError(f"{ad} veri setinde 'toplam_net' sütunu bulunamadı")
        
        # Özellik ve hedef hazırlama
        X = veri[["toplam_net"]].values.reshape(-1, 1)  # 2D array gerekli
        y = veri["siralama"]
        
        # Model eğitimi ve değerlendirme
        if ad == "ayt_sozel" and bilgiler.get("optimize", False):
            print("\nAYT Sözel için parametre optimizasyonu yapılıyor...")
            model = optimize_sozel_model(X, y)
        else:
            model = bilgiler["model"]
        
        # Model eğitimi ve değerlendirme
        model = model_egit_ve_degerlendir(X, y, model, ad)
        
        # Model kaydetme
        model_dosya_yolu = f"modeller/{ad}_model.pkl"
        with open(model_dosya_yolu, "wb") as f:
            pickle.dump(model, f)
        print(f"\nModel kaydedildi: {model_dosya_yolu}")
        
    except Exception as e:
        print(f"\nHATA - {ad}: {str(e)}")
        continue

print("\nTüm modeller eğitildi ve kaydedildi.")
