import pandas as pd
import numpy as np
import os
import pickle
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import GridSearchCV   
import sys
import os

# Proje kök dizinini Python path'ine ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from config import Config
import prepared_data

# Model ve veri seti tanımları
VERI_KAYNAKLARI = {
    "tyt": {
        "dosya": "data/veri_setleri/tyt_veri_seti.csv",
        "model": make_pipeline(
            StandardScaler(),
            RandomForestRegressor(
                n_estimators=100,
                max_depth=5,
                min_samples_leaf=3,
                random_state=42
            )
        )
    },
    "ayt_say": {
        "dosya": "data/veri_setleri/ayt_sayisal_veri_seti.csv",
        "model": make_pipeline(
            StandardScaler(),
            RandomForestRegressor(
                n_estimators=200,
                max_depth=5,
                min_samples_leaf=3,
                random_state=42
            )
        )
    },
    "ayt_sozel": {
        "dosya": "data/veri_setleri/ayt_sozel_veri_seti.csv",
        "model": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2),
            RandomForestRegressor(
                n_estimators=50,
                max_depth=3,
                min_samples_leaf=2,
                random_state=42
            )
        ),
        "optimize": True
    },
    "ayt_ea": {
        "dosya": "data/veri_setleri/ayt_ea_veri_seti.csv",
        "model": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(degree=2),
            LinearRegression()
        )
    },
    "ayt_dil": {
        "dosya": "data/veri_setleri/ayt_dil_veri_seti.csv",
        "model": make_pipeline(
            StandardScaler(),
            RandomForestRegressor(
                n_estimators=100,
                max_depth=4,
                min_samples_leaf=2,
                random_state=42
            )
        )
    }
}

def sozel_model_optimize_et(X, y):
    """AYT Sözel için en iyi parametreleri bulan fonksiyon"""
    
    # Temel model pipeline
    model_yapisi = make_pipeline(
        StandardScaler(),
        PolynomialFeatures(),
        RandomForestRegressor()
    )
    
    # Aranacak parametreler
    parametre_araligi = {
        'randomforestregressor__n_estimators': [50, 100],
        'randomforestregressor__max_depth': [2, 3, 4],
        'randomforestregressor__min_samples_leaf': [1, 2, 3]
    }
    
    izgara_arama = GridSearchCV(
        model_yapisi,
        parametre_araligi,
        cv=3,
        scoring='neg_mean_absolute_error',
        verbose=1,
        n_jobs=-1
    )
    
    izgara_arama.fit(X, y)
    return izgara_arama.best_estimator_

def model_egit_ve_degerlendir(X, y, model, sinav_turu):
    """Model eğitimi ve detaylı değerlendirme"""
    
    # Veri seti bölme
    test_boyutu = 0.3 if len(X) < 10 else 0.2
    X_egitim, X_test, y_egitim, y_test = train_test_split(
        X, y, test_size=test_boyutu, random_state=42, shuffle=True
    )
    
    # Model eğitimi
    model.fit(X_egitim, y_egitim)
    
    # Eğitim seti performansı
    y_egitim_tahmin = model.predict(X_egitim)
    egitim_r2 = r2_score(y_egitim, y_egitim_tahmin)
    egitim_mae = mean_absolute_error(y_egitim, y_egitim_tahmin)
    
    # Test seti performansı
    y_test_tahmin = model.predict(X_test)
    test_r2 = r2_score(y_test, y_test_tahmin)
    test_mae = mean_absolute_error(y_test, y_test_tahmin)
    test_rmse = np.sqrt(mean_squared_error(y_test, y_test_tahmin))
    
    print(f"\n{sinav_turu.upper()} Model Değerlendirmesi:")
    print(f"Veri seti boyutu: {len(X)} örnek")
    print(f"Eğitim seti: {len(X_egitim)} örnek")
    print(f"Test seti: {len(X_test)} örnek")
    print("\nEğitim Seti Performansı:")
    print(f"R² skoru: {egitim_r2:.4f}")
    print(f"MAE: {egitim_mae:.4f}")
    print("\nTest Seti Performansı:")
    print(f"R² skoru: {test_r2:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"RMSE: {test_rmse:.4f}")
    
    # Aşırı öğrenme kontrolü
    if egitim_r2 - test_r2 > 0.2:
        print("\nUYARI: Aşırı öğrenme tespit edildi!")
        print(f"Eğitim-Test R² farkı: {egitim_r2 - test_r2:.4f}")
    
    return model

def veri_dogrulama_ve_onisleme(veri, sinav_turu):
    """Veri setini doğrular ve ön işleme yapar"""
    
    # Veri seti boyut kontrolü
    if len(veri) < 10:
        print(f"UYARI: {sinav_turu} için veri seti çok küçük ({len(veri)} örnek)")
    
    # Sınav türüne göre ders sütunlarını belirle
    if sinav_turu == "tyt":
        ders_sutunlari = ["tyt_turkce", "tyt_matematik", "tyt_sosyal", "tyt_fen"]
    elif sinav_turu == "ayt_say":
        ders_sutunlari = ["ayt_matematik", "ayt_kimya", "ayt_biyoloji", "ayt_fizik"]
    elif sinav_turu == "ayt_ea":
        ders_sutunlari = ["ayt_matematik", "ayt_edebiyat", "ayt_cografya1"]
    elif sinav_turu == "ayt_sozel":
        ders_sutunlari = ["ayt_edebiyat", "ayt_tarih1", "ayt_cografya1", "ayt_tarih2", "ayt_cografya2", "ayt_felsefe", "ayt_din_kulturu"]
    elif sinav_turu == "ayt_dil":
        ders_sutunlari = ["ayt_dil"]
    else:
        raise ValueError(f"Geçersiz sınav türü: {sinav_turu}")
    
    # Gerekli sütunların varlığını kontrol et
    eksik_sutunlar = [col for col in ders_sutunlari + ["basari_sirasi"] if col not in veri.columns]
    if eksik_sutunlar:
        raise ValueError(f"Eksik sütunlar: {eksik_sutunlar}")
    
    # Veri tipi kontrolü ve dönüşümü
    for sutun in ders_sutunlari + ["basari_sirasi"]:
        veri[sutun] = pd.to_numeric(veri[sutun], errors='coerce')
    
    # Toplam net hesaplama
    veri['toplam_net'] = veri[ders_sutunlari].sum(axis=1)
    
    # Eksik değer kontrolü
    if veri.isnull().any().any():
        raise ValueError(f"Veri setinde eksik değerler var")
    
    # Değer aralığı kontrolü
    if veri['toplam_net'].max() > 120:
        raise ValueError("Net değerleri 120'den büyük olamaz")
    if veri['basari_sirasi'].min() < 1:
        raise ValueError("Sıralama 1'den küçük olamaz")
    
    # Aykırı değer tespiti (IQR yöntemi)
    Q1 = veri['toplam_net'].quantile(0.25)
    Q3 = veri['toplam_net'].quantile(0.75)
    IQR = Q3 - Q1
    alt_sinir = Q1 - 1.5 * IQR
    ust_sinir = Q3 + 1.5 * IQR
    aykiri_degerler = veri[(veri['toplam_net'] < alt_sinir) | (veri['toplam_net'] > ust_sinir)]
    
    if not aykiri_degerler.empty:
        print(f"UYARI: {len(aykiri_degerler)} adet aykırı değer tespit edildi")
        print(aykiri_degerler)
    
    return veri

# Ana döngü
model_performanslari = {}

# Modeller klasörünü oluştur
os.makedirs("modeller", exist_ok=True)

for ad, bilgiler in VERI_KAYNAKLARI.items():
    try:
        print(f"\n{'='*50}")
        print(f"{ad.upper()} modeli eğitiliyor...")
        
        # Veri yükleme ve ön işleme
        veri = pd.read_csv(bilgiler["dosya"])
        veri = veri_dogrulama_ve_onisleme(veri, ad)
        
        # Özellik ve hedef hazırlama
        X = veri[["toplam_net"]].values.reshape(-1, 1)
        y = veri["basari_sirasi"].values
        
        # Model seçimi ve eğitimi
        if ad == "ayt_sozel" and bilgiler.get("optimize", False):
            print("\nAYT Sözel için parametre optimizasyonu yapılıyor...")
            model = sozel_model_optimize_et(X, y)
        else:
            model = bilgiler["model"]
        
        # Model eğitimi ve değerlendirme
        model = model_egit_ve_degerlendir(X, y, model, ad)
        
        # Model kaydetme
        model_dosya_yolu = os.path.join("modeller", f"{ad}_model.pkl")
        with open(model_dosya_yolu, "wb") as f:
            pickle.dump(model, f)
        print(f"\nModel kaydedildi: {model_dosya_yolu}")
        
        # Performans bilgilerini sakla
        model_performanslari[ad] = {
            'ornek_sayisi': len(X),
            'model_yolu': model_dosya_yolu
        }
        
    except Exception as e:
        print(f"\nHATA - {ad}: {str(e)}")
        continue

# Sonuçları JSON olarak kaydet
import json
with open('model_performanslari.json', 'w') as f:
    json.dump(model_performanslari, f, indent=4)

print("\nTüm modeller eğitildi ve kaydedildi.")
