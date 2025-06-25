#!/usr/bin/env python3
"""
TYT Ensemble Model vs V3 Model Performans Karşılaştırması
Bu script iki farklı TYT modelinin performansını karşılaştırır.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_test_data():
    """Test verisi yükler"""
    data_path = Path("data/veri_setleri/processed_v3/tyt_veri_seti_artirilmis_cleaned_filtered_filtered.csv")
    if not data_path.exists():
        raise FileNotFoundError(f"Test verisi bulunamadı: {data_path}")
    
    df = pd.read_csv(data_path)
    logger.info(f"Test verisi yüklendi. Boyut: {df.shape}")
    
    # Özellikler ve hedef
    X = df[['tyt_turkce', 'tyt_matematik', 'tyt_sosyal', 'tyt_fen']].values
    y = df['basari_sirasi'].values
    
    return train_test_split(X, y, test_size=0.2, random_state=42)

def evaluate_model(model_path, X_test, y_test, model_name):
    """Model performansını değerlendirir"""
    try:
        model = joblib.load(model_path)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        logger.info(f"{model_name} Performansı:")
        logger.info(f"  MAE: {mae:,.0f}")
        logger.info(f"  RMSE: {rmse:,.0f}")
        logger.info(f"  R²: {r2:.4f}")
        
        return {
            'model_name': model_name,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'predictions': y_pred
        }
    except Exception as e:
        logger.error(f"{model_name} değerlendirilirken hata: {str(e)}")
        return None

def compare_models():
    """İki modeli karşılaştırır"""
    logger.info("TYT Model Performans Karşılaştırması Başlatılıyor")
    logger.info("=" * 60)
    
    # Test verisi yükle
    X_train, X_test, y_train, y_test = load_test_data()
    
    # Model yolları
    ensemble_path = Path("data/models/v3/tyt_ensemble_model.joblib")
    v3_path = Path("data/models/v3/tyt_model.joblib")
    
    results = {}
    
    # Ensemble model değerlendir
    if ensemble_path.exists():
        results['ensemble'] = evaluate_model(ensemble_path, X_test, y_test, "TYT Ensemble Model")
    else:
        logger.warning("TYT Ensemble model dosyası bulunamadı!")
    
    # V3 model değerlendir
    if v3_path.exists():
        results['v3'] = evaluate_model(v3_path, X_test, y_test, "TYT V3 Model")
    else:
        logger.warning("TYT V3 model dosyası bulunamadı!")
    
    # Karşılaştırma
    if 'ensemble' in results and 'v3' in results:
        logger.info("\n" + "=" * 60)
        logger.info("PERFORMANS KARŞILAŞTIRMASI")
        logger.info("=" * 60)
        
        ensemble = results['ensemble']
        v3 = results['v3']
        
        # MAE karşılaştırması
        mae_diff = ensemble['mae'] - v3['mae']
        mae_better = "Ensemble" if mae_diff < 0 else "V3"
        logger.info(f"MAE Farkı: {mae_diff:,.0f} ({mae_better} daha iyi)")
        
        # RMSE karşılaştırması
        rmse_diff = ensemble['rmse'] - v3['rmse']
        rmse_better = "Ensemble" if rmse_diff < 0 else "V3"
        logger.info(f"RMSE Farkı: {rmse_diff:,.0f} ({rmse_better} daha iyi)")
        
        # R² karşılaştırması
        r2_diff = ensemble['r2'] - v3['r2']
        r2_better = "Ensemble" if r2_diff > 0 else "V3"
        logger.info(f"R² Farkı: {r2_diff:.4f} ({r2_better} daha iyi)")
        
        # Genel değerlendirme
        ensemble_score = 0
        v3_score = 0
        
        if ensemble['mae'] < v3['mae']:
            ensemble_score += 1
        else:
            v3_score += 1
            
        if ensemble['rmse'] < v3['rmse']:
            ensemble_score += 1
        else:
            v3_score += 1
            
        if ensemble['r2'] > v3['r2']:
            ensemble_score += 1
        else:
            v3_score += 1
        
        logger.info(f"\nGenel Skor:")
        logger.info(f"  Ensemble: {ensemble_score}/3")
        logger.info(f"  V3: {v3_score}/3")
        
        if ensemble_score > v3_score:
            logger.info("🎉 TYT Ensemble Model daha iyi performans gösteriyor!")
            return "ensemble"
        elif v3_score > ensemble_score:
            logger.info("🎉 TYT V3 Model daha iyi performans gösteriyor!")
            return "v3"
        else:
            logger.info("🤝 Modeller eşit performans gösteriyor!")
            return "equal"
    
    return None

def test_sample_predictions():
    """Örnek tahminler yapar"""
    logger.info("\n" + "=" * 60)
    logger.info("ÖRNEK TAHMİNLER")
    logger.info("=" * 60)
    
    # Örnek net puanları
    sample_data = [
        {"name": "Yüksek Başarı", "tyt_turkce": 18, "tyt_matematik": 20, "tyt_sosyal": 17, "tyt_fen": 19},
        {"name": "Orta Başarı", "tyt_turkce": 15, "tyt_matematik": 12, "tyt_sosyal": 14, "tyt_fen": 13},
        {"name": "Düşük Başarı", "tyt_turkce": 8, "tyt_matematik": 6, "tyt_sosyal": 9, "tyt_fen": 7}
    ]
    
    ensemble_path = Path("data/models/v3/tyt_ensemble_model.joblib")
    v3_path = Path("data/models/v3/tyt_model.joblib")
    
    if ensemble_path.exists() and v3_path.exists():
        ensemble_model = joblib.load(ensemble_path)
        v3_model = joblib.load(v3_path)
        
        for sample in sample_data:
            X = np.array([[
                sample["tyt_turkce"], 
                sample["tyt_matematik"], 
                sample["tyt_sosyal"], 
                sample["tyt_fen"]
            ]])
            
            ensemble_pred = int(ensemble_model.predict(X)[0])
            v3_pred = int(v3_model.predict(X)[0])
            
            toplam_net = sample["tyt_turkce"] + sample["tyt_matematik"] + sample["tyt_sosyal"] + sample["tyt_fen"]
            
            logger.info(f"\n{sample['name']}:")
            logger.info(f"  Net: {sample['tyt_turkce']} + {sample['tyt_matematik']} + {sample['tyt_sosyal']} + {sample['tyt_fen']} = {toplam_net}")
            logger.info(f"  Ensemble Tahmin: {ensemble_pred:,}")
            logger.info(f"  V3 Tahmin: {v3_pred:,}")
            logger.info(f"  Fark: {abs(ensemble_pred - v3_pred):,}")

if __name__ == "__main__":
    try:
        # Model karşılaştırması
        best_model = compare_models()
        
        # Örnek tahminler
        test_sample_predictions()
        
        if best_model:
            logger.info(f"\nÖNERİ: {best_model.upper()} modelini kullanın!")
        
    except Exception as e:
        logger.error(f"Karşılaştırma sırasında hata: {str(e)}") 