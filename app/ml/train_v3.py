import pandas as pd
import numpy as np
import os
import joblib
import json
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from datetime import datetime

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.base import BaseEstimator
from xgboost import XGBRegressor

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log', mode='w'),  # Her çalıştırmada yeni log dosyası
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# GridSearchCV için logging
logging.getLogger('sklearn.model_selection').setLevel(logging.INFO)

# Artırılmış veri seti yolları - Bu genel yol yerine dinamik yol oluşturulacak
# DATA_PATHS = { ... }

MODEL_SAVE_DIR = Path("data/models/v3")
MODEL_SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Sınav türlerine göre kullanılacak özellikler ve hedef
PARAMS = {
    "tyt": {
        "features": ["tyt_turkce", "tyt_matematik", "tyt_fen", "tyt_sosyal"],
        "target": "basari_sirasi"
    },
    "ayt_sayisal": {
        "features": ["ayt_matematik", "ayt_fizik", "ayt_kimya", "ayt_biyoloji"],
        "target": "basari_sirasi"
    },
    "ayt_ea": {
        "features": ["ayt_matematik", "ayt_edebiyat", "ayt_cografya1"],
        "target": "basari_sirasi"
    },
    "ayt_sozel": {
        "features": ["ayt_edebiyat", "ayt_cografya1"],
        "target": "basari_sirasi"
    },
    "ayt_dil": {
        "features": ["ayt_dil"],
        "target": "basari_sirasi"
    }
}

class ModelEgitici:
    """Model eğitimi ve değerlendirme işlemlerini yöneten sınıf."""
    
    def __init__(self, veri_klasoru: str, model_klasoru: str):
        """
        Args:
            veri_klasoru: İşlenmiş veri setlerinin bulunduğu klasör
            model_klasoru: Eğitilmiş modellerin kaydedileceği klasör
        """
        self.veri_klasoru = Path(veri_klasoru)
        self.model_klasoru = Path(model_klasoru)
        self.model_klasoru.mkdir(parents=True, exist_ok=True)
        
        # Başlangıç zamanını kaydet
        self.baslangic_zamani = datetime.now()
        logger.info(f"Model eğitimi başlatılıyor: {self.baslangic_zamani}")
        
        # Model tanımları
        self.model_tanimlari = {
            "tyt": {
                "model": make_pipeline(
                    RobustScaler(),
                    PolynomialFeatures(degree=2),
                    GradientBoostingRegressor(
                        n_estimators=200,
                        max_depth=4,
                        learning_rate=0.05,
                        min_samples_leaf=2,
                        subsample=0.8,
                        random_state=42,
                        verbose=2,
                        n_iter_no_change=10
                    )
                ),
                "optimize": True,
                "parametre_araligi": {
                    'gradientboostingregressor__n_estimators': [150, 200],
                    'gradientboostingregressor__max_depth': [3, 4],
                    'gradientboostingregressor__learning_rate': [0.05],
                    'polynomialfeatures__degree': [2]
                }
            },
            "ayt_sayisal": {
                "model": make_pipeline(
                    RobustScaler(),
                    RandomForestRegressor(
                        n_estimators=150,
                        max_depth=5,
                        min_samples_leaf=2,
                        random_state=42,
                        n_jobs=-1,
                        verbose=2
                    )
                ),
                "optimize": True,
                "parametre_araligi": {
                    'randomforestregressor__n_estimators': [100, 150, 200],
                    'randomforestregressor__max_depth': [5, 7, 9],
                    'randomforestregressor__min_samples_leaf': [2, 3]
                }
            },
            "ayt_ea": {
                "model": make_pipeline(
                    RobustScaler(),
                    PolynomialFeatures(degree=2),
                    GradientBoostingRegressor(
                        n_estimators=100,
                        max_depth=3,
                        learning_rate=0.1,
                        min_samples_leaf=4,
                        random_state=42,
                        verbose=0,
                        n_iter_no_change=10
                    )
                ),
                "optimize": True,
                "parametre_araligi": {
                    'gradientboostingregressor__n_estimators': [100, 150],
                    'gradientboostingregressor__max_depth': [3, 4],
                    'gradientboostingregressor__learning_rate': [0.05, 0.1],
                    'polynomialfeatures__degree': [2]
                }
            },
            "ayt_sozel": {
                "model": make_pipeline(
                    RobustScaler(),
                    PolynomialFeatures(degree=2),
                    Ridge()
                ),
                "optimize": True,
                "parametre_araligi": {
                    'polynomialfeatures__degree': [2, 3],
                    'ridge__alpha': [0.1, 0.5, 1.0, 5.0, 10.0]
                }
            },
            "ayt_dil": {
                "model": make_pipeline(
                    RobustScaler(),
                    RandomForestRegressor(
                        n_estimators=75,
                        max_depth=3,
                        min_samples_leaf=5,
                        random_state=42,
                        n_jobs=-1,
                        verbose=0
                    )
                ),
                "optimize": True,
                "parametre_araligi": {
                    'randomforestregressor__n_estimators': [50, 75, 100],
                    'randomforestregressor__max_depth': [3, 4, 5],
                    'randomforestregressor__min_samples_leaf': [3, 5]
                }
            }
        }
        
        # Model performans metriklerini saklamak için
        self.performans_metrikleri = {}
    
    def veri_yukle(self, sinav_turu: str) -> Optional[Tuple[pd.DataFrame, pd.Series]]:
        """Belirtilen sınav türü için filtrelenmiş veri setini yükler."""
        # analiz.py tarafından oluşturulan filtrelenmiş dosya adını hedefliyoruz.
        # Örnek: tyt_veri_seti_artirilmis_cleaned_filtered_filtered.csv
        # Bu yapı biraz karmaşık, bu yüzden daha basit bir eşleme yapalım.
        dosya_eslestirme = {
            "tyt": "tyt_veri_seti_artirilmis_cleaned_filtered_filtered.csv",
            "ayt_sayisal": "ayt_sayisal_veri_seti_artirilmis_cleaned_filtered_filtered.csv",
            "ayt_ea": "ayt_ea_veri_seti_artirilmis_cleaned_cleaned_filtered.csv",
            "ayt_sozel": "ayt_sozel_veri_seti_artirilmis_cleaned_cleaned_filtered.csv",
            "ayt_dil": "ayt_dil_veri_seti_artirilmis_cleaned_cleaned_filtered.csv"
        }
        
        dosya_adi = dosya_eslestirme.get(sinav_turu)
        if not dosya_adi:
            logger.error(f"'{sinav_turu}' için dosya eşleştirmesi bulunamadı.")
            return None
            
        dosya_yolu = self.veri_klasoru / dosya_adi
        
        if not dosya_yolu.exists():
            logger.error(f"Veri seti bulunamadı: {dosya_yolu}")
            return None
        
        df = pd.read_csv(dosya_yolu)
        
        ozellikler = PARAMS[sinav_turu]["features"]
        hedef = PARAMS[sinav_turu]["target"]
        
        # Özelliklerin veri setinde olup olmadığını kontrol et
        eksik_ozellikler = [col for col in ozellikler if col not in df.columns]
        if eksik_ozellikler:
            logger.error(f"'{sinav_turu}' veri setinde şu özellikler eksik: {eksik_ozellikler}")
            return None
            
        X = df[ozellikler]
        y = df[hedef]
        
        logger.info(f"{sinav_turu} veri seti yüklendi. Boyut: {X.shape}")
        return X, y
    
    def model_optimize_et(self, X: pd.DataFrame, y: pd.Series, sinav_turu: str) -> BaseEstimator:
        """Model hiperparametrelerini optimize eder."""
        if not self.model_tanimlari[sinav_turu]["optimize"]:
            return self.model_tanimlari[sinav_turu]["model"]
        
        logger.info(f"\n{sinav_turu} için parametre optimizasyonu başlatılıyor")
        logger.info(f"Test edilecek parametre kombinasyonları: {self.model_tanimlari[sinav_turu]['parametre_araligi']}")
        
        izgara_arama = GridSearchCV(
            self.model_tanimlari[sinav_turu]["model"],
            self.model_tanimlari[sinav_turu]["parametre_araligi"],
            cv=3,
            scoring='neg_mean_absolute_error',
            verbose=2,
            n_jobs=-1
        )
        
        baslangic = datetime.now()
        izgara_arama.fit(X.values, y.values)
        bitis = datetime.now()
        
        logger.info(f"Optimizasyon tamamlandı. Süre: {bitis - baslangic}")
        logger.info(f"En iyi parametreler: {izgara_arama.best_params_}")
        logger.info(f"En iyi skor: {-izgara_arama.best_score_:.4f}")
        
        return izgara_arama.best_estimator_
    
    def model_egit_ve_degerlendir(self, X: pd.DataFrame, y: pd.Series, sinav_turu: str) -> Optional[Dict[str, Any]]:
        """Model eğitimi ve değerlendirmesi yapar."""
        logger.info(f"{sinav_turu} model eğitimi başlatılıyor")
        
        # Model optimizasyonu
        try:
            model = self.model_optimize_et(X, y, sinav_turu)
        except Exception as e:
            logger.error(f"'{sinav_turu}' için model optimizasyonu başarısız oldu: {e}")
            return None
        
        # Veri seti bölme - Stratified sampling için y'yi aralıklara böl
        # Negatif değerler olabileceğinden qcut sorun çıkarabilir, cut kullanalım
        y_bins = pd.cut(y, bins=5, labels=False)
        
        X_egitim, X_test, y_egitim, y_test = train_test_split(
            X.values, y.values, test_size=0.2, random_state=42, stratify=y_bins
        )
        
        # Model eğitimi
        model.fit(X_egitim, y_egitim)
        
        # Cross-validation
        try:
            cv_scores = cross_val_score(model, X.values, y.values, cv=5, scoring='neg_mean_absolute_error', n_jobs=-1)
            cv_mae = -cv_scores.mean()
        except Exception as e:
            logger.warning(f"'{sinav_turu}' için Cross-validation sırasında bir hata oluştu: {e}")
            cv_mae = -1 # Hata durumunda -1 ata
        
        # Performans metrikleri
        metrikler = {
            "egitim": self._performans_hesapla(model, X_egitim, y_egitim),
            "test": self._performans_hesapla(model, X_test, y_test),
            "cv_mae": cv_mae,
            "veri_boyutlari": {
                "toplam": X.shape[0],
                "egitim": X_egitim.shape[0],
                "test": X_test.shape[0]
            }
        }
        
        self._performans_raporu(sinav_turu, metrikler)
        self.model_kaydet(model, sinav_turu, metrikler)
        
        return metrikler
    
    def _performans_hesapla(self, model: BaseEstimator, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Verilen veri seti üzerinde modelin performansını hesaplar."""
        tahminler = model.predict(X)
        
        # y ve tahminlerde NaN veya sonsuz değer olmadığından emin ol
        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(tahminler)):
            logger.error("Hesaplama sırasında geçersiz değerler (NaN/inf) bulundu.")
            return {"mae": -1, "rmse": -1, "r2": -1}

        mae = mean_absolute_error(y, tahminler)
        rmse = np.sqrt(mean_squared_error(y, tahminler))
        r2 = r2_score(y, tahminler)
        
        return {"mae": mae, "rmse": rmse, "r2": r2}
    
    def _performans_raporu(self, sinav_turu: str, metrikler: Dict[str, Any]):
        """Model performans raporunu oluşturur ve loglar."""
        logger.info(f"\n{sinav_turu.upper()} Model Değerlendirmesi:")
        logger.info("\nEğitim Seti Performansı:")
        logger.info(f"R² skoru: {metrikler['egitim']['r2']:.4f}")
        logger.info(f"MAE: {metrikler['egitim']['mae']:.4f}")
        logger.info(f"RMSE: {metrikler['egitim']['rmse']:.4f}")
        
        logger.info("\nTest Seti Performansı:")
        logger.info(f"R² skoru: {metrikler['test']['r2']:.4f}")
        logger.info(f"MAE: {metrikler['test']['mae']:.4f}")
        logger.info(f"RMSE: {metrikler['test']['rmse']:.4f}")
        
        logger.info(f"\nCross-Validation MAE: {metrikler['cv_mae']:.4f}")
    
    def model_kaydet(self, model: BaseEstimator, sinav_turu: str, metrikler: Dict[str, Any]):
        """Eğitilmiş modeli ve performans metriklerini kaydeder."""
        # Model kaydet
        model_dosyasi = self.model_klasoru / f"{sinav_turu}_model.joblib"
        joblib.dump(model, model_dosyasi)
        
        # Performans metriklerini kaydet
        metrik_dosyasi = self.model_klasoru / f"{sinav_turu}_metrikler.json"
        with open(metrik_dosyasi, 'w', encoding='utf-8') as f:
            json.dump(metrikler, f, indent=4)
        
        logger.info(f"{sinav_turu} modeli ve metrikleri kaydedildi")
    
    def tum_modelleri_egit(self):
        """Tüm sınav türleri için modelleri eğitir."""
        
        for sinav_turu in self.model_tanimlari.keys():
            logger.info(f"\n{'='*20} {sinav_turu.upper()} İŞLEMİ BAŞLATILIYOR {'='*20}")
            
            veri = self.veri_yukle(sinav_turu)
            if veri is None:
                logger.error(f"'{sinav_turu}' için veri yüklenemedi. Bu sınav türü atlanıyor.")
                continue

            X, y = veri
            
            # Negatif veya çok büyük değerler varsa log dönüşümü sorun çıkarabilir.
            # Analiz sonucunda log dönüşümünün gereksiz olduğuna karar verdik.
            
            sonuclar = self.model_egit_ve_degerlendir(X, y, sinav_turu)
            
            if sonuclar:
                self.performans_metrikleri[sinav_turu] = sonuclar

        self._genel_performans_raporu()

    def _genel_performans_raporu(self):
        """Tüm modellerin genel performans raporunu oluşturur."""
        logger.info("\n=== GENEL MODEL PERFORMANS RAPORU ===")
        
        for sinav_turu, sonuc in self.performans_metrikleri.items():
            logger.info(f"\n{sinav_turu.upper()}:")
            # Raporlamada 'veri_boyutlari' anahtarının varlığını kontrol et
            if 'veri_boyutlari' in sonuc and 'toplam' in sonuc['veri_boyutlari']:
                 logger.info(f"Veri seti boyutu: {sonuc['veri_boyutlari']['toplam']}")
            if 'test' in sonuc and 'r2' in sonuc['test']:
                logger.info(f"Test R²: {sonuc['test']['r2']:.4f}")
            if 'test' in sonuc and 'mae' in sonuc['test']:
                logger.info(f"Test MAE: {sonuc['test']['mae']:.4f}")

def main():
    """Ana eğitim fonksiyonu."""
    # Veri ve model klasör yolları güncellendi
    veri_yolu = Path("data/veri_setleri/processed_v3")
    model_yolu = Path("data/models/v3")
    
    egitici = ModelEgitici(veri_klasoru=str(veri_yolu), model_klasoru=str(model_yolu))
    egitici.tum_modelleri_egit()

if __name__ == '__main__':
    main() 