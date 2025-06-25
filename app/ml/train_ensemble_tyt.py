import pandas as pd
import numpy as np
import joblib
import json
import logging
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.base import BaseEstimator, TransformerMixin

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log', mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class NetHesaplayici(BaseEstimator, TransformerMixin):
    """Net hesaplama için özel transformer"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # X: [tyt_turkce, tyt_matematik, tyt_sosyal, tyt_fen]
        # Net hesaplama: (doğru - yanlış/4)
        netler = []
        for row in X:
            net = sum(row)  # Zaten net değerler
            netler.append(net)
        return np.array(netler).reshape(-1, 1)

class TYTEnsembleEgitici:
    def __init__(self, veri_klasoru: str, model_klasoru: str):
        self.veri_klasoru = Path(veri_klasoru)
        self.model_klasoru = Path(model_klasoru)
        self.baslangic_zamani = datetime.now()
        
        # Model tanımları
        self.model_tanimlari = {
            "random_forest": RandomForestRegressor(
                n_estimators=200,
                max_depth=8,
                min_samples_leaf=3,
                random_state=42,
                n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingRegressor(
                n_estimators=300,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                random_state=42
            ),
            "ridge": Ridge(alpha=1.0),
            "lasso": Lasso(alpha=0.01, random_state=42, max_iter=10000)
        }
        
        # Ensemble model
        self.ensemble_model = None
        
    def veri_yukle_ve_hazirla(self) -> tuple:
        """TYT veri setini yükler ve hazırlar"""
        # analiz.py tarafından oluşturulan filtrelenmiş dosyayı kullan
        dosya_yolu = self.veri_klasoru / "tyt_veri_seti_artirilmis_cleaned_filtered_filtered.csv"
        if not dosya_yolu.exists():
            raise FileNotFoundError(f"Filtrelenmiş veri seti bulunamadı: {dosya_yolu}")
        
        df = pd.read_csv(dosya_yolu)
        logger.info(f"TYT filtrelenmiş veri seti yüklendi. Boyut: {df.shape}")
        
        # Özellikler ve hedef
        X = df[['tyt_turkce', 'tyt_matematik', 'tyt_sosyal', 'tyt_fen']].values
        y = df['basari_sirasi'].values
        
        # Logaritmik dönüşüm analizler sonucu kaldırıldı.
        return X, y
    
    def model_egit(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Ensemble model eğitimi"""
        logger.info("TYT Ensemble model eğitimi başlatılıyor...")
        
        # Veri seti bölme
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Pipeline'lar oluştur
        pipelines = {}
        
        # Random Forest Pipeline
        rf_pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('rf', self.model_tanimlari['random_forest'])
        ])
        
        # Gradient Boosting Pipeline (PolynomialFeatures olmadan)
        gb_pipeline = Pipeline([
            ('scaler', RobustScaler()),
            ('gb', self.model_tanimlari['gradient_boosting'])
        ])
        
        # Modelleri eğit
        logger.info("Bireysel modeller eğitiliyor...")
        
        rf_pipeline.fit(X_train, y_train)
        gb_pipeline.fit(X_train, y_train)
        
        # Ensemble model oluştur - Sadece en iyi 2 model ile
        # Ağırlıkları GB'ye daha fazla önem verecek şekilde güncelleyelim
        self.ensemble_model = VotingRegressor([
            ('rf', rf_pipeline),
            ('gb', gb_pipeline)
        ], weights=[0.4, 0.6]) # RF: 40%, GB: 60%
        
        # Ensemble modeli eğit
        logger.info("Ensemble model eğitiliyor...")
        self.ensemble_model.fit(X_train, y_train)
        
        # Performans değerlendirme
        sonuclar = {}
        
        # Bireysel modeller
        for name, pipeline in [('rf', rf_pipeline), ('gb', gb_pipeline)]:
            y_pred = pipeline.predict(X_test)
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            
            sonuclar[name] = {
                'mae': mae,
                'rmse': rmse,
                'r2': r2
            }
            
            logger.info(f"{name.upper()} - MAE: {mae:,.0f}, RMSE: {rmse:,.0f}, R²: {r2:.4f}")
        
        # Ensemble model
        y_pred = self.ensemble_model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        sonuclar['ensemble'] = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
        
        logger.info(f"ENSEMBLE - MAE: {mae:,.0f}, RMSE: {rmse:,.0f}, R²: {r2:.4f}")
        
        # Cross-validation
        cv_scores = cross_val_score(self.ensemble_model, X, y, cv=5, 
                                   scoring='neg_mean_absolute_error', n_jobs=-1)
        cv_mae = -cv_scores.mean()
        logger.info(f"Cross-Validation MAE (Doğrudan): {cv_mae:,.0f}")
        
        return {
            'model': self.ensemble_model,
            'sonuclar': sonuclar,
            'cv_mae': cv_mae,
            'veri_boyutlari': {
                'toplam': len(X),
                'egitim': len(X_train),
                'test': len(X_test)
            }
        }
    
    def model_kaydet(self, model: BaseEstimator, sonuclar: dict):
        """Modeli ve sonuçları kaydeder"""
        # Model kaydet
        model_dosyasi = self.model_klasoru / "tyt_ensemble_model.joblib"
        joblib.dump(model, model_dosyasi)
        
        # Sonuçları kaydet
        sonuc_dosyasi = self.model_klasoru / "tyt_ensemble_sonuclar.json"
        with open(sonuc_dosyasi, 'w', encoding='utf-8') as f:
            json.dump(sonuclar, f, indent=4, default=str)
        
        logger.info("TYT Ensemble modeli ve sonuçları kaydedildi")
    
    def egit(self):
        """Ana eğitim fonksiyonu"""
        logger.info("TYT Ensemble Model Eğitimi Başlatılıyor")
        logger.info("=" * 50)
        
        try:
            # Veri yükle
            X, y = self.veri_yukle_ve_hazirla()
            
            # Model eğit
            baslangic = datetime.now()
            sonuc = self.model_egit(X, y)
            bitis = datetime.now()
            
            # Model kaydet
            self.model_kaydet(sonuc['model'], sonuc)
            
            # Süre raporu
            toplam_sure = bitis - baslangic
            logger.info(f"\nEğitim tamamlandı!")
            logger.info(f"Toplam süre: {toplam_sure}")
            logger.info("=" * 50)
            
            return sonuc
            
        except Exception as e:
            logger.error(f"Eğitim sırasında hata: {str(e)}")
            raise

if __name__ == "__main__":
    egitici = TYTEnsembleEgitici(
        veri_klasoru="data/veri_setleri/processed_v3",
        model_klasoru="data/models/v3"
    )
    egitici.egit()
