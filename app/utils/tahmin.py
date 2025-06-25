import joblib
import os
import numpy as np
from typing import Union, Tuple, Dict, Any
from config import Config
import logging
from sklearn.preprocessing import RobustScaler, PolynomialFeatures
from pathlib import Path
import json
import pandas as pd

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Model seçim stratejisi - Performans testleri sonucu belirlendi
MODEL_STRATEGY = {
    "tyt": "ensemble",  # Ensemble model çok daha iyi performans veriyor (MAE: 81k vs 166k)
    "ayt_sayisal": "v3",
    "ayt_ea": "v3", 
    "ayt_sozel": "v3",
    "ayt_dil": "v3"
}

# Sınav türlerine göre gerçekçi sıralama aralıkları
SINAV_SIRALAMA_ARALIKLARI = {
    "tyt": {
        "min": 1,
        "max": 1_500_000,  # TYT'de yaklaşık 1.5 milyon aday
        "net_esikleri": {
            "cok_iyi": 85,    # İlk 10,000
            "iyi": 75,        # İlk 100,000
            "orta": 60,       # İlk 500,000
            "normal": 45      # İlk 1,000,000
        }
    },
    "ayt_sayisal": {
        "min": 1,
        "max": 500_000,      # Sayısal'da yaklaşık 500,000 aday
        "net_esikleri": {
            "cok_iyi": 70,    # İlk 5,000
            "iyi": 60,        # İlk 50,000
            "orta": 45,       # İlk 200,000
            "normal": 30      # İlk 400,000
        }
    },
    "ayt_ea": {
        "min": 1,
        "max": 400_000,      # EA'da yaklaşık 400,000 aday
        "net_esikleri": {
            "cok_iyi": 75,    # İlk 5,000
            "iyi": 65,        # İlk 50,000
            "orta": 50,       # İlk 200,000
            "normal": 35      # İlk 300,000
        }
    },
    "ayt_sozel": {
        "min": 1,
        "max": 300_000,      # Sözel'de yaklaşık 300,000 aday
        "net_esikleri": {
            "cok_iyi": 26,    # İlk 5,000 (30 soru üzerinden)
            "iyi": 22,        # İlk 50,000
            "orta": 18,       # İlk 200,000
            "normal": 14      # İlk 250,000
        }
    },
    "ayt_dil": {
        "min": 1,
        "max": 100_000,      # Dil'de yaklaşık 100,000 aday
        "net_esikleri": {
            "cok_iyi": 70,    # İlk 1,000
            "iyi": 60,        # İlk 10,000
            "orta": 45,       # İlk 50,000
            "normal": 30      # İlk 80,000
        }
    }
}

def _net_to_rank(net: float, sinav_turu: str) -> int:
    """Net puanına göre sıralama tahmini yapar."""
    aralik = SINAV_SIRALAMA_ARALIKLARI[sinav_turu]
    esikler = aralik["net_esikleri"]
    
    # TYT için özel düşük net düzeltmesi
    if sinav_turu == "tyt":
        if net < 45:
            # Düşük netlerde daha yumuşak bir cezalandırma
            net = 45 + (net - 45) * 0.3  # 0.5'ten 0.3'e düşürüldü
        elif net < 60:
            # Orta-düşük netlerde özel düzeltme
            net = 60 + (net - 60) * 0.6  # 0.8'den 0.6'ya düşürüldü
        elif net < 75:
            # Orta-yüksek netlerde düzeltme
            net = 75 + (net - 75) * 0.8
    
    # AYT Sayısal için orta seviye net düzeltmesi
    elif sinav_turu == "ayt_sayisal":
        if net < 45:
            # Düşük netlerde daha yumuşak cezalandırma
            net = 45 + (net - 45) * 0.4
        elif net < 60:
            # Orta seviye netlerde daha hassas tahmin
            net = 60 + (net - 60) * 0.7  # 0.9'dan 0.7'ye düşürüldü
    
    # AYT Sözel için özel düzeltme
    elif sinav_turu == "ayt_sozel":
        if net < 18:
            # Düşük netlerde daha yumuşak cezalandırma
            net = 18 + (net - 18) * 0.4
        elif net < 22:
            # Orta seviye netlerde daha hassas tahmin
            net = 22 + (net - 22) * 0.7
    
    if net >= esikler["cok_iyi"]:
        return int(aralik["min"] + (aralik["max"] - aralik["min"]) * 0.01)
    elif net >= esikler["iyi"]:
        return int(aralik["min"] + (aralik["max"] - aralik["min"]) * 0.1)
    elif net >= esikler["orta"]:
        return int(aralik["min"] + (aralik["max"] - aralik["min"]) * 0.3)
    elif net >= esikler["normal"]:
        return int(aralik["min"] + (aralik["max"] - aralik["min"]) * 0.7)
    else:
        return int(aralik["min"] + (aralik["max"] - aralik["min"]) * 0.9)

def _normalize_prediction(tahmin: float, sinav_turu: str) -> int:
    """
    Model tahminini gerçekçi bir sıralamaya dönüştürür.
    
    Args:
        tahmin (float): Model tahmini
        sinav_turu (str): Sınav türü
        
    Returns:
        int: Normalize edilmiş sıralama
    """
    aralik = SINAV_SIRALAMA_ARALIKLARI[sinav_turu]
    
    # Model tahminini logaritmik olarak normalize et
    # Bu, yüksek sıralamaları daha gerçekçi hale getirir
    log_min = np.log1p(aralik["min"])
    log_max = np.log1p(aralik["max"])
    log_tahmin = np.log1p(max(1, tahmin))
    
    # Logaritmik değeri 0-1 arasına normalize et
    normalized = (log_tahmin - log_min) / (log_max - log_min)
    normalized = max(0, min(1, normalized))
    
    # Normalize edilmiş değeri gerçek sıralama aralığına dönüştür
    rank = int(np.expm1(log_min + normalized * (log_max - log_min)))
    
    return rank

def tahmin_yap(sinav_turu: str, dogru_yanlis: Dict[str, Dict[str, int]], hedef_siralama: int = None) -> Dict[str, Any]:
    """
    Sınav türü ve net puanlara göre sıralama tahmini yapar.
    """
    try:
        # Sınav türü kontrolü
        if sinav_turu not in SINAV_SIRALAMA_ARALIKLARI:
            raise ValueError(f"Geçersiz sınav türü: {sinav_turu}")
        
        # Toplam net hesapla
        toplam_net = 0
        ders_netleri = {}
        for ders, sonuc in dogru_yanlis.items():
            if "dogru" not in sonuc or "yanlis" not in sonuc:
                raise ValueError(f"Eksik veri: {ders} için doğru/yanlış sayısı gerekli")
            ders_neti = sonuc["dogru"] - (sonuc["yanlis"] * 0.25)  # 4 yanlış 1 doğruyu götürür
            ders_netleri[ders] = round(ders_neti, 2)
            toplam_net += ders_neti
        
        # Net değerini sınırla
        toplam_net = max(0, min(100, toplam_net))
        logger.info(f"Toplam net hesaplandı: {toplam_net}")
        
        # Model tahmini yap
        if sinav_turu == "tyt":
            if MODEL_STRATEGY["tyt"] == "ensemble":
                model_path = Path(f"data/models/v3/tyt_ensemble_model.joblib")
            else:
                model_path = Path(f"data/models/v3/tyt_model.joblib")
        else:
            model_path = Path(f"data/models/v3/{sinav_turu}_model.joblib")

        if not model_path.exists():
            raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
        
        try:
            # Modeli yükle
            pipeline = joblib.load(model_path)
            logger.info(f"Model yüklendi: {model_path}")
        except Exception as e:
            logger.error(f"Model yüklenirken hata: {str(e)}")
            raise
        
        try:
            # Ders bazlı net puanları modele gönder
            if sinav_turu == "tyt":
                # TYT: Türkçe, Matematik, Sosyal, Fen
                X = np.array([[
                    ders_netleri.get("tyt_turkce", 0),
                    ders_netleri.get("tyt_matematik", 0),
                    ders_netleri.get("tyt_sosyal", 0),
                    ders_netleri.get("tyt_fen", 0)
                ]])
            elif sinav_turu == "ayt_sayisal":
                # AYT Sayısal: Matematik, Fizik, Kimya, Biyoloji
                X = np.array([[
                    ders_netleri.get("ayt_matematik", 0),
                    ders_netleri.get("ayt_fizik", 0),
                    ders_netleri.get("ayt_kimya", 0),
                    ders_netleri.get("ayt_biyoloji", 0)
                ]])
            elif sinav_turu == "ayt_ea":
                # AYT EA: Matematik, Edebiyat, Coğrafya1
                X = np.array([[
                    ders_netleri.get("ayt_matematik", 0),
                    ders_netleri.get("ayt_edebiyat", 0),
                    ders_netleri.get("ayt_cografya1", 0)
                ]])
            elif sinav_turu == "ayt_sozel":
                # AYT Sözel: Edebiyat, Coğrafya1
                X = np.array([[
                    ders_netleri.get("ayt_edebiyat", 0),
                    ders_netleri.get("ayt_cografya1", 0)
                ]])
            elif sinav_turu == "ayt_dil":
                # AYT Dil: Sadece dil neti
                X = np.array([[
                    ders_netleri.get("ayt_dil", 0)
                ]])
            else:
                raise ValueError(f"Desteklenmeyen sınav türü: {sinav_turu}")
            
            # Model tahmini
            model_tahmini = pipeline.predict(X)[0]
            logger.info(f"Modelin ham tahmini: {model_tahmini}")

            # TYT için, modelin tahminini net puanına göre basitçe kalibre et
            if sinav_turu == "tyt":
                if toplam_net > 90: # Yüksek netler
                    # Beklenen sıralama aralığını daralt
                    tahmini_siralama = np.clip(model_tahmini, 1, 30000)
                elif toplam_net > 70: # İyi netler
                    tahmini_siralama = np.clip(model_tahmini, 20000, 150000)
                elif toplam_net < 30: # Düşük netler
                    # Düşük netler için sıralamayı daha yukarıda tut
                    tahmini_siralama = max(model_tahmini, 800000)
                else:
                    tahmini_siralama = model_tahmini
            else:
                # Diğer sınav türleri için doğrudan model tahminini kullan
                tahmini_siralama = model_tahmini

            logger.info(f"Kalibre edilmiş tahmin: {tahmini_siralama}")
            tahmini_siralama = int(tahmini_siralama)
            
        except Exception as e:
            logger.error(f"Tahmin yapılırken hata: {str(e)}")
            raise
        
        # Tahmini sıralamayı normalize et
        normalize_edilmis_siralama = _normalize_prediction(tahmini_siralama, sinav_turu)
        
        # Hedef sıralama ile karşılaştır
        if hedef_siralama:
            fark = abs(normalize_edilmis_siralama - hedef_siralama)
            if normalize_edilmis_siralama <= hedef_siralama * 0.9:
                degerlendirme = "Hedefinizin üzerindesiniz, tebrikler!"
            elif normalize_edilmis_siralama <= hedef_siralama * 1.1:
                degerlendirme = "Hedefinize çok yakınsınız, bu tempoyu koruyun!"
            elif normalize_edilmis_siralama <= hedef_siralama * 1.3:
                degerlendirme = "Hedefinize yaklaşıyorsunuz, biraz daha gayret!"
            else:
                degerlendirme = "Hedefinize ulaşmak için daha fazla çalışmalısınız."
        else:
            fark = 0
            degerlendirme = "Tahmin tamamlandı."
        
        return {
            "net": {
                "dersler": ders_netleri,
                "toplam": round(toplam_net, 2)
            },
            "siralama": {
                "tahmini": normalize_edilmis_siralama,
                "ham_tahmin": tahmini_siralama,
                "hedef": hedef_siralama,
                "fark": fark,
                "degerlendirme": degerlendirme
            }
        }
        
    except Exception as e:
        logger.error(f"Tahmin yapılırken hata: {str(e)}")
        raise

def tyt_ensemble_predict(ders_netleri):
    toplam_net = sum(ders_netleri.values())
    if toplam_net < 45:
        model_path = "data/models/v3/tyt_dusuk_model.joblib"
    elif toplam_net < 65:
        model_path = "data/models/v3/tyt_orta_model.joblib"
    else:
        model_path = "data/models/v3/tyt_yuksek_model.joblib"
    model = joblib.load(model_path)
    X = np.array([[ders_netleri.get("tyt_turkce",0), ders_netleri.get("tyt_matematik",0),
                   ders_netleri.get("tyt_fen",0), ders_netleri.get("tyt_sosyal",0)]])
    tahmin = model.predict(X)[0]
    # Quantile mapping uygula
    quantiles = pd.read_csv("data/models/v3/tyt_net_quantiles.csv", index_col=0)
    if toplam_net < 45:
        q = quantiles.iloc[0]
    elif toplam_net < 65:
        q = quantiles.iloc[1]
    else:
        q = quantiles.iloc[2]
    # Tahmini, o aralıktaki min-median-max ile sınırla
    tahmin = max(q["basari_sirasi"]["min"], min(q["basari_sirasi"]["max"], tahmin))
    return int(tahmin)

