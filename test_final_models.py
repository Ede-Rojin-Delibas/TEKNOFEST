#!/usr/bin/env python3
"""
Final Model Testi - TYT Ensemble + AYT V3 Modeller
Bu script tüm sınav türleri için doğru modellerin kullanıldığını test eder.
"""

import requests
import json
import time
import logging

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API endpoint
BASE_URL = "http://localhost:5000/api"

def test_tyt_ensemble():
    """TYT Ensemble model testi"""
    logger.info("🧪 TYT Ensemble Model Testi")
    
    test_data = {
        "sinav_turu": "tyt",
        "dogru_yanlis": {
            "tyt_turkce": {"dogru": 18, "yanlis": 2},
            "tyt_matematik": {"dogru": 20, "yanlis": 0},
            "tyt_sosyal": {"dogru": 17, "yanlis": 3},
            "tyt_fen": {"dogru": 19, "yanlis": 1}
        },
        "hedef_siralama": 50000
    }
    
    try:
        response = requests.post(f"{BASE_URL}/tahmin", json=test_data)
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ TYT Test Başarılı!")
            logger.info(f"   Toplam Net: {result['net']['toplam']}")
            logger.info(f"   Tahmini Sıralama: {result['siralama']['tahmini']:,}")
            logger.info(f"   Değerlendirme: {result['siralama']['degerlendirme']}")
            return True
        else:
            logger.error(f"❌ TYT Test Başarısız: {response.status_code}")
            logger.error(response.text)
            return False
    except Exception as e:
        logger.error(f"❌ TYT Test Hatası: {str(e)}")
        return False

def test_ayt_sayisal():
    """AYT Sayısal V3 model testi"""
    logger.info("🧪 AYT Sayısal V3 Model Testi")
    
    test_data = {
        "sinav_turu": "ayt_sayisal",
        "dogru_yanlis": {
            "ayt_matematik": {"dogru": 20, "yanlis": 5},  # 25 soru
            "ayt_fizik": {"dogru": 15, "yanlis": 5},      # 20 soru
            "ayt_kimya": {"dogru": 13, "yanlis": 7},      # 20 soru
            "ayt_biyoloji": {"dogru": 15, "yanlis": 0}    # 15 soru
        },
        "hedef_siralama": 10000
    }
    
    try:
        response = requests.post(f"{BASE_URL}/tahmin", json=test_data)
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ AYT Sayısal Test Başarılı!")
            logger.info(f"   Toplam Net: {result['net']['toplam']}")
            logger.info(f"   Tahmini Sıralama: {result['siralama']['tahmini']:,}")
            return True
        else:
            logger.error(f"❌ AYT Sayısal Test Başarısız: {response.status_code}")
            logger.error(response.text)
            return False
    except Exception as e:
        logger.error(f"❌ AYT Sayısal Test Hatası: {str(e)}")
        return False

def test_ayt_ea():
    """AYT EA V3 model testi"""
    logger.info("🧪 AYT EA V3 Model Testi")
    
    test_data = {
        "sinav_turu": "ayt_ea",
        "dogru_yanlis": {
            "ayt_matematik": {"dogru": 20, "yanlis": 10},
            "ayt_edebiyat": {"dogru": 24, "yanlis": 0},
            "ayt_cografya1": {"dogru": 18, "yanlis": 2}
        },
        "hedef_siralama": 15000
    }
    
    try:
        response = requests.post(f"{BASE_URL}/tahmin", json=test_data)
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ AYT EA Test Başarılı!")
            logger.info(f"   Toplam Net: {result['net']['toplam']}")
            logger.info(f"   Tahmini Sıralama: {result['siralama']['tahmini']:,}")
            return True
        else:
            logger.error(f"❌ AYT EA Test Başarısız: {response.status_code}")
            logger.error(response.text)
            return False
    except Exception as e:
        logger.error(f"❌ AYT EA Test Hatası: {str(e)}")
        return False

def test_ayt_sozel():
    """AYT Sözel V3 model testi"""
    logger.info("🧪 AYT Sözel V3 Model Testi")
    
    test_data = {
        "sinav_turu": "ayt_sozel",
        "dogru_yanlis": {
            "ayt_edebiyat": {"dogru": 20, "yanlis": 4},      # 24 soru
            "ayt_cografya1": {"dogru": 5, "yanlis": 1}       # 6 soru
        },
        "hedef_siralama": 20000
    }
    
    try:
        response = requests.post(f"{BASE_URL}/tahmin", json=test_data)
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ AYT Sözel Test Başarılı!")
            logger.info(f"   Toplam Net: {result['net']['toplam']}")
            logger.info(f"   Tahmini Sıralama: {result['siralama']['tahmini']:,}")
            return True
        else:
            logger.error(f"❌ AYT Sözel Test Başarısız: {response.status_code}")
            logger.error(response.text)
            return False
    except Exception as e:
        logger.error(f"❌ AYT Sözel Test Hatası: {str(e)}")
        return False

def test_ayt_dil():
    """AYT Dil V3 model testi"""
    logger.info("🧪 AYT Dil V3 Model Testi")
    
    test_data = {
        "sinav_turu": "ayt_dil",
        "dogru_yanlis": {
            "ayt_dil": {"dogru": 75, "yanlis": 5}  # 80 soru, 75 doğru, 5 yanlış = 73.75 net
        },
        "hedef_siralama": 5000
    }
    
    try:
        response = requests.post(f"{BASE_URL}/tahmin", json=test_data)
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ AYT Dil Test Başarılı!")
            logger.info(f"   Toplam Net: {result['net']['toplam']}")
            logger.info(f"   Tahmini Sıralama: {result['siralama']['tahmini']:,}")
            return True
        else:
            logger.error(f"❌ AYT Dil Test Başarısız: {response.status_code}")
            logger.error(response.text)
            return False
    except Exception as e:
        logger.error(f"❌ AYT Dil Test Hatası: {str(e)}")
        return False

def main():
    """Ana test fonksiyonu"""
    logger.info("🚀 Final Model Testi Başlatılıyor")
    logger.info("=" * 60)
    
    # API'nin çalışıp çalışmadığını kontrol et
    try:
        response = requests.get(f"{BASE_URL}/health")
        if response.status_code != 200:
            logger.error("❌ API çalışmıyor! Önce 'python run.py' komutunu çalıştırın.")
            return
        logger.info("✅ API çalışıyor")
    except:
        logger.error("❌ API'ye bağlanılamıyor! Önce 'python run.py' komutunu çalıştırın.")
        return
    
    # Testleri çalıştır
    tests = [
        test_tyt_ensemble,
        test_ayt_sayisal,
        test_ayt_ea,
        test_ayt_sozel,
        test_ayt_dil
    ]
    
    results = {}
    for test in tests:
        results[test.__name__] = test()
        time.sleep(1)  # API'yi yormamak için bekle
    
    # Sonuçları özetle
    logger.info("\n" + "=" * 60)
    logger.info("TEST SONUÇLARI")
    logger.info("=" * 60)
    
    success_count = sum(results.values())
    total_count = len(results)
    
    for test_name, success in results.items():
        status = "✅ BAŞARILI" if success else "❌ BAŞARISIZ"
        logger.info(f"{test_name}: {status}")
    
    logger.info(f"\nToplam: {success_count}/{total_count} test başarılı")
    
    if success_count == total_count:
        logger.info("🎉 Tüm modeller başarıyla çalışıyor!")
        logger.info("\n📋 MODEL STRATEJİSİ:")
        logger.info("  • TYT: Ensemble Model (En iyi performans)")
        logger.info("  • AYT Sayısal: V3 Model")
        logger.info("  • AYT EA: V3 Model")
        logger.info("  • AYT Sözel: V3 Model")
        logger.info("  • AYT Dil: V3 Model")
    else:
        logger.error("⚠️  Bazı modellerde sorun var!")

if __name__ == "__main__":
    main() 