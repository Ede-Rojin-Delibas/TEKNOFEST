import unittest
import requests
import json
import logging
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('test_api.log', mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SinavTahminTestCase(unittest.TestCase):
    def setUp(self):
        """Test öncesi hazırlıklar"""
        self.base_url = "http://localhost:5000/api"
        self.headers = {"Content-Type": "application/json"}
        
        # Test verileri - Gerçek veri setinden alınmış senaryolar
        self.test_senaryolari = {
            "tyt": {
                "sinav_turu": "tyt",
                "test_verileri": [
                    {
                        "dogru_yanlis": {
                            "tyt_turkce": {"dogru": 35, "yanlis": 5},      # Net: 33.75
                            "tyt_matematik": {"dogru": 35, "yanlis": 5},   # Net: 33.75
                            "tyt_sosyal": {"dogru": 17, "yanlis": 3},    # Net: 16.25
                            "tyt_fen": {"dogru": 17, "yanlis": 3}         # Net: 16.25
                        },                                             # Toplam Net: 100.00
                        "hedef_siralama": 2000 
                    },
                    {
                        "dogru_yanlis": {
                            "tyt_turkce": {"dogru": 30, "yanlis": 10},     # Net: 27.5
                            "tyt_matematik": {"dogru": 30, "yanlis": 10},  # Net: 27.5
                            "tyt_sosyal": {"dogru": 15, "yanlis": 5},     # Net: 13.75
                            "tyt_fen": {"dogru": 15, "yanlis": 5}         # Net: 13.75
                        },                                             # Toplam Net: 82.5
                        "hedef_siralama": 35000
                    },
                    {
                        "dogru_yanlis": {
                            "tyt_turkce": {"dogru": 25, "yanlis": 15},     # Net: 21.25
                            "tyt_matematik": {"dogru": 20, "yanlis": 20},  # Net: 15.0
                            "tyt_sosyal": {"dogru": 12, "yanlis": 8},     # Net: 10.0
                            "tyt_fen": {"dogru": 12, "yanlis": 8}         # Net: 10.0
                        },                                             # Toplam Net: 56.25
                        "hedef_siralama": 400000
                    },
                    {
                        "dogru_yanlis": {
                            "tyt_turkce": {"dogru": 15, "yanlis": 20},    # Net: 10.0
                            "tyt_matematik": {"dogru": 5, "yanlis": 10},   # Net: 2.5
                            "tyt_sosyal": {"dogru": 8, "yanlis": 8},      # Net: 6.0
                            "tyt_fen": {"dogru": 5, "yanlis": 10}         # Net: 2.5
                        },                                             # Toplam Net: 21.0
                        "hedef_siralama": 1200000
                    }
                ]
            },
            "ayt_sayisal": {
                "sinav_turu": "ayt_sayisal",
                "test_verileri": [
                    {
                        "dogru_yanlis": {
                            "ayt_matematik": {"dogru": 36, "yanlis": 4},
                            "ayt_fizik": {"dogru": 12, "yanlis": 2},
                            "ayt_kimya": {"dogru": 11, "yanlis": 2},
                            "ayt_biyoloji": {"dogru": 12, "yanlis": 1}
                        },
                        "hedef_siralama": 40000
                    },
                    {
                        "dogru_yanlis": {
                            "ayt_matematik": {"dogru": 28, "yanlis": 10},
                            "ayt_fizik": {"dogru": 8, "yanlis": 4},
                            "ayt_kimya": {"dogru": 9, "yanlis": 4},
                            "ayt_biyoloji": {"dogru": 10, "yanlis": 3}
                        },
                        "hedef_siralama": 150000
                    }
                ]
            },
            "ayt_ea": {
                "sinav_turu": "ayt_ea",
                "test_verileri": [
                    {
                        "dogru_yanlis": {
                            "ayt_matematik": {"dogru": 36, "yanlis": 4},
                            "ayt_edebiyat": {"dogru": 22, "yanlis": 2},
                            "ayt_tarih1": {"dogru": 9, "yanlis": 1},
                            "ayt_cografya1": {"dogru": 6, "yanlis": 0}
                        },
                        "hedef_siralama": 30000
                    },
                    {
                        "dogru_yanlis": {
                            "ayt_matematik": {"dogru": 28, "yanlis": 10},
                            "ayt_edebiyat": {"dogru": 18, "yanlis": 6},
                            "ayt_tarih1": {"dogru": 7, "yanlis": 3},
                            "ayt_cografya1": {"dogru": 4, "yanlis": 2}
                        },
                        "hedef_siralama": 120000
                    }
                ]
            },
            "ayt_sozel": {
                "sinav_turu": "ayt_sozel",
                "test_verileri": [
                    {
                        "dogru_yanlis": {
                            "ayt_edebiyat": {"dogru": 22, "yanlis": 2},
                            "ayt_tarih1": {"dogru": 9, "yanlis": 1},
                            "ayt_cografya1": {"dogru": 6, "yanlis": 0},
                            "ayt_tarih2": {"dogru": 10, "yanlis": 1},
                            "ayt_cografya2": {"dogru": 10, "yanlis": 1},
                            "ayt_felsefe": {"dogru": 11, "yanlis": 1},
                            "ayt_din_kulturu": {"dogru": 5, "yanlis": 1}
                        },
                        "hedef_siralama": 25000
                    },
                    {
                        "dogru_yanlis": {
                            "ayt_edebiyat": {"dogru": 18, "yanlis": 6},
                            "ayt_tarih1": {"dogru": 7, "yanlis": 3},
                            "ayt_cografya1": {"dogru": 4, "yanlis": 2},
                            "ayt_tarih2": {"dogru": 8, "yanlis": 3},
                            "ayt_cografya2": {"dogru": 9, "yanlis": 2},
                            "ayt_felsefe": {"dogru": 9, "yanlis": 3},
                            "ayt_din_kulturu": {"dogru": 5, "yanlis": 1}
                        },
                        "hedef_siralama": 100000
                    }
                ]
            },
            "ayt_dil": {
                "sinav_turu": "ayt_dil",
                "test_verileri": [
                    {
                        "dogru_yanlis": {
                            "ayt_dil": {"dogru": 75, "yanlis": 5}
                        },
                        "hedef_siralama": 10000
                    },
                    {
                        "dogru_yanlis": {
                            "ayt_dil": {"dogru": 60, "yanlis": 15}
                        },
                        "hedef_siralama": 40000
                    }
                ]
            }
        }

    def test_hatali_girisler(self):
        """Hatalı giriş senaryolarını test eder"""
        hatali_veriler = [
            {"sinav_turu": "gecersiz", "dogru_yanlis": {}, "hedef_siralama": 1000},
            {"sinav_turu": "tyt", "dogru_yanlis": {"tyt_turkce": {"dogru": -5, "yanlis": 10}}, "hedef_siralama": 1000},
            {"sinav_turu": "tyt", "dogru_yanlis": {"tyt_turkce": {"dogru": 50, "yanlis": 10}}, "hedef_siralama": 1000},  # 40 soru var
        ]
        
        for veri in hatali_veriler:
            with self.subTest(veri=veri):
                response = requests.post(
                    f"{self.base_url}/tahmin",
                    headers=self.headers,
                    json=veri
                )
                self.assertNotEqual(response.status_code, 200, f"Beklenmeyen başarı: {veri}")

    def test_model_performansi(self):
        """Model performansını değerlendirir"""
        for sinav_turu, veri in self.test_senaryolari.items():
            logger.info(f"\n{sinav_turu.upper()} Model Performans Değerlendirmesi")
            logger.info("-" * 50)
            
            tahminler = []
            hedefler = []
            
            for test_verisi in veri["test_verileri"]:
                try:
                    response = requests.post(
                        f"{self.base_url}/tahmin",
                        headers=self.headers,
                        json={
                            "sinav_turu": sinav_turu,
                            "dogru_yanlis": test_verisi["dogru_yanlis"],
                            "hedef_siralama": test_verisi["hedef_siralama"]
                        }
                    )
                    
                    if response.status_code == 200:
                        data = response.json()
                        tahmin = data["siralama"]["tahmini"]
                        hedef = test_verisi["hedef_siralama"]
                        
                        tahminler.append(tahmin)
                        hedefler.append(hedef)
                        
                        logger.info(f"\nTest Verisi:")
                        logger.info(f"Net: {data['net']['toplam']:.2f}")
                        logger.info(f"Tahmini Sıralama: {tahmin:,}")
                        logger.info(f"Hedef Sıralama: {hedef:,}")
                        logger.info(f"Fark: {abs(tahmin - hedef):,}")
                        logger.info(f"Değerlendirme: {data['siralama']['degerlendirme']}")
                        
                except Exception as e:
                    logger.error(f"Test sırasında hata: {str(e)}")
            
            if tahminler and hedefler:
                # Performans metriklerini hesapla
                mae = mean_absolute_error(hedefler, tahminler)
                rmse = np.sqrt(mean_squared_error(hedefler, tahminler))
                r2 = r2_score(hedefler, tahminler)
                
                # Ortalama hata yüzdesi
                hata_yuzdeleri = [abs(t-h)/h*100 for t,h in zip(tahminler, hedefler)]
                ortalama_hata_yuzdesi = np.mean(hata_yuzdeleri)
                
                logger.info(f"\nModel Performans Metrikleri:")
                logger.info(f"Ortalama Mutlak Hata (MAE): {mae:,.2f}")
                logger.info(f"Kök Ortalama Kare Hata (RMSE): {rmse:,.2f}")
                logger.info(f"R² Skoru: {r2:.4f}")
                logger.info(f"Ortalama Hata Yüzdesi: {ortalama_hata_yuzdesi:.2f}%")
                
                # Performans değerlendirmesi
                if ortalama_hata_yuzdesi < 10:
                    performans = "Mükemmel"
                elif ortalama_hata_yuzdesi < 20:
                    performans = "Çok İyi"
                elif ortalama_hata_yuzdesi < 30:
                    performans = "İyi"
                elif ortalama_hata_yuzdesi < 40:
                    performans = "Orta"
                else:
                    performans = "İyileştirilmeli"
                
                logger.info(f"\nGenel Performans Değerlendirmesi: {performans}")
                
                # Sonuçları CSV'ye kaydet
                sonuclar_df = pd.DataFrame({
                    'Sınav Türü': [sinav_turu] * len(tahminler),
                    'Tahmini Sıralama': tahminler,
                    'Hedef Sıralama': hedefler,
                    'Fark': [abs(t-h) for t,h in zip(tahminler, hedefler)],
                    'Hata Yüzdesi': hata_yuzdeleri
                })
                
                sonuclar_df.to_csv(f'test_sonuclari_{sinav_turu}.csv', index=False)
                logger.info(f"\nSonuçlar 'test_sonuclari_{sinav_turu}.csv' dosyasına kaydedildi.")

if __name__ == "__main__":
    unittest.main()