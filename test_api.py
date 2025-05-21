import unittest
import requests
import json

class SinavTahminTestCase(unittest.TestCase):
    def setUp(self):
        """Test öncesi hazırlıklar"""
        self.base_url = "http://localhost:5000/api"
        self.headers = {"Content-Type": "application/json"}
        
        # Test verileri
        self.test_senaryolari = {
            "tyt": {
                "normal": {
                    "sinav_turu": "tyt",
                    "dogru_yanlis": {
                        "tyt_turkce": {"dogru": 35, "yanlis": 5},
                        "tyt_matematik": {"dogru": 30, "yanlis": 10},
                        "tyt_sosyal": {"dogru": 15, "yanlis": 5},
                        "tyt_fen": {"dogru": 15, "yanlis": 5}
                    },
                    "hedef_siralama": 10000
                },
                "hacettepe_psikoloji": {
                    "sinav_turu": "tyt",
                    "dogru_yanlis": {
                        "tyt_turkce": {"dogru": 38, "yanlis": 2},
                        "tyt_matematik": {"dogru": 35, "yanlis": 5},
                        "tyt_sosyal": {"dogru": 17, "yanlis": 3},
                        "tyt_fen": {"dogru": 17, "yanlis": 3}
                    },
                    "hedef_siralama": 5000
                }
            },
            "ayt_say": {
                "normal": {
                    "sinav_turu": "ayt_say",
                    "dogru_yanlis": {
                        "ayt_matematik": {"dogru": 35, "yanlis": 5},
                        "ayt_fizik": {"dogru": 15, "yanlis": 5},
                        "ayt_kimya": {"dogru": 15, "yanlis": 5},
                        "ayt_biyoloji": {"dogru": 15, "yanlis": 5}
                    },
                    "hedef_siralama": 5000
                }
            },
            "ayt_soz": {
                "normal": {
                    "sinav_turu": "ayt_soz",
                    "dogru_yanlis": {
                        "ayt_turkce": {"dogru": 35, "yanlis": 5},
                        "ayt_tarih1": {"dogru": 15, "yanlis": 5},
                        "ayt_cografya1": {"dogru": 15, "yanlis": 5},
                        "ayt_tarih2": {"dogru": 15, "yanlis": 5},
                        "ayt_cografya2": {"dogru": 15, "yanlis": 5},
                        "ayt_felsefe": {"dogru": 15, "yanlis": 5},
                        "ayt_din": {"dogru": 15, "yanlis": 5}
                    },
                    "hedef_siralama": 5000
                }
            },
            "ayt_ea": {
                "normal": {
                    "sinav_turu": "ayt_ea",
                    "dogru_yanlis": {
                        "ayt_matematik": {"dogru": 35, "yanlis": 5},
                        "ayt_edebiyat": {"dogru": 35, "yanlis": 5},
                        "ayt_cografya1": {"dogru": 15, "yanlis": 5}
                    },
                    "hedef_siralama": 5000
                },
                "hacettepe_psikoloji": {
                    "sinav_turu": "ayt_ea",
                    "dogru_yanlis": {
                        "ayt_matematik": {"dogru": 38, "yanlis": 2},
                        "ayt_edebiyat": {"dogru": 38, "yanlis": 2},
                        "ayt_cografya1": {"dogru": 17, "yanlis": 3}
                    },
                    "hedef_siralama": 5000
                }
            },
            "ayt_dil": {
                "normal": {
                    "sinav_turu": "ayt_dil",
                    "dogru_yanlis": {
                        "ayt_dil": {"dogru": 75, "yanlis": 5}
                    },
                    "hedef_siralama": 5000
                },
                "hacettepe_dil": {
                    "sinav_turu": "ayt_dil",
                    "dogru_yanlis": {
                        "ayt_dil": {"dogru": 78, "yanlis": 2}
                    },
                    "hedef_siralama": 2000
                }
            }
        }

    def test_basarili_tahminler(self):
        """Tüm sınav türleri için başarılı tahmin testleri"""
        for sinav_turu, senaryolar in self.test_senaryolari.items():
            for senaryo_adi, veri in senaryolar.items():
                with self.subTest(f"{sinav_turu} - {senaryo_adi}"):
                    response = requests.post(
                        f"{self.base_url}/tahmin",
                        headers=self.headers,
                        json=veri
                    )
                    
                    # HTTP durumu kontrolü
                    self.assertEqual(response.status_code, 200)
                    
                    # Yanıt yapısı kontrolü
                    data = response.json()
                    self.assertIn("sinav_turu", data)
                    self.assertIn("net", data)
                    self.assertIn("siralama", data)
                    
                    # Değer kontrolleri
                    self.assertEqual(data["sinav_turu"], veri["sinav_turu"])
                    self.assertGreater(data["net"]["toplam"], 0)
                    self.assertLess(data["siralama"]["tahmini"], 2_000_000)
                    
                    print(f"\n{sinav_turu.upper()} - {senaryo_adi}")
                    print(f"Net: {data['net']['toplam']}")
                    print(f"Tahmini Sıralama: {data['siralama']['tahmini']:,}")
                    print(f"Hedef ile Fark: {data['siralama']['fark']:,}")
                    print(f"Değerlendirme: {data['siralama']['degerlendirme']}")

if __name__ == '__main__':
    unittest.main()