import unittest
import requests
import json

class SinavTahminTestCase(unittest.TestCase):
    def setUp(self):
        """Test öncesi hazırlıklar"""
        self.base_url = "http://localhost:5000"
        self.headers = {"Content-Type": "application/json"}
        
        # Test verileri
        self.test_senaryolari = {
            "tyt": {
                "normal": {
                    "sinav_turu": "tyt",
                    "dogru_yanlis": {"dogru": 40, "yanlis": 10},
                    "hedef_siralama": 10000
                },
                "yuksek_net": {
                    "sinav_turu": "tyt",
                    "dogru_yanlis": {"dogru": 110, "yanlis": 5},
                    "hedef_siralama": 1000
                },
                "dusuk_net": {
                    "sinav_turu": "tyt",
                    "dogru_yanlis": {"dogru": 20, "yanlis": 30},
                    "hedef_siralama": 500000
                }
            },
            "ayt_say": {
                "normal": {
                    "sinav_turu": "ayt_say",
                    "dogru_yanlis": {"dogru": 35, "yanlis": 15},
                    "hedef_siralama": 20000
                },
                "yuksek_net": {
                    "sinav_turu": "ayt_say",
                    "dogru_yanlis": {"dogru": 75, "yanlis": 5},
                    "hedef_siralama": 5000
                }
            },
            "ayt_ea": {
                "normal": {
                    "sinav_turu": "ayt_ea",
                    "dogru_yanlis": {"dogru": 45, "yanlis": 15},
                    "hedef_siralama": 15000
                }
            },
            "ayt_soz": {
                "normal": {
                    "sinav_turu": "ayt_soz",
                    "dogru_yanlis": {"dogru": 50, "yanlis": 10},
                    "hedef_siralama": 25000
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
                    self.assertGreater(data["net"], 0)
                    self.assertLess(data["siralama"]["tahmini"], 2_000_000)
                    
                    print(f"\n{sinav_turu.upper()} - {senaryo_adi}")
                    print(f"Net: {data['net']}")
                    print(f"Tahmini Sıralama: {data['siralama']['tahmini']:,}")
                    print(f"Hedef ile Fark: {data['siralama']['fark']:,}")

    def test_hatali_durumlar(self):
        """Hata durumları testleri"""
        hatali_senaryolar = [
            {
                "aciklama": "Geçersiz sınav türü",
                "veri": {
                    "sinav_turu": "invalid",
                    "dogru_yanlis": {"dogru": 40, "yanlis": 10},
                    "hedef_siralama": 10000
                }
            },
            {
                "aciklama": "Negatif doğru sayısı",
                "veri": {
                    "sinav_turu": "tyt",
                    "dogru_yanlis": {"dogru": -5, "yanlis": 10},
                    "hedef_siralama": 10000
                }
            },
            {
                "aciklama": "Çok yüksek yanlış sayısı",
                "veri": {
                    "sinav_turu": "tyt",
                    "dogru_yanlis": {"dogru": 40, "yanlis": 200},
                    "hedef_siralama": 10000
                }
            },
            {
                "aciklama": "Geçersiz hedef sıralama",
                "veri": {
                    "sinav_turu": "tyt",
                    "dogru_yanlis": {"dogru": 40, "yanlis": 10},
                    "hedef_siralama": -1000
                }
            }
        ]
        
        for senaryo in hatali_senaryolar:
            with self.subTest(senaryo["aciklama"]):
                response = requests.post(
                    f"{self.base_url}/tahmin",
                    headers=self.headers,
                    json=senaryo["veri"]
                )
                
                self.assertGreaterEqual(response.status_code, 400)
                data = response.json()
                self.assertIn("error", data)
                
                print(f"\nHata Testi: {senaryo['aciklama']}")
                print(f"Durum Kodu: {response.status_code}")
                print(f"Hata Mesajı: {data['error']}")

def run_tests():
    """Testleri çalıştır ve sonuçları raporla"""
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(SinavTahminTestCase)
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

if __name__ == '__main__':
    print("Sınav Tahmin Sistemi Test Başlıyor...\n")
    run_tests()
