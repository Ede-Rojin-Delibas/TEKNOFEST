import unittest
import os
import sys

# Proje kök dizinini Python yoluna ekle
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.utils.tahmin import tahmin_yap, tyt_ensemble_predict

class TahminTest(unittest.TestCase):
    
    def test_tyt_tahmin(self):
        """TYT için geçerli bir tahmin senaryosunu test eder."""
        veri = {
            "tyt_turkce": {"dogru": 30, "yanlis": 5},
            "tyt_matematik": {"dogru": 25, "yanlis": 5},
            "tyt_sosyal": {"dogru": 15, "yanlis": 3},
            "tyt_fen": {"dogru": 10, "yanlis": 2}
        }
        sonuc = tahmin_yap("tyt", veri)
        self.assertIn("siralama", sonuc)
        self.assertIn("tahmini", sonuc["siralama"])
        self.assertIsInstance(sonuc["siralama"]["tahmini"], int)
        self.assertGreater(sonuc["siralama"]["tahmini"], 0)
        print(f"\nTYT Test Sonucu: {sonuc}")

    def test_ayt_sayisal_tahmin(self):
        """AYT Sayısal için geçerli bir tahmin senaryosunu test eder."""
        veri = {
            "ayt_matematik": {"dogru": 35, "yanlis": 3},
            "ayt_fizik": {"dogru": 10, "yanlis": 2},
            "ayt_kimya": {"dogru": 11, "yanlis": 1},
            "ayt_biyoloji": {"dogru": 9, "yanlis": 3}
        }
        sonuc = tahmin_yap("ayt_sayisal", veri)
        self.assertIn("siralama", sonuc)
        self.assertIn("tahmini", sonuc["siralama"])
        self.assertIsInstance(sonuc["siralama"]["tahmini"], int)
        self.assertGreater(sonuc["siralama"]["tahmini"], 0)
        print(f"AYT Sayısal Test Sonucu: {sonuc}")

    def test_ayt_ea_tahmin(self):
        """AYT Eşit Ağırlık için geçerli bir tahmin senaryosunu test eder."""
        veri = {
            "ayt_matematik": {"dogru": 28, "yanlis": 4},
            "ayt_edebiyat": {"dogru": 20, "yanlis": 2},
            "ayt_cografya1": {"dogru": 5, "yanlis": 1}
        }
        sonuc = tahmin_yap("ayt_ea", veri)
        self.assertIn("siralama", sonuc)
        self.assertIn("tahmini", sonuc["siralama"])
        self.assertIsInstance(sonuc["siralama"]["tahmini"], int)
        self.assertGreater(sonuc["siralama"]["tahmini"], 0)
        print(f"AYT EA Test Sonucu: {sonuc}")
        
    def test_ayt_sozel_tahmin(self):
        """AYT Sözel için geçerli bir tahmin senaryosunu test eder."""
        veri = {
            "ayt_edebiyat": {"dogru": 21, "yanlis": 3},
            "ayt_cografya1": {"dogru": 4, "yanlis": 2}
        }
        sonuc = tahmin_yap("ayt_sozel", veri)
        self.assertIn("siralama", sonuc)
        self.assertIn("tahmini", sonuc["siralama"])
        self.assertIsInstance(sonuc["siralama"]["tahmini"], int)
        self.assertGreater(sonuc["siralama"]["tahmini"], 0)
        print(f"AYT Sözel Test Sonucu: {sonuc}")

    def test_ayt_dil_tahmin(self):
        """AYT Dil için geçerli bir tahmin senaryosunu test eder."""
        veri = {
            "ayt_dil": {"dogru": 70, "yanlis": 5}
        }
        sonuc = tahmin_yap("ayt_dil", veri)
        self.assertIn("siralama", sonuc)
        self.assertIn("tahmini", sonuc["siralama"])
        self.assertIsInstance(sonuc["siralama"]["tahmini"], int)
        self.assertGreater(sonuc["siralama"]["tahmini"], 0)
        print(f"AYT Dil Test Sonucu: {sonuc}")

    def test_gecersiz_sinav_turu(self):
        """Geçersiz bir sınav türü gönderildiğinde hata fırlatılmasını test eder."""
        with self.assertRaises(ValueError):
            tahmin_yap("gecersiz_tur", {})
            
    def test_eksik_veri(self):
        """Doğru/yanlış anahtarları eksik veri gönderildiğinde hata fırlatılmasını test eder."""
        veri = {
            "tyt_turkce": {"dogru": 30} # "yanlis" eksik
        }
        with self.assertRaises(ValueError):
            tahmin_yap("tyt", veri)

if __name__ == '__main__':
    unittest.main()