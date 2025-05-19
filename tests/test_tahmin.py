import unittest
from app.utils.tahmin import tahmin_yap

class TahminTestCase(unittest.TestCase):
    def test_gecerli_tahmin(self):
        sonuc = tahmin_yap("tyt", 80.5)
        self.assertGreater(sonuc, 0)
        self.assertLess(sonuc, 2_000_000)

    def test_gecersiz_sinav_turu(self):
        with self.assertRaises(ValueError):
            tahmin_yap("invalid", 80.5)

    def test_gecersiz_net_degeri(self):
        with self.assertRaises(ValueError):
            tahmin_yap("tyt", -10)

if __name__ == "__main__":
    unittest.main()