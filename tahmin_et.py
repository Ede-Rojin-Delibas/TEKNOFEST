import joblib
import os
import numpy as np
from typing import Union

def tahmin_yap(sinav_turu: str, net: Union[float, int]) -> int:
    """
    Verilen sınav türü ve net değerine göre sıralama tahmini yapar.
    
    Args:
        sinav_turu (str): Sınav türü ('tyt', 'ayt_ea', 'ayt_say', 'ayt_soz')
        net (float|int): Toplam net değeri
        
    Returns:
        int: Tahmini sıralama
        
    Raises:
        ValueError: Geçersiz sınav türü veya net değeri için
        FileNotFoundError: Model dosyası bulunamadığında
    """
    # Sınav türü kontrolü
    sinav_turu = sinav_turu.lower()
    GECERLI_SINAV_TURLERI = ["tyt", "ayt_ea", "ayt_say", "ayt_soz"]
    if sinav_turu not in GECERLI_SINAV_TURLERI:
        raise ValueError(f"Geçersiz sınav türü. Geçerli türler: {', '.join(GECERLI_SINAV_TURLERI)}")
    
    # Net değeri kontrolü
    try:
        net = float(net)
        if net < 0 or net > 120:  # Maksimum net kontrolü
            raise ValueError("Net değeri 0 ile 120 arasında olmalıdır")
    except (TypeError, ValueError):
        raise ValueError("Geçersiz net değeri")
    
    # Model dosyası kontrolü
    model_path = os.path.join("modeller", f"{sinav_turu}_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"{sinav_turu} için model dosyası bulunamadı: {model_path}")
    
    try:
        # Model yükleme
        model = joblib.load(model_path)
        
        # Tahmin
        net_array = np.array([[net]])  # 2D array'e çevir
        tahmin = model.predict(net_array)
        
        # Sonucu tamsayıya çevir ve sınırla
        siralama = int(round(tahmin[0]))
        return max(1, min(siralama, 2_000_000))  # Sıralama sınırları
        
    except Exception as e:
        raise RuntimeError(f"Tahmin sırasında hata oluştu: {str(e)}")

# Test kodu
if __name__ == "__main__":
    test_cases = [
        ("tyt", 80.5),
        ("ayt_say", 65.25),
        ("ayt_ea", 72.75),
        ("ayt_soz", 58.5)
    ]
    
    for sinav, net in test_cases:
        try:
            siralama = tahmin_yap(sinav, net)
            print(f"{sinav.upper()} - Net: {net} -> Tahmini Sıralama: {siralama:,}")
        except Exception as e:
            print(f"HATA ({sinav}): {str(e)}")
