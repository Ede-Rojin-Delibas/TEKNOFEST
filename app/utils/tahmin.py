import joblib
import os
import numpy as np
from typing import Union
from config import Config
import logging

logger = logging.getLogger(__name__)

def tahmin_yap(sinav_turu: str, net: float) -> int:
    """
    Verilen sınav türü ve net değerine göre sıralama tahmini yapar.
    
    Args:
        sinav_turu (str): Sınav türü ('tyt', 'ayt_ea', 'ayt_say', 'ayt_soz')
        net (float): Toplam net değeri
        
    Returns:
        int: Tahmini sıralama
        
    Raises:
        ValueError: Geçersiz sınav türü veya net değeri için
        FileNotFoundError: Model dosyası bulunamadığında
    """
    logger.info(f"Tahmin işlemi başlatıldı: Sınav Türü={sinav_turu}, Net={net}")
    # Sınav türü kontrolü
    sinav_turu = sinav_turu.lower()
    GECERLI_SINAV_TURLERI = Config.ALLOWED_EXAM_TYPES
    if sinav_turu not in GECERLI_SINAV_TURLERI:
        raise ValueError(f"Geçersiz sınav türü '{sinav_turu}'. Geçerli türler: {', '.join(GECERLI_SINAV_TURLERI)}")
        
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
        raise FileNotFoundError(f"{sinav_turu} için model dosyası bulunamadı. Lütfen '{model_path}' dizinini kontrol edin.")
    
    try:
        # Model yükleme
        model = joblib.load(model_path)
        
        # Tahmin
        net_array = np.array([[net]])  # 2D array'e çevir
        tahmin = model.predict(net_array)
        
        # Sonucu tamsayıya çevir ve sınırla
        siralama = int(round(tahmin[0]))
        if siralama < 1 or siralama > 2_000_000:
            raise ValueError(f"Tahmin edilen sıralama ({siralama}) geçerli bir aralıkta değil.")
        siralama = max(1, min(siralama, 2_000_000))  # Sıralama sınırları
        logger.info(f"Tahmin tamamlandı: Tahmini Sıralama={siralama}")
        return siralama
        
    except Exception as e:
        raise RuntimeError(f"Tahmin sırasında hata oluştu: {str(e)}")

