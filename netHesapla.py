def net_hesapla(sinav_turu, dogru_yanlis_verileri):
    """
    Sınav türüne göre toplam net hesaplama yapar.
    
    Args:
        sinav_turu (str): Sınav türü ('tyt', 'ayt_ea', 'ayt_say', 'ayt_soz')
        dogru_yanlis_verileri (dict): Toplam doğru ve yanlış sayıları
        
    Returns:
        float: Toplam net
    """
    YANLIS_KATSAYI = 0.25
    
    try:
        # Toplam doğru ve yanlış sayılarını al
        toplam_dogru = float(dogru_yanlis_verileri.get("dogru", 0))
        toplam_yanlis = float(dogru_yanlis_verileri.get("yanlis", 0))
        
        # Geçerlilik kontrolleri
        if toplam_dogru < 0 or toplam_yanlis < 0:
            raise ValueError("Negatif değer girilemez")
            
        # Net hesaplama
        toplam_net = toplam_dogru - (toplam_yanlis * YANLIS_KATSAYI)
        
        return round(toplam_net, 2)
        
    except (TypeError, ValueError) as e:
        raise ValueError(f"Hatalı veri: {str(e)}")
