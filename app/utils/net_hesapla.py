def net_hesapla(sinav_turu:str, dogru_yanlis_verileri:dict) -> float:
    """
    Sınav türüne göre toplam net hesaplama yapar.
    
    Args:
        sinav_turu (str): Sınav türü ('tyt', 'ayt_ea', 'ayt_say', 'ayt_soz')
        dogru_yanlis_verileri (dict): Toplam doğru ve yanlış sayıları

    Returns:
        float:Toplam net  
    Raises:
        ValueError:Geçersiz veri durumunda
    """
    YANLIS_KATSAYI = 0.25
    
    try:
        # Toplam doğru ve yanlış sayılarını al
        toplam_dogru = float(dogru_yanlis_verileri.get("dogru", 0))
        toplam_yanlis = float(dogru_yanlis_verileri.get("yanlis", 0))
        
        # Geçerlilik kontrolleri
        if toplam_dogru < 0 or toplam_yanlis < 0:
            raise ValueError("Negatif değer girilemez")
        
        if toplam_dogru > 120 or toplam_yanlis > 120:
            raise ValueError("Doğru/yanlış sayısı 120'den büyük olamaz")
            
        # Net hesaplama
        toplam_net = toplam_dogru - (toplam_yanlis * YANLIS_KATSAYI)
        
        return round(toplam_net, 2)
        
    except (TypeError, ValueError) as e:
        raise ValueError(f"Hatalı veri: {str(e)}")

def toplam_net_hesapla(sinav_turu, netler):
    """
    Ders bazında netleri toplayarak toplam neti döndürür.
    Args:
        sinav_turu (str): Sınav türü ('tyt', 'ayt_ea', 'ayt_say', 'ayt_soz')
        netler (dict): {'ayt_matematik': 30, ...}
    Returns:
        float: Toplam net
    """
    if sinav_turu == "tyt":
        dersler = ["tyt_turkce", "tyt_matematik", "tyt_sosyal", "tyt_fen"]
    elif sinav_turu == "ayt_say":
        dersler = ["ayt_matematik", "ayt_kimya", "ayt_biyoloji", "ayt_fizik"]
    elif sinav_turu == "ayt_ea":
        dersler = ["ayt_matematik", "ayt_turkce", "ayt_tarih1", "ayt_cografya1"]
    elif sinav_turu == "ayt_soz":
        dersler = ["ayt_turkce", "ayt_tarih1", "ayt_cografya1", "ayt_tarih2", "ayt_cografya2", "ayt_felsefe", "ayt_din"]
    else:
        raise ValueError("Geçersiz sınav türü")
    return sum([float(netler.get(ders, 0)) for ders in dersler])
