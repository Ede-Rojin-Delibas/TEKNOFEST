import pandas as pd
import json
import logging
from pathlib import Path

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def veri_yukle(dosya_yolu: str) -> pd.DataFrame:
    """JSON veri dosyasını yükler ve DataFrame'e dönüştürür."""
    try:
        with open(dosya_yolu, 'r', encoding='utf-8') as f:
            veri = json.load(f)
        df = pd.DataFrame(veri)
        logger.info(f"Veri seti yüklendi. Toplam kayıt sayısı: {len(df)}")
        return df
    except Exception as e:
        logger.error(f"Veri yükleme hatası: {str(e)}")
        raise

def veri_analizi_yap(df: pd.DataFrame):
    """Veri setinin detaylı analizini yapar."""
    logger.info("\n=== DETAYLI VERİ ANALİZİ ===")
    
    # Tüm sınav türleri için gerekli dersler
    sinav_turleri = {
        "TYT": ["tyt_turkce", "tyt_matematik", "tyt_sosyal", "tyt_fen"],
        "AYT Sayısal": ["ayt_matematik", "ayt_kimya", "ayt_biyoloji", "ayt_fizik"],
        "AYT EA": ["ayt_matematik", "ayt_edebiyat", "ayt_cografya1"],
        "AYT Sözel": ["ayt_edebiyat", "ayt_cografya1"],
        "AYT Dil": ["ayt_dil"]
    }
    
    # Her sınav türü için veri dağılımı
    for sinav_turu, dersler in sinav_turleri.items():
        # Bu sınav türü için en az bir dersin verisi olan programlar
        en_az_bir_veri = df[dersler].notna().any(axis=1).sum()
        # Bu sınav türü için tüm derslerin verisi olan programlar
        tum_veriler = df[dersler].notna().all(axis=1).sum()
        
        logger.info(f"\n{sinav_turu} Veri Dağılımı:")
        logger.info(f"En az bir dersin verisi olan program sayısı: {en_az_bir_veri}")
        logger.info(f"Tüm derslerin verisi olan program sayısı: {tum_veriler}")
        logger.info(f"Eksik veri oranı: {((en_az_bir_veri - tum_veriler) / en_az_bir_veri * 100):.2f}%")
        
        # Her ders için veri sayısı
        for ders in dersler:
            veri_sayisi = df[ders].notna().sum()
            logger.info(f"{ders}: {veri_sayisi} kayıt ({veri_sayisi/len(df)*100:.2f}%)")

def veri_ayir_ve_kaydet(df: pd.DataFrame, hedef_klasor: str):
    """Veriyi sınav türlerine göre ayırır ve kaydeder."""
    Path(hedef_klasor).mkdir(parents=True, exist_ok=True)
    
    # 1. TYT Verileri
    tyt_veri = df[["basari_sirasi", "tyt_turkce", "tyt_matematik", "tyt_sosyal", "tyt_fen"]].dropna()
    tyt_veri["toplam_net"] = tyt_veri[["tyt_turkce", "tyt_matematik", "tyt_sosyal", "tyt_fen"]].sum(axis=1)
    tyt_veri.to_csv(f"{hedef_klasor}/tyt_veri_seti.csv", index=False)
    logger.info(f"TYT veri seti kaydedildi. Kayıt sayısı: {len(tyt_veri)}")
    
    # 2. AYT Sayısal Verileri
    ayt_sayisal = df[["basari_sirasi", "ayt_matematik", "ayt_kimya", "ayt_biyoloji", "ayt_fizik"]].dropna()
    ayt_sayisal["toplam_net"] = ayt_sayisal[["ayt_matematik", "ayt_kimya", "ayt_biyoloji", "ayt_fizik"]].sum(axis=1)
    ayt_sayisal.to_csv(f"{hedef_klasor}/ayt_sayisal_veri_seti.csv", index=False)
    logger.info(f"AYT Sayısal veri seti kaydedildi. Kayıt sayısı: {len(ayt_sayisal)}")
    
    # 3. AYT EA Verileri
    ayt_ea = df[["basari_sirasi", "ayt_matematik", "ayt_edebiyat", "ayt_cografya1"]].dropna()
    ayt_ea["toplam_net"] = ayt_ea[["ayt_matematik", "ayt_edebiyat", "ayt_cografya1"]].sum(axis=1)
    ayt_ea.to_csv(f"{hedef_klasor}/ayt_ea_veri_seti.csv", index=False)
    logger.info(f"AYT EA veri seti kaydedildi. Kayıt sayısı: {len(ayt_ea)}")
    
    # 4. AYT Sözel Verileri
    ayt_sozel = df[["basari_sirasi", "ayt_edebiyat", "ayt_cografya1"]].dropna()
    ayt_sozel["toplam_net"] = ayt_sozel[["ayt_edebiyat", "ayt_cografya1"]].sum(axis=1)
    ayt_sozel.to_csv(f"{hedef_klasor}/ayt_sozel_veri_seti.csv", index=False)
    logger.info(f"AYT Sözel veri seti kaydedildi. Kayıt sayısı: {len(ayt_sozel)}")
    
    # 5. AYT Dil Verileri
    ayt_dil = df[["basari_sirasi", "ayt_dil"]].dropna()
    ayt_dil["toplam_net"] = ayt_dil["ayt_dil"]
    ayt_dil.to_csv(f"{hedef_klasor}/ayt_dil_veri_seti.csv", index=False)
    logger.info(f"AYT Dil veri seti kaydedildi. Kayıt sayısı: {len(ayt_dil)}")
    
    # Veri seti istatistiklerini göster
    for sinav_turu, veri in [
        ("TYT", tyt_veri),
        ("AYT Sayısal", ayt_sayisal),
        ("AYT EA", ayt_ea),
        ("AYT Sözel", ayt_sozel),
        ("AYT Dil", ayt_dil)
    ]:
        logger.info(f"\n{sinav_turu} İstatistikleri:")
        logger.info(f"Ortalama toplam net: {veri['toplam_net'].mean():.2f}")
        logger.info(f"Ortalama sıralama: {veri['basari_sirasi'].mean():.2f}")
        logger.info(f"Minimum sıralama: {veri['basari_sirasi'].min():.0f}")
        logger.info(f"Maksimum sıralama: {veri['basari_sirasi'].max():.0f}")

def main():
    """Ana veri hazırlama işlemi."""
    try:
        # Yolları ayarla
        veri_dosyasi = "data/veri_setleri/tum_programlar_netler.json"
        hedef_klasor = "data/veri_setleri/processed_v3"
        
        # Veriyi yükle
        logger.info("Veri yükleniyor...")
        df = veri_yukle(veri_dosyasi)
        
        # Detaylı veri analizi yap
        veri_analizi_yap(df)
        
        # Veriyi ayır ve kaydet
        logger.info("\nVeri sınav türlerine göre ayrılıyor ve kaydediliyor...")
        veri_ayir_ve_kaydet(df, hedef_klasor)
        
        logger.info("\nVeri hazırlama işlemi tamamlandı.")
        
    except Exception as e:
        logger.error(f"Veri hazırlama hatası: {str(e)}")
        raise

if __name__ == "__main__":
    main() 