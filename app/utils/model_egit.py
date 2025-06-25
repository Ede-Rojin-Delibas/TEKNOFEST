import json
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import logging
from typing import Dict, List, Tuple
from config import Config

# Logging ayarları
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sınav türlerine göre ders tanımlamaları
SINAV_DERSLERI = {
    "tyt": {
        "dersler": ["tyt_turkce", "tyt_matematik", "tyt_sosyal", "tyt_fen"],
        "toplam_soru": 120
    },
    "ayt_say": {
        "dersler": ["ayt_matematik", "ayt_fizik", "ayt_kimya", "ayt_biyoloji"],
        "toplam_soru": 80
    },
    "ayt_ea": {
        "dersler": ["ayt_matematik", "ayt_edebiyat", "ayt_cografya1"],
        "toplam_soru": 80
    },
    "ayt_soz": {
        "dersler": ["ayt_edebiyat", "ayt_tarih1", "ayt_cografya1", "ayt_tarih2", "ayt_cografya2", "ayt_felsefe", "ayt_din_kulturu"],
        "toplam_soru": 80
    },
    "ayt_dil": {
        "dersler": ["ayt_dil"],
        "toplam_soru": 80
    }
}

def json_veri_isle(json_path: str, output_dir: str = "data/processed") -> Dict[str, str]:
    """
    JSON veri setini sınav türlerine göre işler ve CSV dosyalarına kaydeder.
    Her sınav türü için hem toplam net hem de ders bazlı verileri kaydeder.
    
    Args:
        json_path (str): JSON dosyasının yolu
        output_dir (str): CSV dosyalarının kaydedileceği dizin
        
    Returns:
        Dict[str, str]: Sınav türü -> CSV dosya yolu eşleştirmesi
    """
    logger.info(f"JSON dosyası okunuyor: {json_path}")
    
    # Çıktı dizinini oluştur
    os.makedirs(output_dir, exist_ok=True)
    
    # JSON dosyasını oku
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Her sınav türü için ayrı DataFrame oluştur
    csv_dosyalari = {}
    for sinav_turu in Config.ALLOWED_EXAM_TYPES:
        if sinav_turu not in SINAV_DERSLERI:
            logger.warning(f"{sinav_turu} için ders tanımı bulunamadı, atlanıyor...")
            continue
            
        # Sınav türüne göre verileri filtrele
        sinav_verileri = []
        for kayit in data:
            if sinav_turu in kayit.get('sinav_turu', '').lower():
                # Temel verileri al
                kayit_verisi = {
                    'siralama': kayit.get('siralama', 0),
                    'toplam_net': kayit.get('toplam_net', 0)
                }
                
                # Ders bazlı verileri ekle
                for ders in SINAV_DERSLERI[sinav_turu]['dersler']:
                    ders_net = kayit.get(ders, {}).get('net', 0)
                    ders_dogru = kayit.get(ders, {}).get('dogru', 0)
                    ders_yanlis = kayit.get(ders, {}).get('yanlis', 0)
                    kayit_verisi.update({
                        f"{ders}_net": ders_net,
                        f"{ders}_dogru": ders_dogru,
                        f"{ders}_yanlis": ders_yanlis
                    })
                
                # Geçerli kayıtları al
                if kayit_verisi['toplam_net'] > 0 and kayit_verisi['siralama'] > 0:
                    sinav_verileri.append(kayit_verisi)
        
        if sinav_verileri:
            # DataFrame oluştur
            df = pd.DataFrame(sinav_verileri)
            
            # Veri doğrulama
            df = veri_dogrulama(df, sinav_turu)
            
            # CSV'ye kaydet
            csv_path = os.path.join(output_dir, f"{sinav_turu}_data.csv")
            df.to_csv(csv_path, index=False)
            csv_dosyalari[sinav_turu] = csv_path
            
            # İstatistikleri raporla
            rapor_istatistikler(df, sinav_turu)
            
            logger.info(f"{sinav_turu} için {len(df)} kayıt işlendi ve {csv_path} kaydedildi")
    
    return csv_dosyalari

def veri_dogrulama(df: pd.DataFrame, sinav_turu: str) -> pd.DataFrame:
    """
    Veri setini doğrular ve temizler.
    
    Args:
        df (pd.DataFrame): İşlenecek veri seti
        sinav_turu (str): Sınav türü
        
    Returns:
        pd.DataFrame: Temizlenmiş veri seti
    """
    # Veri seti boyut kontrolü
    if len(df) < 10:
        logger.warning(f"{sinav_turu} için veri seti çok küçük ({len(df)} örnek)")
    
    # Sayısal sütunları dönüştür
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Eksik değerleri temizle
    df = df.dropna()
    
    # Değer aralığı kontrolü
    if df['toplam_net'].max() > SINAV_DERSLERI[sinav_turu]['toplam_soru']:
        logger.warning(f"{sinav_turu}: Bazı kayıtlarda toplam net değeri {SINAV_DERSLERI[sinav_turu]['toplam_soru']}'den büyük")
        df = df[df['toplam_net'] <= SINAV_DERSLERI[sinav_turu]['toplam_soru']]
    
    if df['siralama'].min() < 1:
        logger.warning(f"{sinav_turu}: Bazı kayıtlarda sıralama 1'den küçük")
        df = df[df['siralama'] >= 1]
    
    # Aykırı değerleri temizle (IQR yöntemi)
    for col in ['toplam_net', 'siralama']:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        alt_sinir = Q1 - 1.5 * IQR
        ust_sinir = Q3 + 1.5 * IQR
        aykiri_mask = (df[col] < alt_sinir) | (df[col] > ust_sinir)
        if aykiri_mask.any():
            logger.warning(f"{sinav_turu}: {col} için {aykiri_mask.sum()} aykırı değer tespit edildi")
            df = df[~aykiri_mask]
    
    return df

def rapor_istatistikler(df: pd.DataFrame, sinav_turu: str):
    """
    Veri seti istatistiklerini raporlar.
    
    Args:
        df (pd.DataFrame): Veri seti
        sinav_turu (str): Sınav türü
    """
    logger.info(f"\n{sinav_turu.upper()} Veri Seti İstatistikleri:")
    logger.info(f"Toplam kayıt sayısı: {len(df)}")
    
    # Genel istatistikler
    logger.info("\nGenel İstatistikler:")
    logger.info(f"Toplam Net - Ortalama: {df['toplam_net'].mean():.2f}, Std: {df['toplam_net'].std():.2f}")
    logger.info(f"Sıralama - Ortalama: {df['siralama'].mean():.2f}, Std: {df['siralama'].std():.2f}")
    
    # Ders bazlı istatistikler
    logger.info("\nDers Bazlı İstatistikler:")
    for ders in SINAV_DERSLERI[sinav_turu]['dersler']:
        net_col = f"{ders}_net"
        if net_col in df.columns:
            logger.info(f"\n{ders}:")
            logger.info(f"Net - Ortalama: {df[net_col].mean():.2f}, Std: {df[net_col].std():.2f}")
            logger.info(f"En yüksek net: {df[net_col].max():.2f}")
            logger.info(f"En düşük net: {df[net_col].min():.2f}")

def model_egit(csv_path: str, sinav_turu: str, model_dir: str = "modeller") -> Tuple[RandomForestRegressor, float]:
    """
    Verilen CSV dosyasından model eğitir ve kaydeder.
    
    Args:
        csv_path (str): CSV dosyasının yolu
        sinav_turu (str): Sınav türü
        model_dir (str): Model dosyalarının kaydedileceği dizin
        
    Returns:
        Tuple[RandomForestRegressor, float]: Eğitilmiş model ve R2 skoru
    """
    logger.info(f"{sinav_turu} için model eğitimi başlatılıyor")
    
    # CSV'yi oku
    df = pd.read_csv(csv_path)
    
    # Veriyi hazırla
    X = df[['toplam_net']].values
    y = df['siralama'].values
    
    # Eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model eğit
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Test seti üzerinde değerlendir
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    logger.info(f"{sinav_turu} model performansı - R2: {r2:.4f}, RMSE: {rmse:.2f}")
    
    # Modeli kaydet
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{sinav_turu}_model.pkl")
    joblib.dump(model, model_path)
    logger.info(f"Model kaydedildi: {model_path}")
    
    return model, r2

def main():
    """Ana işlem fonksiyonu"""
    # Yolları ayarla
    json_path = "data/veri_setleri/tum_programlar_netler.json"
    output_dir = "data/processed"
    model_dir = "modeller"
    
    try:
        # JSON'ı işle ve CSV'lere kaydet
        csv_files = json_veri_isle(json_path, output_dir)
        
        # Her sınav türü için model eğit
        results = {}
        for sinav_turu, csv_path in csv_files.items():
            model, r2 = model_egit(csv_path, sinav_turu, model_dir)
            results[sinav_turu] = {
                'r2_score': r2,
                'model_path': os.path.join(model_dir, f"{sinav_turu}_model.pkl")
            }
        
        # Sonuçları raporla
        logger.info("\nModel Eğitim Sonuçları:")
        for sinav_turu, result in results.items():
            logger.info(f"{sinav_turu}: R2 = {result['r2_score']:.4f}")
        
    except Exception as e:
        logger.error(f"Hata oluştu: {str(e)}")
        raise

if __name__ == "__main__":
    main() 