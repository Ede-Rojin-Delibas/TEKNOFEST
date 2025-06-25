import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import RobustScaler

# Sınav türleri ve dosya yolları
SINAV_TURLERI = {
    "tyt": "data/veri_setleri/processed_v3/tyt_veri_seti_artirilmis_cleaned_filtered.csv",
    "ayt_sayisal": "data/veri_setleri/processed_v3/ayt_sayisal_veri_seti_artirilmis_cleaned_filtered.csv",
    "ayt_ea": "data/veri_setleri/processed_v3/ayt_ea_veri_seti_artirilmis_cleaned_cleaned.csv",
    "ayt_sozel": "data/veri_setleri/processed_v3/ayt_sozel_veri_seti_artirilmis_cleaned_cleaned.csv",
    "ayt_dil": "data/veri_setleri/processed_v3/ayt_dil_veri_seti_artirilmis_cleaned_cleaned.csv"
}

# Orta net aralıkları (her sınav türü için)
ORTA_NET_ARALIKLARI = {
    "tyt": (45, 65),
    "ayt_sayisal": (35, 55),
    "ayt_ea": (35, 55),
    "ayt_sozel": (35, 55),
    "ayt_dil": (35, 55)
}

def analiz_et(sinav_turu, dosya_yolu, orta_aralik):
    print(f"\n--- {sinav_turu.upper()} Analizi ---")
    if not os.path.exists(dosya_yolu):
        print(f"Dosya bulunamadı: {dosya_yolu}")
        return
    df = pd.read_csv(dosya_yolu)
    print(f"Toplam örnek sayısı: {len(df)}")
    # Hedef ve net sütunlarını bul
    hedef_kolon = [c for c in df.columns if 'basari' in c][0]
    net_kolon = [c for c in df.columns if 'toplam_net' in c][0]
    # Orta net aralığı
    orta_df = df[(df[net_kolon] >= orta_aralik[0]) & (df[net_kolon] <= orta_aralik[1])]
    print(f"Orta net aralığında ({orta_aralik[0]}-{orta_aralik[1]}) örnek sayısı: {len(orta_df)}")
    print("Orta aralıkta sıralama istatistikleri:")
    print(orta_df[hedef_kolon].describe())
    print("Aykırı sıralamalar (en düşük 10, en yüksek 10):")
    print(orta_df[hedef_kolon].sort_values().head(10).to_list())
    print(orta_df[hedef_kolon].sort_values(ascending=False).head(10).to_list())
    # Histogram
    plt.figure(figsize=(10,5))
    plt.hist(orta_df[hedef_kolon], bins=50)
    plt.title(f"Orta Net Aralığında Sıralama Dağılımı ({sinav_turu.upper()})")
    plt.xlabel("Başarı Sırası")
    plt.ylabel("Örnek Sayısı")
    plt.tight_layout()
    plt.savefig(f"analiz_{sinav_turu}_hist.png")
    plt.close()
    print(f"Histogram kaydedildi: analiz_{sinav_turu}_hist.png")

# Analiz edilecek dosyalar
DATASETS = {
    'tyt': 'data/veri_setleri/processed_v3/tyt_veri_seti_artirilmis_cleaned_filtered.csv',
    'ayt_sayisal': 'data/veri_setleri/processed_v3/ayt_sayisal_veri_seti_artirilmis_cleaned_filtered.csv',
    'ayt_ea': 'data/veri_setleri/processed_v3/ayt_ea_veri_seti_artirilmis_cleaned_cleaned.csv',
    'ayt_sozel': 'data/veri_setleri/processed_v3/ayt_sozel_veri_seti_artirilmis_cleaned_cleaned.csv',
    'ayt_dil': 'data/veri_setleri/processed_v3/ayt_dil_veri_seti_artirilmis_cleaned_cleaned.csv',
}

# Her veri seti için ilgili net sütunları
NET_COLUMNS = {
    'tyt': ['tyt_turkce', 'tyt_matematik', 'tyt_sosyal', 'tyt_fen'],
    'ayt_sayisal': ['ayt_matematik', 'ayt_kimya', 'ayt_biyoloji', 'ayt_fizik'],
    'ayt_ea': ['ayt_matematik', 'ayt_edebiyat', 'ayt_cografya1'],
    'ayt_sozel': ['ayt_edebiyat', 'ayt_cografya1'],
    'ayt_dil': ['ayt_dil'],
}

def analyze_dataset(name, df):
    print(f'\n===== {name.upper()} ANALİZİ =====')
    if df is None or df.empty:
        print('  Veri seti boş veya okunamadı.')
        return

    print(f'  Toplam örnek sayısı: {len(df)}')
    print(f'  Sütunlar: {df.columns.tolist()}')

    # Eksik değer kontrolü
    print('\n--- Eksik Değerler ---')
    print(df.isnull().sum())

    # Sıfır ve negatif değer kontrolü
    print('\n--- Sıfır/Negatif Değerler ---')
    net_cols_to_check = [col for col in NET_COLUMNS[name] + ['toplam_net'] if col in df.columns]
    for col in net_cols_to_check:
        zero_count = (df[col] == 0).sum()
        neg_count = (df[col] < 0).sum()
        print(f'  {col}: Sıfır={zero_count}, Negatif={neg_count}')

    # toplam_net ile diğer netlerin toplamı
    print('\n--- toplam_net ile Diğer Netlerin Toplamı Farkı ---')
    net_sum = sum([df[c] for c in NET_COLUMNS[name] if c in df.columns])
    diff = np.abs(df['toplam_net'] - net_sum)
    print(f'  Farkı 0.5 puandan fazla olan satır sayısı: {(diff > 0.5).sum()}')
    print(f'  En büyük farklar:')
    print(diff.sort_values(ascending=False).head())

    # Uç değer kontrolü
    print('\n--- Uç Değerler ---')
    if 'basari_sirasi' in df.columns:
        print('  toplam_net < 10:', (df['toplam_net'] < 10).sum())
        print('  toplam_net > 120:', (df['toplam_net'] > 120).sum())
        print('  basari_sirasi < 1:', (df['basari_sirasi'] < 1).sum())
        print('  basari_sirasi > 1_000_000:', (df['basari_sirasi'] > 1_000_000).sum())
    else:
        print('  toplam_net < 10:', (df['toplam_net'] < 10).sum())
        print('  toplam_net > 120:', (df['toplam_net'] > 120).sum())

    # Temel istatistikler
    print('\n--- Temel İstatistikler ---')
    print(df.describe())

def clean_and_filter_data(name, df):
    print(f'\n===== {name.upper()} VERİ TEMİZLEME VE FİLTRELEME =====')
    if df is None or df.empty:
        print('  Veri seti boş, işlem yapılamadı.')
        return None
    
    before = len(df)
    
    # Kullanıcının geri bildirimine göre filtreleme güncellendi.
    # Ana filtreleme kriteri: geçerli bir başarı sırası.
    # Başarı sırası 0'dan büyük olmalı ve üst limit 2 milyona çıkarıldı.
    if 'basari_sirasi' in df.columns:
        df = df[df['basari_sirasi'] > 0].copy()
        df = df[df['basari_sirasi'] <= 2_000_000].copy()

    # Negatif netleri olan ama geçerli sıralaması olanları tutmak için
    # toplam_net > 0 filtresi kaldırıldı.
    
    after = len(df)
    print(f'  Kalan satır: {after} / {before} (Atılan: {before-after})')
    
    if after == 0:
        print("  UYARI: Filtreleme sonrası hiç veri kalmadı!")
        return None
        
    return df

def train_and_evaluate_log_model(name, df):
    print(f"\n===== {name.upper()} NORMAL VE LOGARİTMİK MODEL KARŞILAŞTIRMASI =====")
    if df is None or df.empty:
        print('  Veri seti boş, model eğitilemedi.')
        return

    if 'basari_sirasi' not in df.columns:
        print('  basari_sirasi kolonu yok!')
        return

    # 'ayt_dil' için özel durum: sadece bir özellik var, ölçekleme ve karmaşık model gereksiz olabilir.
    if name == 'ayt_dil' and len(NET_COLUMNS[name]) == 1:
        X = df[NET_COLUMNS[name]].values.reshape(-1, 1)
    else:
        X = df[NET_COLUMNS[name]].values

    y = df['basari_sirasi'].values
    
    # Veri çok azsa RobustScaler hata verebilir
    if X.shape[0] < 2:
        print("  Eğitim için yetersiz veri.")
        return

    # Ölçekleme
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    # Eğitim/test bölme
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    
    if len(X_train) == 0 or len(X_test) == 0:
        print("  Eğitim veya test seti oluşturulamadı (yetersiz veri).")
        return

    # Model (GradientBoosting örnek)
    model = GradientBoostingRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42, subsample=0.8)

    # Normal hedefle eğitim
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\n--- Normal Model (Doğrudan Sıralama Tahmini) ---")
    print(f"MAE: {mean_absolute_error(y_test, y_pred):,.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred)):,.2f}")
    print(f"R²: {r2_score(y_test, y_pred):.4f}")

    # Log hedefle eğitim
    y_train_log = np.log1p(y_train)
    model.fit(X_train, y_train_log)
    y_pred_log = model.predict(X_test)
    y_pred_exp = np.expm1(y_pred_log)
    print("\n--- Logaritmik Model (Log(Sıralama) Tahmini) ---")
    print(f"MAE: {mean_absolute_error(y_test, y_pred_exp):,.2f}")
    print(f"RMSE: {np.sqrt(mean_squared_error(y_test, y_pred_exp)):,.2f}")
    print(f"R²: {r2_score(y_test, y_pred_exp):.4f}")

if __name__ == '__main__':
    for name, path in DATASETS.items():
        # 1. Orijinal veriyi oku
        if not os.path.exists(path):
            print(f'\n!!! Dosya bulunamadı: {path}')
            continue
        try:
            original_df = pd.read_csv(path)
        except Exception as e:
            print(f'\n!!! Okuma hatası: {path} - {e}')
            continue
            
        # 2. Temizlemeden ÖNCE analiz et
        print("\n" + "="*50)
        print(f"İŞLEM BAŞLIYOR: {name.upper()} (TEMİZLEME ÖNCESİ)")
        print("="*50)
        analyze_dataset(name, original_df.copy())

        # 3. Veriyi temizle ve filtrele
        cleaned_df = clean_and_filter_data(name, original_df)
        
        if cleaned_df is None:
            print(f"!!! {name.upper()} için temizleme sonrası veri kalmadı, sonraki adıma geçiliyor.")
            continue

        # 4. Temizlenmiş veriyi kaydet
        out_path = path.replace('.csv', '_filtered.csv')
        cleaned_df.to_csv(out_path, index=False)
        print(f'  Temizlenmiş ve filtrelenmiş dosya kaydedildi: {out_path}')
        
        # 5. Temizlemeden SONRA analiz et (Sadece veri değiştiyse)
        if original_df.shape != cleaned_df.shape:
            print("\n" + "="*50)
            print(f"İŞLEM TAMAMLANDI: {name.upper()} (TEMİZLEME SONRASI)")
            print("="*50)
            analyze_dataset(name, cleaned_df.copy())
        else:
            print(f"  Veri setinde değişiklik olmadı, tekrar analiz edilmiyor.")

        # 6. Temiz veri üzerinde model karşılaştırması yap
        train_and_evaluate_log_model(name, cleaned_df)

    print("\n\nAnaliz ve filtreleme tamamlandı.") 