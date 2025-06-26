import pandas as pd
import numpy as np
import os

# Veri setinin bulunduğu dizin
data_dir = 'data/veri_setleri/processed_v3/'
input_filename = 'ayt_sozel_veri_seti_artirilmis_cleaned_cleaned_filtered.csv'
output_filename = 'ayt_sozel_veri_seti_doldurulmus.csv'

input_path = os.path.join(data_dir, input_filename)
output_path = os.path.join(data_dir, output_filename)

# Dosyayı oku
try:
    df = pd.read_csv(input_path)
    print(f"'{input_path}' dosyası başarıyla okundu.")
except FileNotFoundError:
    print(f"HATA: '{input_path}' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
    exit()

# Eklenecek dersler ve istatistikleri (ÖSYM 2023 verilerine yakın tahminler)
dersler_to_add = {
    'ayt_tarih1':    {'mean': 3.5, 'std': 2.5, 'max': 10},
    'ayt_tarih2':    {'mean': 2.5, 'std': 2.0, 'max': 11},
    'ayt_cografya2': {'mean': 2.8, 'std': 1.8, 'max': 11}, # Coğrafya-2 soru sayısı 11
    'ayt_felsefe':   {'mean': 2.0, 'std': 2.0, 'max': 12},
    'ayt_din':       {'mean': 2.5, 'std': 1.5, 'max': 6},
}

# Eksik sütunları ekle ve doldur
for ders, stats in dersler_to_add.items():
    if ders not in df.columns:
        print(f"'{ders}' sütunu ekleniyor...")
        # Normal dağılımdan rastgele veri üret
        random_data = np.random.normal(stats['mean'], stats['std'], len(df))
        # Değerleri [0, max_soru] aralığına kırp ve ondalıklı bırak (netler ondalıklı olabilir)
        df[ders] = np.round(random_data, 2).clip(0, stats['max'])

print("Eksik sütunlar eklendi.")

# Toplam neti yeniden hesapla
# AYT SÖZEL puanını oluşturan tüm derslerin netlerini topla
net_sutunlari = [
    'ayt_edebiyat', 'ayt_cografya1', 'ayt_tarih1', 'ayt_tarih2',
    'ayt_cografya2', 'ayt_felsefe', 'ayt_din'
]

# Veri setinde mevcut olan sütunları filtrele
mevcut_net_sutunlari = [sutun for sutun in net_sutunlari if sutun in df.columns]

df['toplam_net'] = df[mevcut_net_sutunlari].sum(axis=1)
print("'toplam_net' sütunu güncellendi.")

# Sonucu yeni bir CSV dosyasına kaydet
df.to_csv(output_path, index=False)

print(f"\nİşlem tamamlandı! Doldurulmuş veri seti şu yola kaydedildi: '{output_path}'")
print("\nYeni veri setinin ilk 5 satırı:")
print(df.head()) 