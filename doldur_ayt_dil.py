import pandas as pd
import numpy as np
import os

# Veri setinin bulunduğu dizin ve dosya adları
data_dir = 'data/veri_setleri/processed_v3/'
input_filename = 'ayt_dil_veri_seti_artirilmis_cleaned_cleaned_filtered.csv'
output_filename = 'ayt_dil_veri_seti_doldurulmus.csv'

input_path = os.path.join(data_dir, input_filename)
output_path = os.path.join(data_dir, output_filename)

# Dosyayı oku
try:
    df = pd.read_csv(input_path)
    print(f"'{input_path}' dosyası başarıyla okundu. Toplam {len(df)} satır.")
except FileNotFoundError:
    print(f"HATA: '{input_path}' dosyası bulunamadı. Lütfen dosya yolunu kontrol edin.")
    exit()

# 'ayt_dil' sütunundaki 0 veya NaN değerleri bul
target_column = 'ayt_dil'
problematic_rows = df[target_column].isna() | (df[target_column] == 0)
num_problematic = problematic_rows.sum()

if num_problematic > 0:
    print(f"'{target_column}' sütununda {num_problematic} adet 0 veya boş değerli satır bulundu.")
    
    # Bu değerleri 1 ile 5 arasında rastgele ondalıklı sayılarla doldur
    replacement_values = np.random.uniform(1, 5, num_problematic)
    df.loc[problematic_rows, target_column] = np.round(replacement_values, 2)
    
    print("Bu satırlar 1-5 arası rastgele net değerleri ile güncellendi.")
else:
    print(f"'{target_column}' sütununda 0 veya boş değerli satır bulunamadı. Dosya değiştirilmedi.")

# Toplam neti yeniden hesapla (DİL için sadece ayt_dil neti var)
if 'toplam_net' in df.columns:
    df['toplam_net'] = df['ayt_dil']
    print("'toplam_net' sütunu güncellendi.")

# Sonucu yeni bir CSV dosyasına kaydet
df.to_csv(output_path, index=False)

print(f"\nİşlem tamamlandı! Yeni veri seti şu yola kaydedildi: '{output_path}'")
print("\nYeni veri setinin ilk 5 satırı:")
print(df.head()) 