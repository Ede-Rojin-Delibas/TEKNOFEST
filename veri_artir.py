import pandas as pd
import numpy as np
import os

SINAV_TURLERI = {
    "tyt": "data/veri_setleri/processed_v3/tyt_veri_seti.csv",
    "ayt_sayisal": "data/veri_setleri/processed_v3/ayt_sayisal_veri_seti.csv",
    "ayt_ea": "data/veri_setleri/processed_v3/ayt_ea_veri_seti.csv",
    "ayt_sozel": "data/veri_setleri/processed_v3/ayt_sozel_veri_seti.csv",
    "ayt_dil": "data/veri_setleri/processed_v3/ayt_dil_veri_seti.csv"
}

ORTA_NET_ARALIKLARI = {
    "tyt": (45, 65),
    "ayt_sayisal": (35, 55),
    "ayt_ea": (35, 55),
    "ayt_sozel": (35, 55),
    "ayt_dil": (35, 55)
}

ORTA_SIRALAMA_ARALIGI = {
    "tyt": (200_000, 600_000),
    "ayt_sayisal": (50_000, 200_000),
    "ayt_ea": (40_000, 150_000),
    "ayt_sozel": (30_000, 120_000),
    "ayt_dil": (10_000, 50_000)
}

def veri_artir(df, net_kolon, siralama_kolon, net_aralik, siralama_aralik, n_artir=2):
    orta_df = df[
        (df[net_kolon] >= net_aralik[0]) & (df[net_kolon] <= net_aralik[1]) &
        (df[siralama_kolon] >= siralama_aralik[0]) & (df[siralama_kolon] <= siralama_aralik[1])
    ]
    print(f"Artırılacak örnek sayısı: {len(orta_df)} (her biri {n_artir} kez eklenecek)")
    if len(orta_df) == 0:
        return df
    artirilmis = orta_df.sample(n=len(orta_df)*n_artir, replace=True, random_state=42)
    yeni_df = pd.concat([df, artirilmis], ignore_index=True)
    return yeni_df

if __name__ == "__main__":
    for sinav_turu, dosya_yolu in SINAV_TURLERI.items():
        print(f"\n--- {sinav_turu.upper()} Veri Artırımı ---")
        if not os.path.exists(dosya_yolu):
            print(f"Dosya bulunamadı: {dosya_yolu}")
            continue
        df = pd.read_csv(dosya_yolu)
        hedef_kolon = [c for c in df.columns if 'basari' in c][0]
        net_kolon = [c for c in df.columns if 'toplam_net' in c][0]
        yeni_df = veri_artir(
            df, net_kolon, hedef_kolon,
            ORTA_NET_ARALIKLARI[sinav_turu],
            ORTA_SIRALAMA_ARALIGI[sinav_turu],
            n_artir=2
        )
        yeni_dosya = dosya_yolu.replace('.csv', '_artirilmis.csv')
        yeni_df.to_csv(yeni_dosya, index=False)
        print(f"Yeni veri seti kaydedildi: {yeni_dosya}") 