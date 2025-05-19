import pandas as pd

#dosyanın kodlamasını tespit et
# import chardet

# with open("data/veri_setleri/tum_programlar_netler.json",  "r", encoding="utf-8", errors="replace") as f:
#     content= f.read()  

# #Temizlenmiş içeriği yeni bir dosyaya kaydet
# with open("data/veri_setleri/tum_programlar_netler_cleaned.json", "w", encoding="utf-8") as f:
#     f.write(content)

# print("Bozuk karakterler temizlendi ve dosya 'tum_programlar_netler_cleaned.json' olarak kaydedildi.")
#dosya Johab kodlamasında , korece karakterler var

#dosyayı UTF-8 kodlamasına çevirme

# with open("data/veri_setleri/tum_programlar_netler.json", "r", encoding="johab") as f:
#     content = f.read()

# with open("data/veri_setleri/tum_programlar_netler_utf8.json", "w", encoding="utf-8") as f:
#     f.write(content)

# print("Dosya UTF-8 formatına dönüştürüldü.")

#tam veri seti
# tam_veri=pd.read_json("data/veri_setleri/tum_programlar_netler_cleaned.json")
# tam_veri.to_csv("data/veri_setleri/sıralama_net_verileri.csv", index=False)
# print("Veri seti CSV formatına dönüştürüldü ve 'sıralama_net_verileri.csv' olarak kaydedildi.")

#yeni veri setini yükleme
tum_veriler=pd.read_csv("data/veri_setleri/sıralama_net_verileri.csv")

# print(tum_veriler.head())
# print(tum_veriler.columns)

# TYT verilerini ayırma
tyt_veri = tum_veriler[["basari_sirasi", "tyt_turkce", "tyt_matematik", "tyt_sosyal", "tyt_fen"]].dropna()

# TYT verilerini kontrol et
# print("TYT Verileri:")
# print(tyt_veri.head())

# TYT verilerini kaydet
# tyt_veri.to_csv("data/veri_setleri/tyt_veri_seti.csv", index=False)
# print("TYT veri seti başarıyla kaydedildi.")

# 3. AYT Sayısal Verilerini Ayırma
# ayt_sayisal_veri = tum_veriler[["basari_sirasi", "ayt_matematik", "ayt_kimya", "ayt_biyoloji", "ayt_fizik"]].dropna()
# ayt_sayisal_veri.to_csv("data/veri_setleri/ayt_sayisal_veri_seti.csv", index=False)
# print("AYT Sayısal veri seti başarıyla kaydedildi.")

# # 4. AYT Eşit Ağırlık (EA) Verilerini Ayırma
# ayt_ea_veri = tum_veriler[["basari_sirasi", "ayt_matematik", "ayt_edebiyat", "ayt_cografya1"]].dropna()
# ayt_ea_veri.to_csv("data/veri_setleri/ayt_ea_veri_seti.csv", index=False)
# print("AYT Eşit Ağırlık veri seti başarıyla kaydedildi.")

# 5. AYT Sözel Verilerini Ayırma
# ayt_sozel_veri = tum_veriler[[
#     "basari_sirasi", "ayt_edebiyat", "ayt_tarih1", "ayt_cografya1",
#     "ayt_tarih2", "ayt_cografya2", "ayt_felsefe", "ayt_din_kulturu"
# ]].dropna()
# ayt_sozel_veri.to_csv("data/veri_setleri/ayt_sozel_veri_seti.csv", index=False)
# print("AYT Sözel veri seti başarıyla kaydedildi.")

# # 6. AYT Dil Verilerini Ayırma
# ayt_dil_veri = tum_veriler[["basari_sirasi", "ayt_dil"]].dropna()
# ayt_dil_veri.to_csv("data/veri_setleri/ayt_dil_veri_seti.csv", index=False)
# print("AYT Dil veri seti başarıyla kaydedildi.")

# Eksik sütunları doldur
# tum_veriler["ayt_tarih1"] = 0
# tum_veriler["ayt_tarih2"] = 0
# tum_veriler["ayt_cografya2"] = 0
# tum_veriler["ayt_felsefe"] = 0
# tum_veriler["ayt_din_kulturu"] = 0

# AYT Sözel Verilerini Ayırma
# ayt_sozel_veri = tum_veriler[["basari_sirasi", "ayt_edebiyat", "ayt_tarih1", "ayt_cografya1",
#     "ayt_tarih2", "ayt_cografya2", "ayt_felsefe", "ayt_din_kulturu"]].dropna()
# ayt_sozel_veri.to_csv("data/veri_setleri/ayt_sozel_veri_seti.csv", index=False)
# print("AYT Sözel veri seti başarıyla kaydedildi.")

#toplam net hesaplama

df = pd.read_csv("data/veri_setleri/ayt_sayisal_veri_seti.csv")
df["toplam_net"] = df[["ayt_matematik", "ayt_kimya", "ayt_biyoloji", "ayt_fizik"]].sum(axis=1)

df_ea = pd.read_csv("data/veri_setleri/ayt_ea_veri_seti.csv")
df_ea["toplam_net"] = df_ea[["ayt_matematik", "ayt_edebiyat", "ayt_cografya1"]].sum(axis=1)

df_tyt = pd.read_csv("data/veri_setleri/tyt_veri_seti.csv")
df_tyt["toplam_net"] = df_tyt[["tyt_turkce", "tyt_matematik", "tyt_sosyal", "tyt_fen"]].sum(axis=1)

df_sozel = pd.read_csv("data/veri_setleri/ayt_sozel_veri_seti.csv")
df_sozel["toplam_net"] = df_sozel[["ayt_edebiyat", "ayt_tarih1", "ayt_cografya1","ayt_tarih2", "ayt_cografya2", "ayt_felsefe", "ayt_din_kulturu"]].sum(axis=1)




