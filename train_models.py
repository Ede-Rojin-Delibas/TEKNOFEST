import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures

#veri
# df=pd.read_csv("veriler/tyt.csv") 

# #özellikler ve hedef değişken
# X=df[["tyt_net"]] #bağımsız değişken
# y=df["siralama"]  #bağımlı değişken

# #veriyi eğitim ve test olarak ayırma
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# #modeli oluşturma ve eğitme
# model=LinearRegression()
# model.fit(X_train, y_train)

# #model performansını kontrol etme
# y_pred=model.predict(X_test)
# mse=mean_squared_error(y_test, y_pred)
# print(f"Model MSE: {mse:.2f}")

# #modeli dosyaya kaydetme
# with open ("modeller/model_tyt.pkl","wb") as f:
#     pickle.dump(model, f)

# print("TYT modeli başarıyla kaydedildi: modeller/model_tyt.pkl")

def modeli_egit_ve_kaydet(dosya_adi, ozellik, hedef, model_adi):
    print(f"{model_adi} modeli eğitiliyor...")

    #veri seti okuma
    df=pd.read_csv(dosya_adi)

    # Bağımsız ve bağımlı değişkenleri ayırma
    X = df[[ozellik]]
    y = df[hedef] 

    # Eğitim ve test setlerine ayır
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Modeli oluştur ve eğit
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Tahmin ve hata hesabı
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    print(f"{model_adi} Model MSE: {mse:.2f}")

    # Modeli .pkl dosyasına kaydet
    with open(f"modeller/{model_adi}.pkl", "wb") as f:
        pickle.dump(model, f)

    print(f"{model_adi} başarıyla kaydedildi.\n")
    
# Tüm alanlar için modelleri eğit
modeli_egit_ve_kaydet("veriler/tyt.csv", "tyt_net", "siralama", "model_tyt")
modeli_egit_ve_kaydet("veriler/ayt_sayisal.csv", "ayt_net", "siralama", "model_sayisal")
modeli_egit_ve_kaydet("veriler/ayt_sozel.csv", "ayt_net", "siralama", "model_sozel")
modeli_egit_ve_kaydet("veriler/ayt_ea.csv", "ayt_net", "siralama", "model_ea")

#ea için log dönüşümü yapıldı ve ona uygun bir model olan polinomial regresyon geliştirildi.
# Orijinal veri
data = {
    "siralama": [1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 150000, 200000],
    "ayt_net": [66.62, 59.36, 56.98, 51.36, 49.82, 45.96, 44.43, 44.21, 42.96, 40.43, 39.61, 39.11, 34.82, 30.82]
}
df = pd.DataFrame(data)

# 1. Veri Sayısını Artır
extra_rows = []
for i in range(10_000, 200_000, 10_000):
    if i not in df["siralama"].values:
        net = np.interp(i, df["siralama"], df["ayt_net"])
        extra_rows.append({"siralama": i, "ayt_net": net})

df_extra = pd.DataFrame(extra_rows)
df = pd.concat([df, df_extra], ignore_index=True).sort_values(by="siralama")

# 2. Log Dönüşümü
df["log_siralama"] = np.log(df["siralama"])

# 3. Polynomial Regression
X = df[["ayt_net"]]
y = df["log_siralama"]

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_poly, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# 4. Modeli Kaydet
with open("modeller/model_ayt_ea.pkl", "wb") as f:
    pickle.dump((model, poly), f)

print("AYT Eşit Ağırlık modeli başarıyla kaydedildi.")

# 5. Tahmin Sonuçları ve CSV’ye Kaydetme
df["tahmin_log_siralama"] = model.predict(poly.transform(df[["ayt_net"]]))
df["tahmin_siralama"] = np.exp(df["tahmin_log_siralama"])
df.to_csv("veriler/ayt_ea_processed.csv", index=False)
