import pickle
import numpy as np
import sys

def tahmin_et(model_path, net, is_polinomial=False):
    with open(model_path, "rb") as f:
        if is_polinomial:
            model,poly = pickle.load(f)
            net_poly = poly.transform(np.array([[net]]))
            log_siralama=model.predict(net_poly)[0]
            return int(np.exp(log_siralama))
        else:
            model = pickle.load(f)
            return int(model.predict(np.array([[net]]))[0])
        
# Komut satırından parametre alma
if len(sys.argv) != 3:
    print("Kullanım: python tahmin_et.py <sinav_turu> <net>")
    print("Örnek: python tahmin_et.py tyt 85.75")
    sys.exit()

sinav = sys.argv[1].lower()
net = float(sys.argv[2])

# Model yolunu ve polynomial kullanılıp kullanılmadığını belirleme
model_paths = {
    "tyt": ("modeller/model_tyt.pkl", False),
    "sayisal": ("modeller/model_sayisal.pkl", False),
    "sozel": ("modeller/model_sozel.pkl", False),
    "ea": ("modeller/model_ayt_ea.pkl", True)
}

if sinav not in model_paths:
    print("Geçersiz sınav türü! Geçerli türler: tyt, sayisal, sozel, ea")
    sys.exit()

model_path, is_poly = model_paths[sinav]
tahmini_siralama = tahmin_et(model_path, net, is_polinomial=is_poly)

print(f"Tahmini Sıralama ({sinav.upper()}): {tahmini_siralama:,}")