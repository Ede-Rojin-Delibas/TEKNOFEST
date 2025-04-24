## NET HESAPLAMA ##

#kullanıcıdan verileri al
def sinav_turu_sec():
    print("Sınav Türü: [1] TYT, [2] AYT")
    sinav_turu= input("Seçiminiz: ")
    
    if sinav_turu=='1':
        return "TYT",None
    elif sinav_turu=='2':
        print("Alanınız:[1] Sayısal, [2] Eşit Ağırlık, [3] Sözel")
        alan= input("Seçiminiz: ")
        if alan=='1':
            return "AYT", "Sayısal"
        elif alan=='2':
            return "AYT", "Eşit_Ağırlık"
        elif alan=='3':
            return "AYT", "Sözel"
        else:
            print("Geçersiz alan seçimi. Lütfen tekrar deneyin.")
            return sinav_turu_sec()
    else:
        print("Geçersiz seçim! Lütfen tekrar seçim yapınız.")
        return sinav_turu_sec()

# netleri al
def netleri_al(sinav_türü, alan=None):
    dersler=[]
    if sinav_türü=="TYT":
        dersler=["Türkçe","Matematik","Sosyal Bilimler","Fen Bilimleri"]
    elif sinav_türü=="AYT":
        if alan=="Sayısal":
            dersler=["Matematik","Fizik","Kimya","Biyoloji"]
        elif alan=="Eşit_Ağırlık":
            dersler=["Matematik","Türk Dili ve Edebiyatı","Tarih-1","Coğrafya-1"]
        elif alan=="Sözel":
            dersler=["Türk Dili ve Edebiyatı", "Tarih-1", "Coğrafya-1",
                "Tarih-2", "Coğrafya-2", "Felsefe Grubu", "Din Kültürü"
            ]
    net_verileri={}
    for ders in dersler:
        print(f"\n{ders}:")
        dogru=int(input("Doğru Sayısı: "))
        yanlis=int(input("Yanlış Sayısı: "))
        net_verileri[ders]={"dogru": dogru, "yanlis": yanlis}
    return net_verileri


#net hesaplama 
def net_hesapla(net_verileri):
    net_sonuclar = {}
    for ders, bilgiler in net_verileri.items():
        dogru = bilgiler["dogru"]
        yanlis = bilgiler["yanlis"]
        net = dogru - (yanlis * 0.25)
        net_sonuclar[ders] = round(net, 2)
    return net_sonuclar


### main / Ana program ###
def main():
    sinav, alan = sinav_turu_sec()
    netler = netleri_al(sinav, alan)
    net_sonuclar = net_hesapla(netler)

    print("\nNetleriniz:")
    for ders, net in net_sonuclar.items():
        print(f"{ders}: {net}")

#programı başlat
if __name__=="__main__":
    main()



