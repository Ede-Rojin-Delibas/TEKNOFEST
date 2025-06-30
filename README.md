# Yapay Zeka Destekli YKS Başarı Sıralaması Tahmin Projesi

Bu proje, üniversite adayı öğrencilerin deneme sınavı sonuçlarına (netlerine) dayanarak Yükseköğretim Kurumları Sınavı (YKS) başarı sıralamalarını tahmin etmelerine yardımcı olmak amacıyla geliştirilmiş bir yapay zeka uygulamasıdır.

## Projenin Amacı

Üniversiteye hazırlık süreci, öğrenciler için uzun ve zorlu bir maratondur. Bu süreçte öğrencilerin en büyük motivasyon kaynaklarından biri, hedeflerine ne kadar yakın olduklarını görmektir. Bu proje, öğrencilerin deneme sınavlarındaki netlerini kullanarak potansiyel başarı sıralamalarını tahmin eder ve onlara yol haritalarını çizerken gerçekçi bir veri sunar.

Projenin temel hedefleri şunlardır:
-   Farklı sınav türleri (TYT, AYT Sayısal, Sözel, Eşit Ağırlık ve Dil) için ders bazlı netlere dayalı başarı sıralaması tahminleri yapmak.
-   Geçmiş yılların sınav verileriyle eğitilmiş makine öğrenmesi modelleri kullanarak güvenilir sonuçlar üretmek.
-   Geliştirilen tahmin modellerini, farklı platformlarla kolayca entegre olabilecek bir API servisi olarak sunmak.

## Kullanılan Teknolojiler

-   **Programlama Dili:** Python
-   **Veri Analizi ve İşleme:** Pandas, NumPy
-   **Makine Öğrenmesi:** Scikit-learn
-   **Web Servisi (API):** Flask

## Proje Yapısı

Projenin ana dizin yapısı, modüler ve anlaşılır bir şekilde tasarlanmıştır:

```
.
├── app/                  # Uygulamanın ana kodları
│   ├── api/              # API endpoint'lerinin tanımlandığı modül
│   ├── ml/               # Veri işleme ve model eğitimi script'leri
│   ├── utils/            # Net hesaplama ve tahmin gibi yardımcı fonksiyonlar
│   └── __init__.py
├── data/                 # Veri setleri ve eğitilmiş modeller
│   ├── models/           # Eğitilmiş ve kaydedilmiş .joblib modelleri
│   └── veri_setleri/     # Ham ve işlenmiş .csv/.json veri setleri
├── tests/                # Test script'leri
├── run.py                # Uygulamayı (API) başlatan ana script
└── requirements.txt      # Gerekli Python kütüphaneleri
```

## Kurulum ve Kullanım

Projeyi yerel makinenizde çalıştırmak için aşağıdaki adımları izleyebilirsiniz.

1.  **Projeyi Klonlayın:**
    ```bash
    git clone <repository-url>
    cd teknofestProject
    ```

2.  **Gerekli Kütüphaneleri Yükleyin:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **API Sunucusunu Başlatın:**
    ```bash
    python run.py
    ```
    Sunucu varsayılan olarak `http://127.0.0.1:5000` adresinde çalışmaya başlayacaktır.

4.  **Tahmin İsteği Gönderin:**
    API'ye bir `POST` isteği göndererek tahmin alabilirsiniz. Aşağıda `TYT` için bir `curl` örneği verilmiştir:

    ```bash
    curl -X POST http://127.0.0.1:5000/api/tahmin/tyt \
    -H "Content-Type: application/json" \
    -d '{
        "turkce": 35.0,
        "sosyal": 15.0,
        "matematik": 30.0,
        "fen": 18.0
    }'
    ```

### Modelleri Yeniden Eğitmek (İsteğe Bağlı)

Eğer kendi veri setinizle modelleri yeniden eğitmek isterseniz:

1.  **Veriyi Hazırlayın:**
    `app/ml/prepare_data_v3.py` script'ini çalıştırarak ham veriyi modellere uygun hale getirin.
    ```bash
    python -m app.ml.prepare_data_v3
    ```
2.  **Modeli Eğitin:**
    `app/ml/train_v3.py` script'ini çalıştırarak yeni modelleri eğitin.
    ```bash
    python -m app.ml.train_v3
    ```
    Yeni modeller `data/models/v3` dizinine kaydedilecektir.

## Katkıda Bulunma

Projeye katkıda bulunmak isterseniz, lütfen bir "pull request" açın. Tüm katkılara açığız! 