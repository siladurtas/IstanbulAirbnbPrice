# 🏡 Airbnb İstanbul Fiyat Tahmin Modeli

Bu proje, İstanbul’daki Airbnb ilanlarının gecelik konaklama fiyatlarını tahmin etmeyi amaçlayan bir makine öğrenmesi uygulamasıdır. Sistem, veri ön işleme, özellik mühendisliği, model eğitimi, değerlendirme ve REST API üzerinden tahmin sunma adımlarını içeren uçtan uca bir çözüm sunar.

---

## 📁 Proje Yapısı

```
├── src/
│   ├── DataLoader.py         # Ham veriyi okuma ve ilk kontroller
│   ├── Preprocessing.py      # Eksik veri temizliği, dönüşümler ve encoding
│   ├── FeatureSelection.py   # Korelasyon ve model bazlı özellik seçimi
│   ├── Train.py              # Makine öğrenmesi modellerinin eğitimi
│   ├── Evaluation.py         # Performans metriklerinin hesaplanması
│   ├── Test.py               # Test işlemleri ve tahmin çıktısı üretimi
│   ├── Main.py               # Eğitim ve değerlendirme sürecini başlatan ana script
│   └── app.py                # Flask tabanlı REST API
├── data/
│   └── listings.csv.gz       # Airbnb İstanbul veri seti
├── outputs/
│   ├── models/               # Eğitilmiş modellerin kayıtlı halleri
│   ├── figures/              # Görselleştirmeler (özellik önemleri, korelasyon haritası vb.)
│   └── reports/              # Model karşılaştırmaları ve değerlendirme raporları
└── requirements.txt          # Proje için gerekli Python kütüphaneleri
```

---

## 🔧 Özellikler

* ✅ Eksik veri temizliği ve dönüşümleri
* ✅ Özellik mühendisliği ve seçimi
* ✅ 6 farklı regresyon modelinin karşılaştırılması
* ✅ Ensemble (Stacking) modeliyle gelişmiş tahmin
* ✅ Detaylı görselleştirme: Korelasyon, öğrenme eğrileri, model karşılaştırmaları
* ✅ Flask tabanlı API ile gerçek zamanlı tahmin desteği

---

## 🎯 Kullanılan Özellikler

Modelde kullanılan öznitelikler:

* `price_per_bedroom`: Oda başına düşen fiyat
* `accommodates`: Konaklayabilecek maksimum kişi sayısı
* `bedrooms`: Yatak odası sayısı
* `minimum_nights`: Minimum konaklama süresi (gece)

> Özellik mühendisliğinde bu değişkenler normalize edilmiş ve eksik değerler kontrol edilmiştir.

---

## 🧠 Kullanılan Modeller ve Sonuçları

| Model                 | RMSE       | R² Skoru   |
| --------------------- | ---------- | ---------- |
| Random Forest         | 0.1270     | 0.9592     |
| XGBoost               | 0.1302     | 0.9572     |
| Support Vector Reg.   | 0.1463     | 0.9459     |
| K-Nearest Neighbors   | 0.1332     | 0.9552     |
| Lasso Regression      | 0.5548     | 0.2219     |
| **Stacking Ensemble** | **0.1219** | **0.9624** |

> 📌 En iyi sonuç **Stacking Ensemble** modeli ile elde edilmiştir.

---

## ⚙️ Kurulum

1. Gerekli kütüphaneleri yükleyin:

   ```bash
   pip install -r requirements.txt
   ```

2. Veri dosyasını `data/` klasörüne yerleştirin (`listings.csv.gz`).

---

## 🚀 Kullanım

### 1. Model Eğitimi

```bash
python src/Main.py
```

### 2. Model Testi

```bash
python src/Test.py
```

### 3. API'yi Başlatma

```bash
python src/app.py
```

### 4. API Üzerinden Tahmin

```python
import requests

data = {
    "price_per_bedroom": 1000,
    "accommodates": 3,
    "bedrooms": 1,
    "minimum_nights": 3
}

response = requests.post('http://localhost:5000/predict', json=data)
print(response.json())
```

---

## 📊 Çıktılar

* `outputs/models/`: Eğitilmiş modellerin `.pkl` halleri
* `outputs/figures/`:

  * Korelasyon matrisi
  * Özellik önemleri (barplot)
  * Model karşılaştırma grafikleri
* `outputs/reports/`:

  * RMSE, MAE, R² skorları
  * En iyi modelin açıklaması
  * Tahmin sonuçları

* Eksik veri oranı %40’ı aşan sütunlar silinmiştir.
* Özellik gruplamaları (örneğin renk, bölge gibi) gerekiyorsa 0-1 dummy değil, anlamlı kategorik sınıflamalarla birleştirilmiştir.
* Korelasyon analizi hem ham verilerle hem de özellik mühendisliği sonrası yapılmıştır.
