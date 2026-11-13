# 🛡️ Kullanıcı Davranış Analizi ile Yapay Zeka Destekli İç Tehdit Tespiti

> **Yapay Zeka ve Siber Güvenlik Dersi — Proje 7**

---

## 📋 Proje Özeti

Kurum içi veri sızıntıları ve yetki istismarları (örneğin iç tehditler gibi) dışarıdan gelen saldırılardan çok daha zor tespit edilmektedir. Bu projede; kullanıcıların sisteme **giriş saatleri**, **dosya erişimleri** ve **veri aktarım boyutları** gibi günlük aktiviteleri bir zaman serisi olarak ele alınmaktadır.

**Isolation Forest** ve **Autoencoder** gibi anomali tespit algoritmalarını kullanarak, ele geçirilmiş hesapları veya kötü niyetli çalışanları tespit eden bir **Kullanıcı Davranış Analizi (UBA)** sistemi tasarlanmıştır.

---

## 📊 Kullanılan Veri Seti

**Synthetic Financial Datasets For Fraud Detection (PaySim)**  
🔗 https://www.kaggle.com/datasets/ealaxi/paysim1/data

| PaySim Sütunu | Kullanıcı Davranış Karşılığı |
|---|---|
| `step` | Zaman adımı (saat cinsinden, 1-744) |
| `amount` | Veri aktarım boyutu (MB) |
| `type` | Aktivite türü (LOGIN, FILE_ACCESS, DATA_TRANSFER...) |
| `nameOrig` | Kullanıcı ID |
| `isFraud` | Anomali etiketi (0=normal, 1=iç tehdit) |

PaySim veri seti, gerçek dünya finansal işlem paternlerini taklit eden sentetik bir veri setidir. Bu projede kurumsal kullanıcı davranış verisine dönüştürülmüştür.

---

## 🏗️ Proje Yapısı

```
insider-threat-detection/
├── main.py                        ← Ana pipeline (tüm adımları çalıştırır)
├── generate_sample_data.py        ← PaySim tabanlı sentetik veri üretici
├── requirements.txt               ← Python bağımlılıkları
├── src/
│   ├── data_preprocessing.py      ← Özellik mühendisliği & normalizasyon
│   ├── isolation_forest.py        ← Isolation Forest modeli
│   ├── autoencoder.py             ← Autoencoder modeli (TensorFlow/Keras)
│   ├── evaluation.py              ← Model karşılaştırma & FP analizi
│   └── visualization.py           ← Matplotlib grafikleri
├── dashboard/
│   ├── index.html                 ← Web dashboard
│   ├── style.css                  ← Dark tema CSS
│   └── app.js                     ← Chart.js interaktif grafikler
├── data/
│   └── user_behavior_logs.csv     ← Üretilen veri (main.py sonrası)
└── results/
    ├── anomaly_scores.csv         ← Model tahminleri
    ├── model_comparison.csv       ← Karşılaştırma tablosu
    ├── isolation_forest_model.pkl ← IF modeli
    ├── autoencoder_model.keras    ← Autoencoder modeli
    └── plots/                     ← Üretilen grafikler (8 adet)
```

---

## ⚙️ Kurulum ve Çalıştırma

### 1. Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

### 2. Pipeline'ı Çalıştır

```bash
python main.py
```

**Pipeline adımları:**
1. Sentetik kullanıcı davranış verisi üretimi (PaySim tabanlı)
2. Veri görselleştirmesi
3. Özellik mühendisliği ve normalizasyon
4. Isolation Forest eğitimi ve değerlendirme
5. Autoencoder eğitimi ve değerlendirme
6. Karşılaştırmalı değerlendirme (Kural tabanlı vs. AI)
7. Grafiklerin oluşturulması

### 3. Web Dashboard'ı Aç

`main.py` çalıştıktan sonra `dashboard/index.html` dosyasını tarayıcıda açın.

---

## 🤖 Kullanılan Modeller

### Isolation Forest
```
Parametreler:
  n_estimators=200     (200 ağaç)
  contamination=0.02   (beklenen anomali oranı %2)
  max_samples="auto"   (otomatik örnekleme)

Çalışma prensibi:
  Anomaliler normal gözlemlerden daha az bölme gerektirir.
  "İzole etme" sayısı az → Anomali
```

### Autoencoder (TensorFlow/Keras)
```
Mimari: Encoder-Decoder
  Encoder: 13 → 64 → 32 → 16 (latent space)
  Decoder: 16 → 32 → 64 → 13
  
Kayıp: MSE (Mean Squared Error)
Anomali kriteri: Reconstruction Error > threshold (P95)

Çalışma prensibi:
  Normal veriler düşük yeniden yapılandırma hatası üretir.
  Anomaliler (normal profile uymayan) yüksek hata üretir → Tespit
```

---

## 🏷️ Özellik Mühendisliği

| Özellik | Açıklama |
|---|---|
| `data_volume_mb` | Ham veri transfer hacmi |
| `hour_of_day` | Günün saati (0-23) |
| `is_working_hours` | Mesai saatinde mi? (1/0) |
| `failed_logins` | Başarısız giriş denemesi sayısı |
| `unique_resources` | Erişilen kaynak sayısı |
| `volume_zscore` | **Kullanıcı bazında z-score** (normal profilden sapma) |
| `off_hours_volume` | Mesai dışı veri transfer hacmi |
| `login_failure_rate` | Başarısız giriş/kaynak oranı |
| `resource_intensity` | Kaynak erişim yoğunluğu |
| `balance_drop_ratio` | Aktivite skoru değişim oranı |

---

## 📈 Üretilen Grafikler

| # | Dosya | İçerik |
|---|---|---|
| 1 | `01_data_distribution.png` | Veri dağılımı (pasta, histogram, saatlik) |
| 2 | `02_roc_curves.png` | ROC-AUC eğrisi karşılaştırması |
| 3 | `03_precision_recall.png` | Precision-Recall eğrisi |
| 4 | `04_confusion_matrices.png` | Confusion matrix (IF + AE) |
| 5 | `05_false_positive_comparison.png` | **FP karşılaştırması (Kongre odak)** |
| 6 | `06_anomaly_timeline.png` | Anomali skor zaman serisi |
| 7 | `07_user_risk_heatmap.png` | Kullanıcı risk profili haritası |
| 8 | `08_autoencoder_training.png` | Autoencoder eğitim süreci |

---

## 🎯 Beklenen Sonuçlar

### ✅ Normal Çalışan Profili Öğrenme
Her kullanıcının bireysel davranış profili çıkarılır. Giriş saatleri, transfer boyutları ve kaynak erişim paternleri referans profil olarak öğrenilir.

### ✅ Nadir Sızıntı Paternlerinin Yüksek Doğrulukla Saptanması
3 farklı iç tehdit senaryosu tespit edilir:
- **DATA_EXFILTRATION** — Gece büyük veri transferleri
- **PRIVILEGE_ABUSE** — Yetkisiz kaynak erişimleri
- **CREDENTIAL_THEFT** — Çoklu başarısız giriş denemeleri

### ✅ Kongre Bildirisi Odak Noktası
> *"Kurumsal ağlarda kural tabanlı alarm sistemlerinin ürettiği 'False Positive' (yanlış pozitif) yorgunluğunun, yapay zeka tabanlı davranışsal anomali tespiti ile nasıl azaltılabileceği"*

| Yöntem | False Positive Rate |
|---|---|
| Kural Tabanlı (SIEM) | ~%60-70 |
| Isolation Forest (AI) | ~%8-12 |
| Autoencoder (AI) | ~%6-10 |

---

## 📚 Referanslar

1. Liu, F.T., Ting, K.M., Zhou, Z.H. (2008). *Isolation Forest*. ICDM 2008.
2. Hinton, G.E., Salakhutdinov, R.R. (2006). *Reducing the dimensionality of data with neural networks*. Science.
3. Lopez-Rojas, E.A. (2016). *PaySim: A Financial Mobile Money Simulator for Fraud Detection*. EMSS 2016.
4. Chandola, V., Banerjee, A., Kumar, V. (2009). *Anomaly detection: A survey*. ACM Computing Surveys.

---

## 👥 Proje Bilgileri

- **Ders:** Yapay Zeka ve Siber Güvenlik
- **Konu:** Kullanıcı Davranış Analizi ile İç Tehdit Tespiti
- **Veri Seti:** Synthetic Financial Datasets For Fraud Detection (PaySim)
- **Algoritmalar:** Isolation Forest, Autoencoder
