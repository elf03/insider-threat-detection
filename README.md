# Yapay Zekâ Destekli İç Tehdit Tespit Sistemi

Kullanıcı davranışlarını analiz ederek olağan dışı aktiviteleri ve potansiyel iç tehditleri tespit etmek amacıyla geliştirilmiş, yapay zekâ destekli bir siber güvenlik projesidir.

Sistem; birden fazla makine öğrenmesi modeli, risk değerlendirme mekanizması ve Büyük Dil Modeli (LLM) destekli tehdit açıklama modülünü bir araya getirir.

Proje kapsamında özel olarak oluşturulmuş sentetik kullanıcı davranış verileri kullanılmaktadır.

## Projenin Amacı ve Özellikleri

Sistem, kullanıcı davranış kayıtlarını analiz ederek şüpheli aktiviteleri belirlemeyi ve olası iç tehditleri tespit etmeyi amaçlamaktadır.

Başlıca özellikler:

- Kullanıcı davranış verilerinin ön işlenmesi ve analizi
- Isolation Forest ile anomali tespiti
- Autoencoder tabanlı anomali tespiti
- Isolation Forest ve Autoencoder modellerini birleştiren Hybrid tespit sistemi
- LSTM tabanlı zamansal anomali tespiti
- Risk puanlama ve tehdit önceliklendirme mekanizması
- Ollama üzerinden yerel LLM destekli tehdit açıklamaları
- Etkileşimli web paneli (Dashboard)
- Modellerin performans karşılaştırması
- Tehdit raporlarının oluşturulması
- Risk puanlama ve tehdit açıklama modülleri için birim testleri

## Sistem Mimarisi

Sistem aşağıdaki aşamalardan oluşmaktadır:

1. **Veri Üretimi:** Sentetik kullanıcı davranış kayıtları oluşturulur.
2. **Veri Ön İşleme:** Veriler temizlenir, dönüştürülür ve kronolojik olarak eğitim ve test kümelerine ayrılır.
3. **Anomali Tespiti:** Isolation Forest, Autoencoder, Hybrid ve LSTM modelleri çalıştırılır.
4. **Model Değerlendirme:** Modeller Precision, Recall, F1-score, ROC-AUC ve False Positive Rate metrikleriyle karşılaştırılır.
5. **Risk Motoru:** Tespit edilen şüpheli davranışlara göre risk puanları hesaplanır.
6. **Tehdit Açıklama:** Yerel Ollama LLM modeli kullanılarak tespit edilen tehditler anlaşılır bir dille açıklanır.
7. **Dashboard:** Model performansları, kullanıcı risk seviyeleri, aktivite eğilimleri ve tehdit raporları görselleştirilir.

## Kullanılan Teknolojiler

- Python
- NumPy
- Pandas
- Scikit-learn
- Matplotlib
- Ollama
- HTML
- CSS
- JavaScript
- Pytest

## Proje Dosya Yapısı

```text
insider-threat-detection/
│
├── dashboard/
│   ├── index.html
│   ├── app.js
│   ├── style.css
│   └── data.json
│
├── src/
│   ├── autoencoder.py
│   ├── data_preprocessing.py
│   ├── evaluation.py
│   ├── hybrid_detector.py
│   ├── isolation_forest.py
│   ├── lstm_detector.py
│   ├── risk_engine.py
│   ├── threat_explainer.py
│   └── visualization.py
│
├── main.py
├── requirements.txt
├── test_risk_engine.py
├── test_threat_explainer.py
└── README.md
```

## Kurulum

Öncelikle GitHub deposunu bilgisayarınıza klonlayın:

```bash
git clone https://github.com/elf03/insider-threat-detection.git
cd insider-threat-detection
```

Python sanal ortamı oluşturun:

```bash
python -m venv venv
```

Windows PowerShell üzerinden sanal ortamı etkinleştirin:

```powershell
.\venv\Scripts\Activate.ps1
```

Gerekli kütüphaneleri yükleyin:

```bash
pip install -r requirements.txt
```

## Projenin Çalıştırılması

Ana uygulamayı çalıştırmak için:

```bash
python main.py
```

Bu komut veri hazırlama, model eğitimi, anomali tespiti, model değerlendirme, risk analizi, tehdit raporlarının oluşturulması ve Dashboard verilerinin hazırlanması aşamalarını yürütür.

## Dashboard'un Çalıştırılması

Proje ana dizinindeyken aşağıdaki komutu çalıştırın:

```bash
python -m http.server 8000 --directory dashboard
```

Ardından tarayıcıdan aşağıdaki adrese gidin:

```text
http://localhost:8000
```

Dashboard üzerinden model performanslarını, riskli kullanıcıları, aktivite grafiklerini ve oluşturulan tehdit raporlarını inceleyebilirsiniz.

## Testlerin Çalıştırılması

Projedeki testleri çalıştırmak için:

```bash
python -m pytest -v
```

## Model Değerlendirme Metrikleri

Sistemdeki modeller aşağıdaki performans metrikleri kullanılarak değerlendirilir:

| Metrik | Açıklama |
|---|---|
| Precision | Tespit edilen tehditlerin ne kadarının gerçekten tehdit olduğunu gösterir. |
| Recall | Gerçek tehditlerin ne kadarının tespit edildiğini gösterir. |
| F1-score | Precision ve Recall değerlerinin harmonik ortalamasıdır. |
| ROC-AUC | Modelin sınıfları ayırt etme başarısını ölçer. |
| False Positive Rate (FPR) | Normal davranışların yanlışlıkla tehdit olarak sınıflandırılma oranıdır. |

Model sonuçları, proje kapsamında oluşturulan sentetik test veri kümesine dayanmaktadır. Bu sonuçlar gerçek ortamlardaki performansın garantisi olarak değerlendirilmemelidir.

## LLM Entegrasyonu

Tehdit açıklama modülünde yerel olarak çalışan Ollama modeli kullanılmaktadır.

LLM, tespit edilen şüpheli aktiviteleri analiz ederek anlaşılır açıklamalar ve güvenlik analistlerine yardımcı olabilecek değerlendirmeler üretir.

LLM açıklama modülü, deterministik çalışan risk puanlama motorunu destekler; onun yerine geçmez.

## Projenin Sınırları

Bu proje eğitim ve araştırma amacıyla geliştirilmiştir. Sentetik kullanıcı davranış verileri kullanmaktadır ve doğrudan kurumsal ortamlarda kullanılabilecek üretim seviyesinde bir güvenlik izleme sistemi olarak değerlendirilmemelidir.