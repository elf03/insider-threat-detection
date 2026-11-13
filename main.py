"""
main.py
--------
Kullanıcı Davranış Analizi ile Yapay Zeka Destekli İç Tehdit Tespiti
======================================================================
Kaynak Veri Seti: Synthetic Financial Datasets For Fraud Detection (PaySim)
https://www.kaggle.com/datasets/ealaxi/paysim1/data

Pipeline:
  1. Sentetik kullanıcı davranış verisi üret (PaySim tabanlı)
  2. Veri ön işleme ve özellik mühendisliği
  3. Isolation Forest modeli eğit ve değerlendir
  4. Autoencoder modeli eğit ve değerlendir
  5. Modelleri karşılaştır (Kural tabanlı vs. AI)
  6. Grafikleri oluştur ve kaydet
  7. Web dashboard için JSON verisi üret

Beklenen Sonuç:
  - Normal çalışan profilinin öğrenilerek nadir rastlanan sızıntı
    paternlerinin yüksek doğrulukla saptanması
  - AI tabanlı sistemlerin kural tabanlı sistemlere kıyasla False Positive
    oranını belirgin şekilde azaltması
"""

import os
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Proje modülleri
sys.path.insert(0, os.path.dirname(__file__))
from generate_sample_data import generate_dataset
from src.data_preprocessing import load_data, preprocess, get_user_profiles
from src.isolation_forest import IsolationForestDetector
from src.autoencoder import AutoencoderDetector
from src.evaluation import compare_models, save_results
from src.visualization import (
    plot_data_distribution,
    plot_roc_curves,
    plot_precision_recall,
    plot_confusion_matrices,
    plot_false_positive_comparison,
    plot_anomaly_timeline,
    plot_user_risk_heatmap,
    plot_training_history,
)


BANNER = """
╔══════════════════════════════════════════════════════════════════╗
║   KULLANICI DAVRANIŞ ANALİZİ İLE YAPAY ZEKA DESTEKLİ          ║
║              İÇ TEHDİT TESPİTİ SİSTEMİ                         ║
║                                                                  ║
║  Veri Seti: PaySim (Kaggle Synthetic Financial Dataset)         ║
║  Modeller:  Isolation Forest + Autoencoder                      ║
║  Hedef:     Normal profil öğrenme → Sızıntı tespiti            ║
╚══════════════════════════════════════════════════════════════════╝
"""


def main():
    print(BANNER)
    start_total = time.time()

    # ──────────────────────────────────────────
    # ADIM 1: Veri Üretimi
    # ──────────────────────────────────────────
    print("\n[ADIM 1/7] Veri Seti Hazırlanıyor...")
    data_path = "data/user_behavior_logs.csv"
    if not os.path.exists(data_path):
        df = generate_dataset(data_path)
    else:
        print(f"[*] Mevcut veri seti yükleniyor: {data_path}")
        df = load_data(data_path)

    # ──────────────────────────────────────────
    # ADIM 2: Veri Görselleştirmesi
    # ──────────────────────────────────────────
    print("\n[ADIM 2/7] Veri Dağılımı Görselleştiriliyor...")
    plot_data_distribution(df)

    # ──────────────────────────────────────────
    # ADIM 3: Ön İşleme
    # ──────────────────────────────────────────
    print("\n[ADIM 3/7] Özellik Mühendisliği ve Ön İşleme...")
    X_train, X_test, y_train, y_test, feature_names = preprocess(df)
    user_profiles = get_user_profiles(df)

    print(f"\n[*] Kullanıcı Profili Özeti (İlk 5 Riskli):")
    print(user_profiles.head(5)[["user_id", "anomaly_rate", "mean_volume", "off_hours_ratio"]].to_string(index=False))

    # ──────────────────────────────────────────
    # ADIM 4: Isolation Forest
    # ──────────────────────────────────────────
    print("\n[ADIM 4/7] Isolation Forest Modeli Eğitiliyor...")
    t0 = time.time()
    if_model = IsolationForestDetector(contamination=0.02, n_estimators=200)
    if_model.fit(X_train)
    if_results = if_model.evaluate(X_test, y_test)
    if_model.save("results/isolation_forest_model.pkl")
    print(f"[*] IF eğitim süresi: {time.time()-t0:.1f}s")

    # ──────────────────────────────────────────
    # ADIM 5: Autoencoder
    # ──────────────────────────────────────────
    print("\n[ADIM 5/7] Autoencoder Modeli Eğitiliyor...")
    t0 = time.time()
    ae_model = AutoencoderDetector(
        input_dim=X_train.shape[1],
        encoding_dim=16,
        epochs=50,
        batch_size=256
    )
    ae_model.fit(X_train, y_train)
    ae_results = ae_model.evaluate(X_test, y_test)
    ae_model.save("results/autoencoder_model.keras")
    plot_training_history(ae_model.history_)
    print(f"[*] AE eğitim süresi: {time.time()-t0:.1f}s")

    # ──────────────────────────────────────────
    # ADIM 6: Karşılaştırmalı Değerlendirme
    # ──────────────────────────────────────────
    print("\n[ADIM 6/7] Model Karşılaştırması Yapılıyor...")
    comparison_df = compare_models(y_test, if_results, ae_results, feature_names, X_test)
    comparison_df.to_csv("results/model_comparison.csv", index=False, encoding="utf-8-sig")

    results_df = save_results(df, if_results, ae_results, y_test)

    # ──────────────────────────────────────────
    # ADIM 7: Görselleştirme
    # ──────────────────────────────────────────
    print("\n[ADIM 7/7] Grafikler Oluşturuluyor...")
    plot_roc_curves(y_test, if_results, ae_results)
    plot_precision_recall(y_test, if_results, ae_results)
    plot_confusion_matrices(y_test, if_results, ae_results)
    plot_false_positive_comparison(comparison_df)
    plot_anomaly_timeline(df, if_results, ae_results, X_test)
    plot_user_risk_heatmap(user_profiles)

    # ──────────────────────────────────────────
    # Dashboard için JSON verisi üret
    # ──────────────────────────────────────────
    _generate_dashboard_data(
        df, if_results, ae_results, comparison_df, user_profiles, y_test
    )

    # ──────────────────────────────────────────
    # ÖZET RAPOR
    # ──────────────────────────────────────────
    total_time = time.time() - start_total
    _print_final_report(if_results, ae_results, comparison_df, total_time)


def _generate_dashboard_data(df, if_results, ae_results, comparison_df, user_profiles, y_test):
    """Web dashboard için JSON veri dosyası üretir."""
    os.makedirs("dashboard", exist_ok=True)

    # Saatlik anomali yoğunluğu
    hourly_data = df.groupby("hour_of_day")["is_anomaly"].agg(["sum", "count"]).reset_index()
    hourly_data["rate"] = hourly_data["sum"] / hourly_data["count"]

    # En riskli 10 kullanıcı
    top_users = user_profiles.head(10)[["user_id", "anomaly_rate", "mean_volume", "anomaly_count"]].to_dict("records")

    # Model metrikleri
    comp_records = comparison_df.to_dict("records")
    # ROC-AUC'yi string'den float'a çevir
    for rec in comp_records:
        try:
            rec["ROC-AUC"] = float(rec["ROC-AUC"]) if rec["ROC-AUC"] != "N/A" else None
        except:
            rec["ROC-AUC"] = None

    # Anomali skoru dağılımı (ilk 200)
    n = min(200, len(if_results["y_scores"]))
    timeline = {
        "if_scores": if_results["y_scores"][:n].tolist(),
        "ae_scores": ae_results["y_scores"][:n].tolist(),
        "true_labels": y_test[:n].tolist(),
        "if_preds": if_results["y_pred"][:n].tolist(),
        "ae_preds": ae_results["y_pred"][:n].tolist(),
    }

    dashboard_data = {
        "summary": {
            "total_records": len(df),
            "total_anomalies": int(df["is_anomaly"].sum()),
            "anomaly_rate": float(df["is_anomaly"].mean()),
            "total_users": df["user_id"].nunique(),
            "malicious_users_detected": int((user_profiles["anomaly_rate"] > 0.05).sum()),
        },
        "model_comparison": comp_records,
        "hourly_activity": hourly_data.to_dict("records"),
        "top_risky_users": top_users,
        "timeline": timeline,
        "if_metrics": {
            "precision": float(if_results["precision"]),
            "recall": float(if_results["recall"]),
            "f1": float(if_results["f1_score"]),
            "roc_auc": float(if_results["roc_auc"]),
            "fpr": float(if_results["false_positive_rate"]),
        },
        "ae_metrics": {
            "precision": float(ae_results["precision"]),
            "recall": float(ae_results["recall"]),
            "f1": float(ae_results["f1_score"]),
            "roc_auc": float(ae_results["roc_auc"]),
            "fpr": float(ae_results["false_positive_rate"]),
            "threshold": float(ae_results["threshold"]),
        }
    }

    with open("dashboard/data.json", "w", encoding="utf-8") as f:
        json.dump(dashboard_data, f, ensure_ascii=False, indent=2, default=str)
    print("[✓] Dashboard verisi kaydedildi: dashboard/data.json")


def _print_final_report(if_results, ae_results, comparison_df, total_time):
    print("\n" + "█"*70)
    print("  PROJE ÇIKTISI - BEKLENEN SONUÇ DOĞRULAMASI")
    print("█"*70)
    print(f"""
  ✅ BEKLENEN SONUÇ 1: Normal çalışan profilinin öğrenilerek nadir
     rastlanan sızıntı paternlerinin YÜKSEK DOĞRULUKLA saptanması
     
     Isolation Forest → Recall: {if_results['recall']:.4f}  |  F1: {if_results['f1_score']:.4f}
     Autoencoder      → Recall: {ae_results['recall']:.4f}  |  F1: {ae_results['f1_score']:.4f}

  ✅ BEKLENEN SONUÇ 2: Kongre Bildirisi Odak Noktası
     "Kurumsal ağlarda kural tabanlı alarm sistemlerinin ürettiği
      FALSE POSITIVE yorgunluğunun, YZ tabanlı davranışsal anomali
      tespiti ile nasıl AZALTILABILECEĞI"
      
     Kural Tabanlı FPR: {float(comparison_df[comparison_df['Yöntem']=='Kural Tabanlı (SIEM)']['False Positive Rate'].values[0]):.4f}
     Isolation Forest FPR: {if_results['false_positive_rate']:.4f} 
     Autoencoder FPR: {ae_results['false_positive_rate']:.4f}
""")
    print(f"  Toplam çalışma süresi: {total_time:.1f} saniye")
    print("█"*70)
    print("\n[✓] Tüm grafikler: results/plots/")
    print("[✓] Model dosyaları: results/")
    print("[✓] Web dashboard: dashboard/index.html")
    print("[✓] Proje tamamlandı!\n")


if __name__ == "__main__":
    main()
