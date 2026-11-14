"""
main.py
KullanÄ±cÄ± DavranÄ±ÅŸ Analizi ile Yapay Zeka Destekli Ä°Ã§ Tehdit Tespiti

Pipeline:
1. Sentetik kullanÄ±cÄ± davranÄ±ÅŸ verisini yÃ¼kle/Ã¼ret.
2. Veriyi kronolojik olarak eÄŸitim, doÄŸrulama ve test kÃ¼melerine ayÄ±r.
3. Isolation Forest modelini eÄŸit, doÄŸrulama verisiyle eÅŸiÄŸini kalibre et.
4. Autoencoder modelini eÄŸit, doÄŸrulama verisiyle eÅŸiÄŸini belirle.
5. Modelleri test kÃ¼mesinde karÅŸÄ±laÅŸtÄ±r.
6. Grafikleri oluÅŸtur.
7. Web dashboard iÃ§in JSON verisi Ã¼ret.

Not: Mevcut veri Ã¼reticisi sentetik kullanÄ±cÄ± davranÄ±ÅŸ verisi Ã¼retir.
GerÃ§ek PaySim veri seti kullanÄ±lmÄ±yorsa proje PaySim kullanÄ±yor ÅŸeklinde tanÄ±tÄ±lmamalÄ±dÄ±r.
"""

import os
import sys
import json
import time
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Proje kÃ¶k dizinini Python yoluna ekle.
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_DIR)

from generate_sample_data import generate_dataset
from src.data_preprocessing import load_data, preprocess, get_user_profiles
from src.isolation_forest import IsolationForestDetector
from src.autoencoder import AutoencoderDetector
from src.hybrid_detector import HybridAnomalyDetector
from src.lstm_detector import LSTMAnomalyDetector, build_user_sequences
from src.risk_engine import RiskEngine
from src.threat_explainer import ThreatExplainer
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
â•”â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•—
â•‘       KULLANICI DAVRANIÅ ANALÄ°ZÄ° Ä°LE YAPAY ZEKA DESTEKLÄ°        â•‘
â•‘                  Ä°Ã‡ TEHDÄ°T TESPÄ°TÄ° SÄ°STEMÄ°                     â•‘
â•‘                                                                  â•‘
â•‘  Veri Seti: Sentetik KullanÄ±cÄ± DavranÄ±ÅŸ Verisi                  â•‘
â•‘  Modeller:  Isolation Forest + Autoencoder + Hybrid             â•‘
â•‘  Hedef:     Normal profil Ã¶ÄŸrenme â†’ Anomali tespiti             â•‘
â•šâ•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
"""


def _safe_float(value):
    """NaN/inf deÄŸerlerini JSON uyumlu None deÄŸerine dÃ¶nÃ¼ÅŸtÃ¼rÃ¼r."""
    value = float(value)
    return value if np.isfinite(value) else None


def _generate_dashboard_data(
    df,
    if_results,
    ae_results,
    hybrid_results,
    lstm_results,
    comparison_df,
    user_profiles,
    y_test,
    threat_reports=None,
):
    """Web dashboard iÃ§in JSON veri dosyasÄ± Ã¼retir."""
    os.makedirs("dashboard", exist_ok=True)

        # Tehdit raporları parametreyle gelmezse kayıtlı dosyadan yükle.
    if not threat_reports:
        report_path = os.path.join("results", "threat_reports.json")
        if os.path.exists(report_path):
            with open(report_path, "r", encoding="utf-8") as report_file:
                threat_reports = json.load(report_file)

    threat_reports = threat_reports or []
    print(f"[DASHBOARD] Aktarılan tehdit raporu: {len(threat_reports)}")

    # Saatlik anomali yoÄŸunluÄŸu.
    hourly_data = (
        df.groupby("hour_of_day")["is_anomaly"]
        .agg(["sum", "count"])
        .reset_index()
    )
    hourly_data["rate"] = hourly_data["sum"] / hourly_data["count"]

    # En riskli 10 kullanÄ±cÄ±.
    user_columns = [
        "user_id",
        "anomaly_rate",
        "mean_volume",
        "anomaly_count",
    ]
    available_user_columns = [
        col for col in user_columns if col in user_profiles.columns
    ]
    top_users = (
        user_profiles.head(10)[available_user_columns].to_dict("records")
    )

    # Model karÅŸÄ±laÅŸtÄ±rma sonuÃ§larÄ±.
    comp_records = comparison_df.to_dict("records")
    for record in comp_records:
        value = record.get("ROC-AUC")
        try:
            record["ROC-AUC"] = (
                _safe_float(value)
                if value is not None and value != "N/A"
                else None
            )
        except (TypeError, ValueError):
            record["ROC-AUC"] = None

    # Test kÃ¼mesindeki ilk 200 skor.
    n = min(
        200,
        len(if_results["y_scores"]),
        len(ae_results["y_scores"]),
        len(hybrid_results["y_scores"]),
        len(lstm_results["y_scores"]),
    )
    timeline = {
        "if_scores": np.asarray(if_results["y_scores"][:n]).tolist(),
        "ae_scores": np.asarray(ae_results["y_scores"][:n]).tolist(),
        "hybrid_scores": np.asarray(hybrid_results["y_scores"][:n]).tolist(),
        "true_labels": np.asarray(y_test[:n]).tolist(),
        "if_preds": np.asarray(if_results["y_pred"][:n]).tolist(),
        "ae_preds": np.asarray(ae_results["y_pred"][:n]).tolist(),
        "hybrid_preds": np.asarray(hybrid_results["y_pred"][:n]).tolist(),
        "lstm_scores": np.asarray(lstm_results["y_scores"][:n]).tolist(),
        "lstm_preds": np.asarray(lstm_results["y_pred"][:n]).tolist(),
    }

    dashboard_data = {
        "summary": {
            "total_records": int(len(df)),
            "total_anomalies": int(df["is_anomaly"].sum()),
            "anomaly_rate": _safe_float(df["is_anomaly"].mean()),
            "total_users": int(df["user_id"].nunique()),
            "users_over_5pct_anomaly_rate": int(
                (user_profiles["anomaly_rate"] > 0.05).sum()
            ),
        },
        "model_comparison": comp_records,
        "hourly_activity": hourly_data.to_dict("records"),
        "top_risky_users": top_users,
        "threat_reports": threat_reports or [],
        "timeline": timeline,
        "if_metrics": {
            "precision": _safe_float(if_results["precision"]),
            "recall": _safe_float(if_results["recall"]),
            "f1": _safe_float(if_results["f1_score"]),
            "roc_auc": _safe_float(if_results["roc_auc"]),
            "fpr": _safe_float(if_results["false_positive_rate"]),
            "threshold": _safe_float(if_results.get("threshold", np.nan)),
        },
        "ae_metrics": {
            "precision": _safe_float(ae_results["precision"]),
            "recall": _safe_float(ae_results["recall"]),
            "f1": _safe_float(ae_results["f1_score"]),
            "roc_auc": _safe_float(ae_results["roc_auc"]),
            "fpr": _safe_float(ae_results["false_positive_rate"]),
            "threshold": _safe_float(ae_results["threshold"]),
        },
        "lstm_metrics": {
            "precision": _safe_float(lstm_results["precision"]),
            "recall": _safe_float(lstm_results["recall"]),
            "f1": _safe_float(lstm_results["f1_score"]),
            "roc_auc": _safe_float(lstm_results["roc_auc"]),
            "fpr": _safe_float(lstm_results["false_positive_rate"]),
            "threshold": _safe_float(lstm_results["threshold"]),
        },
        "hybrid_metrics": {
            "precision": _safe_float(hybrid_results["precision"]),
            "recall": _safe_float(hybrid_results["recall"]),
            "f1": _safe_float(hybrid_results["f1_score"]),
            "roc_auc": _safe_float(hybrid_results["roc_auc"]),
            "fpr": _safe_float(hybrid_results["false_positive_rate"]),
            "threshold": _safe_float(hybrid_results["threshold"]),
        },
    }


    # JSON dosyasına yazılmadan önce NaN ve sonsuz değerleri temizle.
    def clean_json(value):
        if isinstance(value, dict):
            return {key: clean_json(item) for key, item in value.items()}

        if isinstance(value, (list, tuple)):
            return [clean_json(item) for item in value]

        if isinstance(value, np.ndarray):
            return clean_json(value.tolist())

        if isinstance(value, np.integer):
            return int(value)

        if isinstance(value, (np.floating, float)):
            return float(value) if np.isfinite(value) else None

        return value

    dashboard_data = clean_json(dashboard_data)


    with open("dashboard/data.json", "w", encoding="utf-8") as file:
        json.dump(
            dashboard_data,
            file,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
            default=str,
        )

    print("[âœ“] Dashboard verisi kaydedildi: dashboard/data.json")


def _print_final_report(if_results, ae_results, hybrid_results, lstm_results, comparison_df, total_time):
    """Terminale kÄ±sa sonuÃ§ Ã¶zeti yazdÄ±rÄ±r."""
    print("\n" + "â–ˆ" * 70)
    print("  PROJE SONUÃ‡ RAPORU")
    print("â–ˆ" * 70)

    print(
        f"\nIsolation Forest â†’ Recall: {if_results['recall']:.4f}"
        f" | F1: {if_results['f1_score']:.4f}"
    )
    print(
        f"Autoencoder      â†’ Recall: {ae_results['recall']:.4f}"
        f" | F1: {ae_results['f1_score']:.4f}"
    )
    print(
        f"Hibrit Model     â†’ Recall: {hybrid_results['recall']:.4f}"
        f" | F1: {hybrid_results['f1_score']:.4f}"
    )
    print(
        f"LSTM             â†’ Recall: {lstm_results['recall']:.4f}"
        f" | F1: {lstm_results['f1_score']:.4f}"
    )

    print("\nYanlÄ±ÅŸ Pozitif OranlarÄ± (FPR):")
    print(
        f"Isolation Forest: "
        f"{if_results['false_positive_rate']:.4f}"
    )
    print(
        f"Autoencoder:      "
        f"{ae_results['false_positive_rate']:.4f}"
    )
    print(
        f"Hibrit Model:     "
        f"{hybrid_results['false_positive_rate']:.4f}"
    )
    print(
        f"LSTM:             "
        f"{lstm_results['false_positive_rate']:.4f}"
    )

    if "YÃ¶ntem" in comparison_df.columns:
        baseline = comparison_df[
            comparison_df["YÃ¶ntem"] == "Kural TabanlÄ± (SIEM)"
        ]
        if not baseline.empty:
            baseline_fpr = float(
                baseline["False Positive Rate"].iloc[0]
            )
            print(f"Kural tabanlÄ±:    {baseline_fpr:.4f}")

    best_row = comparison_df.loc[
        comparison_df["F1-Score"].astype(float).idxmax()
    ]
    print(
        f"\nEn yÃ¼ksek F1: {best_row['YÃ¶ntem']} "
        f"(F1={float(best_row['F1-Score']):.4f})"
    )

    print(
        "\n[!] Bu sonuÃ§lar sentetik test verisine aittir. "
        "GerÃ§ek ortam baÅŸarÄ±sÄ±nÄ± kanÄ±tlamaz."
    )
    print(f"\nToplam Ã§alÄ±ÅŸma sÃ¼resi: {total_time:.1f} saniye")
    print("â–ˆ" * 70)
    print("[âœ“] Grafikler: results/plots/")
    print("[âœ“] Model dosyalarÄ±: results/")
    print("[âœ“] Web dashboard verisi: dashboard/data.json")
    print("[âœ“] Proje tamamlandÄ±!\n")


def main():
    print(BANNER)
    start_total = time.time()

    os.makedirs("data", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    os.makedirs("results/plots", exist_ok=True)
    os.makedirs("dashboard", exist_ok=True)

    # --------------------------------------------------
    # ADIM 1: Veri Ã¼retimi veya yÃ¼kleme
    # --------------------------------------------------
    print("\n[ADIM 1/8] Veri Seti HazÄ±rlanÄ±yor...")
    data_path = os.path.join("data", "user_behavior_logs.csv")

    if not os.path.exists(data_path):
        df = generate_dataset(data_path)
    else:
        print(f"[*] Mevcut veri seti yÃ¼kleniyor: {data_path}")
        df = load_data(data_path)

    # --------------------------------------------------
    # ADIM 2: Veri daÄŸÄ±lÄ±mÄ± gÃ¶rselleÅŸtirmesi
    # --------------------------------------------------
    print("\n[ADIM 2/8] Veri DaÄŸÄ±lÄ±mÄ± GÃ¶rselleÅŸtiriliyor...")
    plot_data_distribution(df)

    # --------------------------------------------------
    # ADIM 3: Ã–n iÅŸleme ve kronolojik veri bÃ¶lme
    # --------------------------------------------------
    print("\n[ADIM 3/8] Ã–zellik MÃ¼hendisliÄŸi ve Ã–n Ä°ÅŸleme...")

    (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        feature_names,
    ) = preprocess(df)

    user_profiles = get_user_profiles(df)

    print("\n[*] KullanÄ±cÄ± Profili Ã–zeti (Ä°lk 5 Riskli):")
    profile_columns = [
        "user_id",
        "anomaly_rate",
        "mean_volume",
        "off_hours_ratio",
    ]
    print(
        user_profiles.head(5)[profile_columns].to_string(index=False)
    )

    # --------------------------------------------------
    # ADIM 4: Isolation Forest
    # --------------------------------------------------
    print("\n[ADIM 4/8] Isolation Forest Modeli EÄŸitiliyor...")
    t0 = time.time()

    if_model = IsolationForestDetector(
        contamination="auto",
        n_estimators=200,
    )

    # Model yalnÄ±zca eÄŸitim verisiyle eÄŸitilir.
    if_model.fit(X_train, y_train)

    # EÅŸik yalnÄ±zca normal doÄŸrulama kayÄ±tlarÄ±yla kalibre edilir.
    X_val_normal = X_val[y_val == 0]
    if len(X_val_normal) == 0:
        raise ValueError(
            "Isolation Forest iÃ§in doÄŸrulama kÃ¼mesinde normal kayÄ±t yok."
        )

    if_model.calibrate_threshold(
        X_val_normal,
        percentile=95,
    )

    # Test kÃ¼mesi yalnÄ±zca nihai deÄŸerlendirme iÃ§indir.
    if_results = if_model.evaluate(X_test, y_test)
    if_model.save("results/isolation_forest_model.pkl")

    print(f"[*] IF Ã§alÄ±ÅŸma sÃ¼resi: {time.time() - t0:.1f}s")

    # --------------------------------------------------
    # ADIM 5: Autoencoder
    # --------------------------------------------------
    print("\n[ADIM 5/8] Autoencoder Modeli EÄŸitiliyor...")
    t0 = time.time()

    ae_model = AutoencoderDetector(
        input_dim=X_train.shape[1],
        encoding_dim=16,
        epochs=50,
        batch_size=256,
    )

    # EÄŸitim, erken durdurma ve eÅŸik kalibrasyonu iÃ§in
    # dÄ±ÅŸ doÄŸrulama kÃ¼mesi kullanÄ±lÄ±r.
    ae_model.fit(
        X_train,
        y_train,
        X_val,
        y_val,
    )

    ae_results = ae_model.evaluate(X_test, y_test)
    ae_model.save("results/autoencoder_model.pkl")

    plot_training_history(ae_model.history_)

    print(f"[*] AE Ã§alÄ±ÅŸma sÃ¼resi: {time.time() - t0:.1f}s")

    # --------------------------------------------------
    # ADIM 6: Hibrit model
    # --------------------------------------------------
    print("\n[ADIM 6/8] Hibrit Model Kalibre Ediliyor ve Test Ediliyor...")
    t0 = time.time()

    hybrid_model = HybridAnomalyDetector(
        isolation_forest=if_model,
        autoencoder=ae_model,
        if_weight=0.5,
        ae_weight=0.5,
        threshold_percentile=95.0,
    )

    # Hibrit skor normalizasyonu ve karar eÅŸiÄŸi sadece normal
    # validation kayÄ±tlarÄ± kullanÄ±larak belirlenir.
    hybrid_model.calibrate(X_val_normal)
    hybrid_results = hybrid_model.evaluate(X_test, y_test)
    hybrid_model.save("results/hybrid_model.pkl")

    print(f"[*] Hibrit model Ã§alÄ±ÅŸma sÃ¼resi: {time.time() - t0:.1f}s")

    # --------------------------------------------------
    # ADIM 7: LSTM zamansal davranÄ±ÅŸ modeli
    # --------------------------------------------------
    print("\n[ADIM 7/9] KullanÄ±cÄ± BazlÄ± LSTM Modeli EÄŸitiliyor...")
    t0 = time.time()
    (
        X_train_seq,
        y_train_seq,
        X_val_seq,
        y_val_seq,
        X_test_seq,
        y_test_seq,
        lstm_metadata,
    ) = build_user_sequences(
        df,
        sequence_length=24,
        scaler_path="results/scaler.pkl"
    )
    print(f"[âœ“] LSTM dizileri â€” Train: {X_train_seq.shape}, Validation: {X_val_seq.shape}, Test: {X_test_seq.shape}")
    lstm_test_metadata = lstm_metadata["test"]

    if len(lstm_test_metadata) != len(X_test_seq):
        raise ValueError(
            "LSTM test kayÄ±tlarÄ± ile metadata sayÄ±sÄ± eÅŸleÅŸmiyor."
        )

    print(
        f"[*] LSTM test metadata kayÄ±t sayÄ±sÄ±: "
        f"{len(lstm_test_metadata)}"
    )

    lstm_model = LSTMAnomalyDetector(
        input_dim=X_train_seq.shape[2],
        hidden_dim=12,
        epochs=8,
        batch_size=128,
        learning_rate=0.001,
        threshold_percentile=95.0,
    )
    lstm_model.fit(X_train_seq, y_train_seq, X_val_seq, y_val_seq)
    lstm_results = lstm_model.evaluate(X_test_seq, y_test_seq)

    # --------------------------------------------------
    # ADIM 7.5: DORT MODEL + RISK ENGINE + LLM ENTEGRASYONU
    # --------------------------------------------------
    print("\n[ADIM 7.5] Tehditler birlestiriliyor ve risk analiz ediliyor...")

    sorted_test_df = df.copy()
    sorted_test_df["timestamp"] = pd.to_datetime(
        sorted_test_df["timestamp"], errors="coerce"
    )
    sorted_test_df = sorted_test_df.sort_values(
        ["timestamp", "user_id"], kind="mergesort"
    ).reset_index(drop=True)

    test_start = int(len(sorted_test_df) * 0.80)
    standard_test_df = sorted_test_df.iloc[test_start:].reset_index(drop=True)

    if len(standard_test_df) != len(X_test):
        raise ValueError("Standart test verisi ile model test sonuclari eslesmiyor.")

    model_results = {
        "Isolation Forest": if_results,
        "Autoencoder": ae_results,
        "Hybrid": hybrid_results,
    }

    for model_name, result in model_results.items():
        if len(result["y_pred"]) != len(standard_test_df):
            raise ValueError(f"{model_name} tahmin sayisi test verisiyle uyusmuyor.")

    lstm_lookup = {}

    for i, record in enumerate(lstm_test_metadata):
        key = (
            str(record["user_id"]),
            pd.Timestamp(record["timestamp"]).isoformat()
        )

        if key in lstm_lookup:
            raise ValueError(f"Tekrarlanan LSTM metadata anahtari: {key}")

        lstm_lookup[key] = {
            "prediction": int(lstm_results["y_pred"][i]),
            "score": float(lstm_results["y_scores"][i])
        }

    if len(lstm_lookup) != len(standard_test_df):
        raise ValueError("LSTM metadata ile standart test kayitlari eslesmiyor.")

    risk_engine = RiskEngine()
    candidates = []

    for i, row in standard_test_df.iterrows():
        key = (
            str(row["user_id"]),
            pd.Timestamp(row["timestamp"]).isoformat()
        )

        if key not in lstm_lookup:
            raise ValueError(f"LSTM kaydi bulunamadi: {key}")

        model_scores = {}
        detected_by = []

        for model_name, result in model_results.items():
            prediction = int(result["y_pred"][i])
            model_scores[model_name] = float(result["y_scores"][i])

            if prediction == 1:
                detected_by.append(model_name)

        lstm_prediction = lstm_lookup[key]["prediction"]
        model_scores["LSTM"] = lstm_lookup[key]["score"]

        if lstm_prediction == 1:
            detected_by.append("LSTM")

        if not detected_by:
            continue

        threat_data = {
            "user_id": str(row["user_id"]),
            "timestamp": pd.Timestamp(row["timestamp"]).isoformat(),
            "detected_by": detected_by,
            "model_scores": model_scores,
            "data_volume_mb": float(row["data_volume_mb"]),
            "hour_of_day": int(row["hour_of_day"]),
            "failed_logins": int(row["failed_logins"]),
            "unique_resources": int(row["unique_resources"]),
        }

        risk_result = risk_engine.evaluate(threat_data)

        candidates.append({
            "threat_data": threat_data,
            "risk_result": risk_result
        })

    candidates.sort(
        key=lambda item: (
            item["risk_result"]["risk_score"],
            item["risk_result"]["model_count"],
            max(item["threat_data"]["model_scores"].values())
        ),
        reverse=True
    )

    selected_threats = candidates[:10]
    threat_explainer = ThreatExplainer()
    threat_reports = []

    print(f"[*] Tespit edilen supheli kayit: {len(candidates)}")
    print(f"[*] LLM ile aciklanacak olay sayisi: {len(selected_threats)}")

    for index, item in enumerate(selected_threats, start=1):
        threat_data = item["threat_data"]
        risk_result = item["risk_result"]

        print(
            f"[*] Olay {index}/{len(selected_threats)}: "
            f"{threat_data['user_id']} - "
            f"{risk_result['risk_level']} "
            f"({risk_result['risk_score']}/100)"
        )

        try:
            explanation = threat_explainer.explain_threat(
                threat_data, risk_result
            )
        except Exception as error:
            explanation = {
                "summary": "Yerel LLM aciklamasi olusturulamadi.",
                "confidence_note": str(error),
                "evidence": risk_result["evidence"],
                "recommended_actions": risk_result["recommended_actions"],
                "risk_score": risk_result["risk_score"],
                "risk_level": risk_result["risk_level"],
            }

        threat_reports.append({
            "user_id": threat_data["user_id"],
            "timestamp": threat_data["timestamp"],
            "risk_score": risk_result["risk_score"],
            "risk_level": risk_result["risk_level"],
            "detected_by": threat_data["detected_by"],
            "model_scores": threat_data["model_scores"],
            "threat_data": threat_data,
            "risk_assessment": risk_result,
            "explanation": explanation,
        })

    os.makedirs("results", exist_ok=True)

    with open(
        "results/threat_reports.json",
        "w",
        encoding="utf-8"
    ) as report_file:
        json.dump(
            threat_reports,
            report_file,
            ensure_ascii=False,
            indent=4
        )

    print(f"[OK] Tehdit raporlari kaydedildi: {len(threat_reports)} olay")
    print("[OK] Dosya: results/threat_reports.json")

    lstm_model.save("results/lstm_model.pkl")
    print(f"[*] LSTM Ã§alÄ±ÅŸma sÃ¼resi: {time.time() - t0:.1f}s")

    print("\n============================================================")
    print(" LSTM - SONUÃ‡LAR")
    print("============================================================")
    print(f" Precision: {lstm_results['precision']:.4f}")
    print(f" Recall: {lstm_results['recall']:.4f}")
    print(f" F1-Score: {lstm_results['f1_score']:.4f}")
    print(f" ROC-AUC: {lstm_results['roc_auc']:.4f}")
    print(f" False Positive Rate: {lstm_results['false_positive_rate']:.4f}")
    print(" Confusion Matrix:")
    print(lstm_results["confusion_matrix"])
    print("============================================================")

    # --------------------------------------------------
    # ADIM 8: KarÅŸÄ±laÅŸtÄ±rmalÄ± deÄŸerlendirme
    # --------------------------------------------------
    print("\n[ADIM 8/9] Model KarÅŸÄ±laÅŸtÄ±rmasÄ± YapÄ±lÄ±yor...")

    comparison_df = compare_models(
        y_test,
        if_results,
        ae_results,
        feature_names,
        X_test,
    )

    # Mevcut karÅŸÄ±laÅŸtÄ±rma fonksiyonunu deÄŸiÅŸtirmeden hibrit sonucu ekle.
    hybrid_row = {
        "YÃ¶ntem": "Hibrit Model (IF + AE)",
        "Precision": hybrid_results["precision"],
        "Recall": hybrid_results["recall"],
        "F1-Score": hybrid_results["f1_score"],
        "False Positive Rate": hybrid_results["false_positive_rate"],
        "ROC-AUC": hybrid_results["roc_auc"],
        "YanlÄ±ÅŸ Alarm OranÄ± (%)": hybrid_results["false_positive_rate"] * 100,
    }
    comparison_df = pd.concat(
        [comparison_df, pd.DataFrame([hybrid_row])],
        ignore_index=True,
    )

    lstm_row = {
        "YÃ¶ntem": "LSTM (Zamansal AI)",
        "Precision": lstm_results["precision"],
        "Recall": lstm_results["recall"],
        "F1-Score": lstm_results["f1_score"],
        "False Positive Rate": lstm_results["false_positive_rate"],
        "ROC-AUC": lstm_results["roc_auc"],
        "YanlÄ±ÅŸ Alarm OranÄ± (%)": lstm_results["false_positive_rate"] * 100,
    }
    comparison_df = pd.concat([comparison_df, pd.DataFrame([lstm_row])], ignore_index=True)
    print("\nGÃœNCEL MODEL KARÅILAÅTIRMASI (LSTM dahil):")
    print(comparison_df.to_string(index=False))

    comparison_df.to_csv(
        "results/model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    save_results(
        df,
        if_results,
        ae_results,
        y_test,
    )

    # --------------------------------------------------
    # ADIM 9: Grafikler ve dashboard
    # --------------------------------------------------
    print("\n[ADIM 9/9] Grafikler OluÅŸturuluyor...")

    plot_roc_curves(y_test, if_results, ae_results)
    plot_precision_recall(y_test, if_results, ae_results)
    plot_confusion_matrices(y_test, if_results, ae_results)
    plot_false_positive_comparison(comparison_df)
    plot_anomaly_timeline(df, if_results, ae_results, X_test)
    plot_user_risk_heatmap(user_profiles)

    _generate_dashboard_data(
        df,
        if_results,
        ae_results,
        hybrid_results,
        lstm_results,
        comparison_df,
        user_profiles,
        y_test,
        threat_reports,
    )

    total_time = time.time() - start_total
    _print_final_report(
        if_results,
        ae_results,
        hybrid_results,
        lstm_results,
        comparison_df,
        total_time,
    )


if __name__ == "__main__":
    main()
