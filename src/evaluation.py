
"""
evaluation.py
--------------
Model karşılaştırma, False Positive analizi ve sonuç raporu üretme modülü.

Kural tabanlı baseline ile AI modellerini karşılaştırır.
"""

import os
import warnings
from typing import Dict, List

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
    average_precision_score,
)


def _calculate_metrics(y_true, y_pred, scores=None):
    """Sınıflandırma ve yanlış alarm metriklerini hesaplar."""

    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        output_dict=True,
        zero_division=0,
    )

    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    roc_auc = float("nan")
    avg_precision = float("nan")

    if scores is not None and len(np.unique(y_true)) == 2:
        roc_auc = float(roc_auc_score(y_true, scores))
        avg_precision = float(average_precision_score(y_true, scores))

    return {
        "precision": report["1"]["precision"],
        "recall": recall,
        "f1_score": report["1"]["f1-score"],
        "accuracy": report["accuracy"],
        "false_positive_rate": fpr,
        "true_positive_rate": recall,
        "roc_auc": roc_auc,
        "avg_precision": avg_precision,
        "confusion_matrix": cm,
    }


def rule_based_detector(
    X_test: np.ndarray,
    feature_names: List[str],
    threshold_std: float = 2.0,
) -> np.ndarray:
    """
    Basit kural tabanlı baseline.

    ÖNEMLİ:
    X_test StandardScaler ile ölçeklendirilmiş olmalıdır.
    Bu nedenle önce scaler ile orijinal ölçeğe dönüştürülür.

    Kurallar:
    - Kullanıcı hacim z-score değeri eşik üstünde.
    - Mesai dışı transfer hacmi 100 MB üzerinde.
    - Başarısız giriş sayısı 1'den fazla.
    """

    scaler_path = "results/scaler.pkl"

    if not os.path.exists(scaler_path):
        raise FileNotFoundError(
            f"Scaler bulunamadı: {scaler_path}. "
            "Önce preprocess() çalıştırılmalı."
        )

    scaler = joblib.load(scaler_path)

    X_raw = scaler.inverse_transform(X_test)

    fn_map = {name: i for i, name in enumerate(feature_names)}

    required_features = [
        "volume_zscore",
        "off_hours_volume",
        "failed_logins",
    ]

    missing = [name for name in required_features if name not in fn_map]

    if missing:
        raise ValueError(
            "Kural tabanlı sistem için eksik özellikler: "
            + ", ".join(missing)
        )

    vol_idx = fn_map["volume_zscore"]
    off_idx = fn_map["off_hours_volume"]
    fail_idx = fn_map["failed_logins"]

    predictions = np.zeros(len(X_raw), dtype=int)

    for i in range(len(X_raw)):
        vol_z = X_raw[i, vol_idx]
        off_vol = X_raw[i, off_idx]
        fail = X_raw[i, fail_idx]

        if abs(vol_z) > threshold_std:
            predictions[i] = 1

        elif off_vol > 100:
            predictions[i] = 1

        elif fail > 1:
            predictions[i] = 1

    return predictions


def _print_comparison(comparison: pd.DataFrame, rule_fpr: float):
    print("\n" + "=" * 85)
    print("MODEL KARŞILAŞTIRMA TABLOSU")
    print("=" * 85)
    print(comparison.to_string(index=False))
    print("=" * 85)

    print("\n[★] Kural tabanlı sisteme göre False Positive değişimi:")

    for _, row in comparison.iterrows():
        method = row["Yöntem"]

        if method == "Kural Tabanlı (SIEM)":
            continue

        model_fpr = row["False Positive Rate"]

        if rule_fpr <= 0:
            print(f"    {method}: Karşılaştırma yapılamadı (baseline FPR = 0)")
            continue

        change = ((model_fpr - rule_fpr) / rule_fpr) * 100

        if change < 0:
            print(f"    {method}: %{abs(change):.1f} azalma")
        elif change > 0:
            print(f"    {method}: %{change:.1f} artış")
        else:
            print(f"    {method}: Değişim yok")

    best_model = comparison.loc[
        comparison["F1-Score"].idxmax()
    ]

    print("\n[★] Test sonuçlarına göre en yüksek F1:")
    print(
        f"    {best_model['Yöntem']} "
        f"(F1={best_model['F1-Score']:.4f})"
    )

    print(
        "\n[!] Bu sonuçlar sentetik test verisine aittir. "
        "Gerçek ortam başarısını kanıtlamaz."
    )


def compare_models(
    y_test: np.ndarray,
    if_results: Dict,
    ae_results: Dict,
    feature_names: List[str],
    X_test: np.ndarray,
) -> pd.DataFrame:
    """Isolation Forest, Autoencoder ve kural tabanlı sistemi karşılaştırır."""

    y_test = np.asarray(y_test).astype(int)

    # Kural tabanlı baseline
    rule_pred = rule_based_detector(X_test, feature_names)
    rule_metrics = _calculate_metrics(y_test, rule_pred)

    comparison_rows = [
        {
            "Yöntem": "Kural Tabanlı (SIEM)",
            "Precision": rule_metrics["precision"],
            "Recall": rule_metrics["recall"],
            "F1-Score": rule_metrics["f1_score"],
            "False Positive Rate": rule_metrics["false_positive_rate"],
            "ROC-AUC": "N/A",
            "Yanlış Alarm Oranı (%)": (
                rule_metrics["false_positive_rate"] * 100
            ),
        },
        {
            "Yöntem": "Isolation Forest (AI)",
            "Precision": if_results["precision"],
            "Recall": if_results["recall"],
            "F1-Score": if_results["f1_score"],
            "False Positive Rate": if_results["false_positive_rate"],
            "ROC-AUC": if_results["roc_auc"],
            "Yanlış Alarm Oranı (%)": (
                if_results["false_positive_rate"] * 100
            ),
        },
        {
            "Yöntem": "Autoencoder (AI)",
            "Precision": ae_results["precision"],
            "Recall": ae_results["recall"],
            "F1-Score": ae_results["f1_score"],
            "False Positive Rate": ae_results["false_positive_rate"],
            "ROC-AUC": ae_results["roc_auc"],
            "Yanlış Alarm Oranı (%)": (
                ae_results["false_positive_rate"] * 100
            ),
        },
    ]

    comparison = pd.DataFrame(comparison_rows)

    numeric_columns = [
        "Precision",
        "Recall",
        "F1-Score",
        "False Positive Rate",
        "Yanlış Alarm Oranı (%)",
    ]

    for column in numeric_columns:
        comparison[column] = comparison[column].round(4)

    comparison["ROC-AUC"] = comparison["ROC-AUC"].apply(
        lambda value: round(value, 4)
        if isinstance(value, (int, float, np.number))
        and np.isfinite(value)
        else "N/A"
    )

    _print_comparison(
        comparison,
        rule_metrics["false_positive_rate"],
    )

    return comparison


def save_results(
    df: pd.DataFrame,
    if_results: Dict,
    ae_results: Dict,
    y_test: np.ndarray,
    output_path: str = "results/anomaly_scores.csv",
):
    """Anomali skorlarını ve tahminlerini CSV olarak kaydeder."""

    y_test = np.asarray(y_test)
    n = len(y_test)

    arrays = {
        "if_prediction": if_results["y_pred"],
        "if_anomaly_score": if_results["y_scores"],
        "ae_prediction": ae_results["y_pred"],
        "ae_reconstruction_error": ae_results["y_scores"],
    }

    for name, values in arrays.items():
        if len(values) != n:
            raise ValueError(
                f"{name} uzunluğu ({len(values)}) "
                f"test etiketi uzunluğuyla ({n}) eşleşmiyor."
            )

    results_df = pd.DataFrame({
        "true_label": y_test,
        **arrays,
    })

    output_dir = os.path.dirname(output_path)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    results_df.to_csv(
        output_path,
        index=False,
        encoding="utf-8-sig",
    )

    print(f"[✓] Sonuçlar kaydedildi: {output_path}")

    return results_df
