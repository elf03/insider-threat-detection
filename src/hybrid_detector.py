"""
Hibrit anomali tespit modülü.

Isolation Forest ve Autoencoder skorlarını, yalnızca normal validation
verisi üzerinden ampirik yüzdelik sıralara dönüştürerek birleştirir.
Test verisi kalibrasyon veya eşik belirleme için kullanılmaz.
"""

import os
import joblib
import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


class HybridAnomalyDetector:
    """
    İki önceden eğitilmiş modeli birleştirir.

    Her iki model için de yüksek skor = daha şüpheli kayıt varsayılır.
    Skorlar validation normal dağılımına göre [0, 1] aralığında
    ampirik yüzdelik skora çevrilir.
    """

    def __init__(
        self,
        isolation_forest,
        autoencoder,
        if_weight=0.5,
        ae_weight=0.5,
        threshold_percentile=95.0,
    ):
        if if_weight < 0 or ae_weight < 0:
            raise ValueError("Model ağırlıkları negatif olamaz.")
        if if_weight + ae_weight <= 0:
            raise ValueError("En az bir model ağırlığı pozitif olmalı.")
        if not 0 < threshold_percentile < 100:
            raise ValueError("threshold_percentile 0 ile 100 arasında olmalı.")

        self.isolation_forest = isolation_forest
        self.autoencoder = autoencoder
        total = if_weight + ae_weight
        self.if_weight = if_weight / total
        self.ae_weight = ae_weight / total
        self.threshold_percentile = threshold_percentile

        self.if_reference_scores_ = None
        self.ae_reference_scores_ = None
        self.threshold_ = None
        self.is_fitted_ = False

    @staticmethod
    def _validate_scores(scores, name):
        scores = np.asarray(scores, dtype=float).reshape(-1)
        if scores.size == 0:
            raise ValueError(f"{name} skor dizisi boş.")
        if not np.isfinite(scores).all():
            raise ValueError(f"{name} skorlarında NaN veya sonsuz değer var.")
        return scores

    @staticmethod
    def _percentile_scores(scores, reference_scores):
        """
        Her skoru normal validation referans dağılımındaki ampirik
        yüzdelik konumuna dönüştürür. Yüksek değer daha anormaldir.
        """
        scores = np.asarray(scores, dtype=float).reshape(-1)
        reference_scores = np.sort(
            np.asarray(reference_scores, dtype=float).reshape(-1)
        )
        # Sağdan arama: eşit skorlar aynı veya daha düşük referansların
        # oranını alır. Sonuç [0, 1] aralığındadır.
        return np.searchsorted(
            reference_scores, scores, side="right"
        ) / len(reference_scores)

    def _raw_model_scores(self, X):
        if_scores = self._validate_scores(
            self.isolation_forest.anomaly_scores(X),
            "Isolation Forest",
        )
        ae_scores = self._validate_scores(
            self.autoencoder.anomaly_scores(X),
            "Autoencoder",
        )
        if len(if_scores) != len(ae_scores):
            raise ValueError("İki modelin skor sayıları eşleşmiyor.")
        return if_scores, ae_scores

    def calibrate(self, X_normal_validation):
        """
        Yalnızca normal validation kayıtlarıyla skor referanslarını ve
        hibrit karar eşiğini belirler.
        """
        X_normal_validation = np.asarray(X_normal_validation)
        if X_normal_validation.ndim != 2 or len(X_normal_validation) == 0:
            raise ValueError("Normal validation verisi boş veya geçersiz.")

        if_scores, ae_scores = self._raw_model_scores(
            X_normal_validation
        )

        self.if_reference_scores_ = if_scores.copy()
        self.ae_reference_scores_ = ae_scores.copy()

        if_norm = self._percentile_scores(
            if_scores, self.if_reference_scores_
        )
        ae_norm = self._percentile_scores(
            ae_scores, self.ae_reference_scores_
        )
        hybrid_normal_scores = (
            self.if_weight * if_norm + self.ae_weight * ae_norm
        )

        self.threshold_ = float(
            np.percentile(
                hybrid_normal_scores, self.threshold_percentile
            )
        )
        self.is_fitted_ = True

        print("[✓] Hibrit skor kalibrasyonu tamamlandı.")
        print(
            f"[✓] IF ağırlığı: {self.if_weight:.2f} | "
            f"AE ağırlığı: {self.ae_weight:.2f}"
        )
        print(
            f"[✓] Hibrit eşik: {self.threshold_:.6f} "
            f"(normal validation, P{self.threshold_percentile:g})"
        )
        return self.threshold_

    def anomaly_scores(self, X):
        """Birleştirilmiş hibrit anomali skorunu döndürür."""
        if not self.is_fitted_:
            raise RuntimeError("Önce calibrate() metodunu çağırmalısın.")

        if_scores, ae_scores = self._raw_model_scores(X)
        if_norm = self._percentile_scores(
            if_scores, self.if_reference_scores_
        )
        ae_norm = self._percentile_scores(
            ae_scores, self.ae_reference_scores_
        )
        return (
            self.if_weight * if_norm + self.ae_weight * ae_norm
        )

    def predict(self, X):
        """0=Normal, 1=Anomali."""
        if not self.is_fitted_ or self.threshold_ is None:
            raise RuntimeError("Önce calibrate() metodunu çağırmalısın.")
        return (self.anomaly_scores(X) > self.threshold_).astype(int)

    def evaluate(self, X_test, y_test):
        """Hibrit modeli ayrılmış test kümesinde değerlendirir."""
        y_test = np.asarray(y_test).astype(int).reshape(-1)
        scores = self.anomaly_scores(X_test)

        if len(scores) != len(y_test):
            raise ValueError("X_test ve y_test uzunlukları eşit olmalı.")

        y_pred = (scores > self.threshold_).astype(int)
        report = classification_report(
            y_test, y_pred, labels=[0, 1],
            output_dict=True, zero_division=0,
        )
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        if len(np.unique(y_test)) == 2:
            roc_auc = float(roc_auc_score(y_test, scores))
            avg_precision = float(
                average_precision_score(y_test, scores)
            )
        else:
            roc_auc = float("nan")
            avg_precision = float("nan")

        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        positive = report.get("1", {})

        results = {
            "model": "Hybrid (Isolation Forest + Autoencoder)",
            "precision": positive.get("precision", 0.0),
            "recall": positive.get("recall", 0.0),
            "f1_score": positive.get("f1-score", 0.0),
            "accuracy": report.get("accuracy", 0.0),
            "roc_auc": roc_auc,
            "avg_precision": avg_precision,
            "false_positive_rate": fpr,
            "true_positive_rate": tpr,
            "confusion_matrix": cm,
            "y_pred": y_pred,
            "y_scores": scores,
            "threshold": self.threshold_,
        }

        self._print_results(results, cm)
        return results

    @staticmethod
    def _print_results(results, cm):
        print("\n" + "=" * 60)
        print("  HYBRID MODEL - SONUÇLAR")
        print("=" * 60)
        print(f"  Threshold:           {results['threshold']:.6f}")
        print(f"  Precision:           {results['precision']:.4f}")
        print(f"  Recall:              {results['recall']:.4f}")
        print(f"  F1-Score:            {results['f1_score']:.4f}")
        print(f"  Accuracy:            {results['accuracy']:.4f}")
        print(f"  ROC-AUC:             {results['roc_auc']:.4f}")
        print(f"  Average Precision:   {results['avg_precision']:.4f}")
        print(f"  False Positive Rate: {results['false_positive_rate']:.4f}")
        print("\n  Confusion Matrix:")
        print(f"  TN={cm[0, 0]:5d}  FP={cm[0, 1]:5d}")
        print(f"  FN={cm[1, 0]:5d}  TP={cm[1, 1]:5d}")
        print("=" * 60 + "\n")

    def save(self, path="results/hybrid_model.pkl"):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        joblib.dump(self, path)
        print(f"[✓] Hibrit model kaydedildi: {path}")

    @classmethod
    def load(cls, path):
        return joblib.load(path)
