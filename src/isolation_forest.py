"""
isolation_forest.py
-------------------
Isolation Forest tabanlı anomali tespit modeli.

Isolation Forest, anomalileri "izole etme" fikrine dayanır:
Anomaliler, normal gözlemlerden daha az bölme gerektirir.
Özellikle etiketlenmemiş/yarı-etiketli veri için güçlüdür.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, precision_recall_curve, average_precision_score
)
import joblib
import os


class IsolationForestDetector:
    """
    Isolation Forest tabanlı İç Tehdit Tespit Modeli.

    Parametreler:
        contamination: Beklenen anomali oranı (0.0 - 0.5)
        n_estimators:  Ağaç sayısı
        max_samples:   Her ağaç için örnekleme sayısı
        random_state:  Üretilebilirlik için sabit tohum
    """

    def __init__(
        self,
        contamination: float = 0.02,
        n_estimators: int = 200,
        max_samples: str = "auto",
        random_state: int = 42
    ):
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1
        )
        self.threshold_ = None

    def fit(self, X_train: np.ndarray) -> "IsolationForestDetector":
        """Modeli YALNIZCA normal veri üzerinde eğit (unsupervised)."""
        # Sadece normal kayıtlar üzerinde eğit
        print(f"[*] Isolation Forest eğitiliyor... ({self.n_estimators} ağaç, contamination={self.contamination})")
        self.model.fit(X_train)
        print("[✓] Isolation Forest eğitimi tamamlandı.")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        -1 → Anomali, +1 → Normal (sklearn standardı)
        Dönen değer: 0=Normal, 1=Anomali (proje standardı)
        """
        raw = self.model.predict(X)
        return (raw == -1).astype(int)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Anomali skoru döndürür.
        decision_function: Daha DÜŞÜK = daha anomali
        Tersine çevirerek: Daha YÜKSEK = daha anomali (ROC-AUC için)
        """
        return -self.model.decision_function(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Test seti üzerinde tam değerlendirme raporu üretir."""
        y_pred = self.predict(X_test)
        scores = self.anomaly_scores(X_test)

        # Metrikler
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, scores)
        avg_precision = average_precision_score(y_test, scores)

        # False Positive Rate
        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

        results = {
            "model": "Isolation Forest",
            "precision": report.get("1", {}).get("precision", 0),
            "recall": report.get("1", {}).get("recall", 0),
            "f1_score": report.get("1", {}).get("f1-score", 0),
            "accuracy": report.get("accuracy", 0),
            "roc_auc": roc_auc,
            "avg_precision": avg_precision,
            "false_positive_rate": fpr,
            "true_positive_rate": tpr,
            "confusion_matrix": cm,
            "y_pred": y_pred,
            "y_scores": scores,
        }

        self._print_results(results, cm)
        return results

    def _print_results(self, results: dict, cm: np.ndarray):
        print("\n" + "="*60)
        print("  ISOLATION FOREST - SONUÇLAR")
        print("="*60)
        print(f"  Precision:          {results['precision']:.4f}")
        print(f"  Recall:             {results['recall']:.4f}")
        print(f"  F1-Score:           {results['f1_score']:.4f}")
        print(f"  ROC-AUC:            {results['roc_auc']:.4f}")
        print(f"  Avg Precision:      {results['avg_precision']:.4f}")
        print(f"  False Positive Rate:{results['false_positive_rate']:.4f}")
        print(f"\n  Confusion Matrix:")
        print(f"  TN={cm[0,0]:5d}  FP={cm[0,1]:5d}")
        print(f"  FN={cm[1,0]:5d}  TP={cm[1,1]:5d}")
        print("="*60 + "\n")

    def save(self, path: str = "results/isolation_forest_model.pkl"):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        joblib.dump(self.model, path)
        print(f"[✓] Model kaydedildi: {path}")

    @classmethod
    def load(cls, path: str) -> "IsolationForestDetector":
        detector = cls()
        detector.model = joblib.load(path)
        return detector
