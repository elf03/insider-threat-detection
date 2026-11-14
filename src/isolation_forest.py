
"""
isolation_forest.py
--------------------
Isolation Forest tabanlı anomali tespit modeli.

Model normal kullanıcı davranışını öğrenir.
Normal davranıştan belirgin şekilde sapan kayıtları anomali olarak işaretler.
"""

import os
import warnings

import joblib
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
)


class IsolationForestDetector:
    """Isolation Forest tabanlı İç Tehdit Tespit Modeli."""

    def __init__(
        self,
        contamination="auto",
        n_estimators: int = 200,
        max_samples="auto",
        random_state: int = 42,
    ):
        if contamination != "auto":
            if not isinstance(contamination, (int, float)):
                raise ValueError("contamination 'auto' veya 0 ile 0.5 arasında olmalı.")
            if not 0 < contamination <= 0.5:
                raise ValueError("contamination 0 ile 0.5 arasında olmalı.")

        self.contamination = contamination
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.random_state = random_state

        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            max_samples=max_samples,
            random_state=random_state,
            n_jobs=-1,
        )

        self.threshold_ = None
        self.is_fitted_ = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray = None,
    ) -> "IsolationForestDetector":
        """
        Modeli yalnızca normal kayıtlar üzerinde eğitir.

        y_train:
            0 = Normal
            1 = Anomali

        Etiket verilmesi zorunludur; aksi hâlde normal kayıtlar
        güvenilir biçimde ayıklanamaz.
        """

        X_train = np.asarray(X_train)

        if y_train is None:
            raise ValueError(
                "y_train gerekli. Modeli yalnızca normal kayıtlarla "
                "eğitmek için fit(X_train, y_train) çağırmalısın."
            )

        y_train = np.asarray(y_train)

        if len(X_train) != len(y_train):
            raise ValueError("X_train ve y_train uzunlukları eşit olmalı.")

        normal_mask = y_train == 0
        X_normal = X_train[normal_mask]

        if len(X_normal) == 0:
            raise ValueError("Eğitim verisinde normal kayıt bulunamadı.")

        if not np.isfinite(X_normal).all():
            raise ValueError("Eğitim verisinde NaN veya sonsuz değer var.")

        print(
            f"[*] Isolation Forest eğitiliyor..."
            f" ({self.n_estimators} ağaç)"
        )
        print(f"[*] Toplam eğitim kaydı: {len(X_train):,}")
        print(f"[*] Kullanılan normal kayıt: {len(X_normal):,}")
        print(f"[*] Dışarıda bırakılan anomali: "
              f"{len(X_train) - len(X_normal):,}")

        self.model.fit(X_normal)
        self.threshold_ = None
        self.is_fitted_ = True

        print("[✓] Isolation Forest eğitimi tamamlandı.")

        return self

    def _check_fitted(self):
        if not self.is_fitted_:
            raise RuntimeError("Önce fit() metoduyla modeli eğitmelisin.")

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Anomali skoru üretir.

        Yüksek skor = daha şüpheli davranış.
        """

        self._check_fitted()

        X = np.asarray(X)
        return -self.model.decision_function(X)

    def calibrate_threshold(
        self,
        X_normal_validation: np.ndarray,
        percentile: float = 95.0,
    ) -> float:
        """
        Normal doğrulama verisiyle anomali eşiğini belirler.

        Örneğin percentile=95:
        Normal doğrulama kayıtlarının yaklaşık %5'inden fazlasını
        eşik üstünde bırakmayacak bir başlangıç eşiği oluşturur.

        Bu değer gerçek anomali başarısını garanti etmez.
        """

        self._check_fitted()

        if not 0 < percentile < 100:
            raise ValueError("percentile 0 ile 100 arasında olmalı.")

        scores = self.anomaly_scores(X_normal_validation)
        self.threshold_ = float(np.percentile(scores, percentile))

        print(
            f"[✓] Anomali eşiği kalibre edildi: "
            f"{self.threshold_:.6f} (percentile={percentile})"
        )

        return self.threshold_

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Dönen değer:
        0 = Normal
        1 = Anomali
        """

        self._check_fitted()

        scores = self.anomaly_scores(X)

        if self.threshold_ is not None:
            return (scores > self.threshold_).astype(int)

        # Kalibrasyon yapılmadıysa sklearn'in auto karar eşiğini kullan.
        return (self.model.predict(X) == -1).astype(int)

    def evaluate(
        self,
        X_test: np.ndarray,
        y_test: np.ndarray,
    ) -> dict:
        """Test seti üzerinde değerlendirme raporu üretir."""

        self._check_fitted()

        y_test = np.asarray(y_test).astype(int)

        y_pred = self.predict(X_test)
        scores = self.anomaly_scores(X_test)

        report = classification_report(
            y_test,
            y_pred,
            output_dict=True,
            zero_division=0,
        )

        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        try:
            roc_auc = float(roc_auc_score(y_test, scores))
        except ValueError:
            roc_auc = float("nan")
            warnings.warn("ROC-AUC hesaplanamadı: test verisinde iki sınıf da bulunmalı.")

        try:
            avg_precision = float(
                average_precision_score(y_test, scores)
            )
        except ValueError:
            avg_precision = float("nan")

        tn, fp, fn, tp = cm.ravel()

        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        results = {
            "model": "Isolation Forest",
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "f1_score": report["1"]["f1-score"],
            "accuracy": report["accuracy"],
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

    def _print_results(self, results: dict, cm: np.ndarray):
        print("\n" + "=" * 60)
        print(" ISOLATION FOREST - SONUÇLAR")
        print("=" * 60)

        print(f" Precision:          {results['precision']:.4f}")
        print(f" Recall:             {results['recall']:.4f}")
        print(f" F1-Score:           {results['f1_score']:.4f}")
        print(f" Accuracy:           {results['accuracy']:.4f}")
        print(f" ROC-AUC:            {results['roc_auc']:.4f}")
        print(f" Average Precision:  {results['avg_precision']:.4f}")
        print(f" False Positive Rate:{results['false_positive_rate']:.4f}")
        print("\n Confusion Matrix:")
        print(f" TN={cm[0, 0]:5d}  FP={cm[0, 1]:5d}")
        print(f" FN={cm[1, 0]:5d}  TP={cm[1, 1]:5d}")
        print("=" * 60 + "\n")

    def save(self, path: str = "results/isolation_forest_model.pkl"):
        """Modeli ve ayarlarını kaydeder."""

        os.makedirs(
            os.path.dirname(path) if os.path.dirname(path) else ".",
            exist_ok=True,
        )

        artifact = {
            "model": self.model,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "max_samples": self.max_samples,
            "random_state": self.random_state,
            "threshold": self.threshold_,
            "is_fitted": self.is_fitted_,
        }

        joblib.dump(artifact, path)
        print(f"[✓] Model kaydedildi: {path}")

    @classmethod
    def load(cls, path: str) -> "IsolationForestDetector":
        """Kaydedilmiş modeli yükler."""

        artifact = joblib.load(path)

        detector = cls()

        # Eski sürümde yalnızca sklearn modeli kaydedilmiş olabilir.
        if isinstance(artifact, dict) and "model" in artifact:
            detector.model = artifact["model"]
            detector.contamination = artifact["contamination"]
            detector.n_estimators = artifact["n_estimators"]
            detector.max_samples = artifact["max_samples"]
            detector.random_state = artifact["random_state"]
            detector.threshold_ = artifact["threshold"]
            detector.is_fitted_ = artifact["is_fitted"]
        else:
            detector.model = artifact
            detector.is_fitted_ = True

        return detector
