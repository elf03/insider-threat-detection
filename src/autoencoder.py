"""
autoencoder.py
--------------
Autoencoder tabanlı anomali tespit modeli (TensorFlow/Keras).

Çalışma prensibi:
  1. Model YALNIZCA normal verilerle eğitilir
  2. Normal veriler düşük reconstruction error üretir
  3. Anomaliler yüksek reconstruction error üretir (normal profile uymuyor)
  4. Eşik değer (threshold) üzerindeki kayıtlar anomali olarak işaretlenir
"""

import numpy as np
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # TF uyarılarını gizle

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, average_precision_score
)


class AutoencoderDetector:
    """
    Derin Autoencoder tabanlı İç Tehdit Tespit Modeli.

    Mimari:
        Encoder: input_dim → 64 → 32 → 16 (latent)
        Decoder: 16 → 32 → 64 → input_dim
    """

    def __init__(
        self,
        input_dim: int = 13,
        encoding_dim: int = 16,
        epochs: int = 50,
        batch_size: int = 256,
        learning_rate: float = 1e-3,
        threshold_percentile: float = 95.0
    ):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.threshold_percentile = threshold_percentile
        self.threshold_ = None
        self.history_ = None
        self.model = self._build_model()

    def _build_model(self) -> Model:
        """Encoder-Decoder mimarisini inşa eder."""
        tf.random.set_seed(42)

        # --- ENCODER ---
        inp = keras.Input(shape=(self.input_dim,), name="input")
        x = layers.Dense(64, activation="relu", name="enc_1")(inp)
        x = layers.BatchNormalization(name="bn_1")(x)
        x = layers.Dropout(0.2, name="drop_1")(x)
        x = layers.Dense(32, activation="relu", name="enc_2")(x)
        x = layers.BatchNormalization(name="bn_2")(x)
        x = layers.Dropout(0.1, name="drop_2")(x)
        latent = layers.Dense(self.encoding_dim, activation="relu", name="latent")(x)

        # --- DECODER ---
        x = layers.Dense(32, activation="relu", name="dec_1")(latent)
        x = layers.BatchNormalization(name="bn_3")(x)
        x = layers.Dense(64, activation="relu", name="dec_2")(x)
        x = layers.BatchNormalization(name="bn_4")(x)
        output = layers.Dense(self.input_dim, activation="linear", name="output")(x)

        model = Model(inputs=inp, outputs=output, name="Autoencoder")
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="mse",
            metrics=["mae"]
        )
        return model

    def fit(self, X_train: np.ndarray, y_train: np.ndarray = None) -> "AutoencoderDetector":
        """
        Modeli YALNIZCA normal örnekler üzerinde eğit.
        y_train varsa normal örnekleri filtrele; yoksa tüm veriyi kullan.
        """
        if y_train is not None:
            X_normal = X_train[y_train == 0]
        else:
            X_normal = X_train

        print(f"\n[*] Autoencoder eğitiliyor... ({len(X_normal):,} normal örnek, {self.epochs} epoch)")

        callbacks = [
            keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=8, restore_best_weights=True
            ),
            keras.callbacks.ReduceLROnPlateau(
                monitor="val_loss", factor=0.5, patience=4, verbose=0
            )
        ]

        self.history_ = self.model.fit(
            X_normal, X_normal,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            callbacks=callbacks,
            verbose=1,
            shuffle=True
        )

        # Eşik değeri: Normal verinin %threshold_percentile'lik reconstruction error
        normal_errors = self.reconstruction_error(X_normal)
        self.threshold_ = np.percentile(normal_errors, self.threshold_percentile)
        print(f"[✓] Autoencoder eğitimi tamamlandı.")
        print(f"[✓] Anomali eşiği (MSE): {self.threshold_:.6f} ({self.threshold_percentile}. persentil)")
        return self

    def reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """Her örnek için ortalama kare yeniden yapılandırma hatası döndürür."""
        X_pred = self.model.predict(X, verbose=0)
        return np.mean(np.power(X - X_pred, 2), axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Eşik değerine göre anomali tahmini: 1=Anomali, 0=Normal"""
        errors = self.reconstruction_error(X)
        return (errors > self.threshold_).astype(int)

    def anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """Anomali skoru olarak reconstruction error döndürür."""
        return self.reconstruction_error(X)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        """Test seti üzerinde tam değerlendirme raporu üretir."""
        y_pred = self.predict(X_test)
        scores = self.anomaly_scores(X_test)

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, scores)
        avg_precision = average_precision_score(y_test, scores)

        tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0

        results = {
            "model": "Autoencoder",
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
            "threshold": self.threshold_,
        }

        self._print_results(results, cm)
        return results

    def _print_results(self, results: dict, cm: np.ndarray):
        print("\n" + "="*60)
        print("  AUTOENCODER - SONUÇLAR")
        print("="*60)
        print(f"  Reconstruction Threshold: {results['threshold']:.6f}")
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

    def save(self, path: str = "results/autoencoder_model.keras"):
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        self.model.save(path)
        print(f"[✓] Autoencoder modeli kaydedildi: {path}")

    @classmethod
    def load(cls, path: str, threshold: float = 0.5) -> "AutoencoderDetector":
        detector = cls()
        detector.model = keras.models.load_model(path)
        detector.threshold_ = threshold
        return detector
