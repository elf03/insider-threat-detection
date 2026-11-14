"""
Autoencoder tabanlı anomali tespit modeli.
Saf NumPy implementasyonu; TensorFlow/PyTorch gerektirmez.

Model yalnızca normal eğitim kayıtlarıyla öğrenir. Anomali skoru,
örnek başına ortalama kare yeniden yapılandırma hatasıdır.
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


def _relu(x):
    return np.maximum(0, x)


def _relu_deriv(x):
    return (x > 0).astype(float)


class AutoencoderDetector:
    """Derin, tamamen NumPy ile yazılmış Autoencoder."""

    def __init__(
        self,
        input_dim=13,
        encoding_dim=16,
        epochs=50,
        batch_size=256,
        learning_rate=1e-3,
        threshold_percentile=95.0,
    ):
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.epochs = epochs
        self.batch_size = batch_size
        self.lr = learning_rate
        self.threshold_percentile = threshold_percentile
        self.threshold_ = None
        self.history_ = {"loss": [], "val_loss": []}

        # Ağırlıkların tekrarlanabilir başlatılması.
        self._rng = np.random.default_rng(42)
        self._init_weights()

    def _init_weights(self):
        """He initialization ile ağırlıkları başlat."""
        d = self.input_dim
        e = self.encoding_dim

        def he(fan_in, fan_out):
            return self._rng.standard_normal((fan_in, fan_out)) * np.sqrt(
                2.0 / fan_in
            )

        # Encoder: input -> 64 -> 32 -> latent
        self.W1 = he(d, 64)
        self.b1 = np.zeros((1, 64))
        self.W2 = he(64, 32)
        self.b2 = np.zeros((1, 32))
        self.W3 = he(32, e)
        self.b3 = np.zeros((1, e))

        # Decoder: latent -> 32 -> 64 -> input
        self.W4 = he(e, 32)
        self.b4 = np.zeros((1, 32))
        self.W5 = he(32, 64)
        self.b5 = np.zeros((1, 64))
        self.W6 = he(64, d)
        self.b6 = np.zeros((1, d))

    def _forward(self, X):
        """İleri besleme; aktivasyonları ve çıktıyı döndür."""
        a1 = _relu(X @ self.W1 + self.b1)
        a2 = _relu(a1 @ self.W2 + self.b2)
        a3 = _relu(a2 @ self.W3 + self.b3)
        a4 = _relu(a3 @ self.W4 + self.b4)
        a5 = _relu(a4 @ self.W5 + self.b5)
        out = a5 @ self.W6 + self.b6
        return a1, a2, a3, a4, a5, out

    def _backward(self, X, a1, a2, a3, a4, a5, out):
        """MSE kaybı için geri yayılım ve SGD ağırlık güncellemesi."""
        n = X.shape[0]
        delta = (2.0 / (n * self.input_dim)) * (out - X)

        dW6 = a5.T @ delta
        db6 = delta.sum(axis=0, keepdims=True)

        d5 = (delta @ self.W6.T) * _relu_deriv(a5)
        dW5 = a4.T @ d5
        db5 = d5.sum(axis=0, keepdims=True)

        d4 = (d5 @ self.W5.T) * _relu_deriv(a4)
        dW4 = a3.T @ d4
        db4 = d4.sum(axis=0, keepdims=True)

        d3 = (d4 @ self.W4.T) * _relu_deriv(a3)
        dW3 = a2.T @ d3
        db3 = d3.sum(axis=0, keepdims=True)

        d2 = (d3 @ self.W3.T) * _relu_deriv(a2)
        dW2 = a1.T @ d2
        db2 = d2.sum(axis=0, keepdims=True)

        d1 = (d2 @ self.W2.T) * _relu_deriv(a1)
        dW1 = X.T @ d1
        db1 = d1.sum(axis=0, keepdims=True)

        for name, grad in (
            ("W1", dW1), ("b1", db1),
            ("W2", dW2), ("b2", db2),
            ("W3", dW3), ("b3", db3),
            ("W4", dW4), ("b4", db4),
            ("W5", dW5), ("b5", db5),
            ("W6", dW6), ("b6", db6),
        ):
            setattr(self, name, getattr(self, name) - self.lr * grad)

    def _get_weights(self):
        names = (
            "W1", "b1", "W2", "b2", "W3", "b3",
            "W4", "b4", "W5", "b5", "W6", "b6",
        )
        return [getattr(self, name).copy() for name in names]

    def _set_weights(self, weights):
        names = (
            "W1", "b1", "W2", "b2", "W3", "b3",
            "W4", "b4", "W5", "b5", "W6", "b6",
        )
        for name, value in zip(names, weights):
            setattr(self, name, value.copy())

    def fit(self, X_train, y_train=None, X_val=None, y_val=None):
        """
        Modeli yalnızca normal eğitim örnekleriyle eğitir.
        Validation verisini early stopping ve eşik kalibrasyonu için kullanır.
        """
        X_train = np.asarray(X_train, dtype=float)
        if X_train.ndim != 2 or X_train.shape[1] != self.input_dim:
            raise ValueError(
                f"X_train şekli (n, {self.input_dim}) olmalı; gelen: {X_train.shape}"
            )

        if y_train is not None:
            y_train = np.asarray(y_train)
            X_normal = X_train[y_train == 0]
        else:
            X_normal = X_train

        if len(X_normal) < 2:
            raise ValueError("Autoencoder eğitimi için yeterli normal kayıt yok.")

        if X_val is None:
            # Geriye dönük uyumluluk: normal train verisinin bir kısmını ayır.
            indices = self._rng.permutation(len(X_normal))
            val_size = max(1, int(len(X_normal) * 0.1))
            X_val_normal = X_normal[indices[:val_size]]
            X_tr = X_normal[indices[val_size:]]
        else:
            X_val = np.asarray(X_val, dtype=float)
            if X_val.ndim != 2 or X_val.shape[1] != self.input_dim:
                raise ValueError(
                    f"X_val şekli (n, {self.input_dim}) olmalı; gelen: {X_val.shape}"
                )
            if y_val is not None:
                y_val = np.asarray(y_val)
                X_val_normal = X_val[y_val == 0]
            else:
                X_val_normal = X_val
            X_tr = X_normal

        if len(X_tr) == 0:
            raise ValueError("Eğitim için normal kayıt bulunamadı.")
        if len(X_val_normal) == 0:
            raise ValueError("Doğrulama kümesinde normal kayıt bulunamadı.")

        self.history_ = {"loss": [], "val_loss": []}
        print(
            f"\n[*] Autoencoder eğitiliyor... "
            f"({len(X_tr):,} train, {len(X_val_normal):,} val, "
            f"{self.epochs} epoch)"
        )

        best_val = float("inf")
        best_weights = None
        patience_count = 0
        patience = 8

        for epoch in range(1, self.epochs + 1):
            indices = self._rng.permutation(len(X_tr))
            X_shuffled = X_tr[indices]
            epoch_loss = 0.0
            batches = 0

            for start in range(0, len(X_tr), self.batch_size):
                X_batch = X_shuffled[start:start + self.batch_size]
                a1, a2, a3, a4, a5, out = self._forward(X_batch)
                batch_loss = np.mean((out - X_batch) ** 2)
                self._backward(X_batch, a1, a2, a3, a4, a5, out)
                epoch_loss += batch_loss
                batches += 1

            train_mse = epoch_loss / max(batches, 1)
            _, _, _, _, _, val_out = self._forward(X_val_normal)
            val_mse = float(np.mean((val_out - X_val_normal) ** 2))

            self.history_["loss"].append(float(train_mse))
            self.history_["val_loss"].append(val_mse)

            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"  Epoch {epoch:3d}/{self.epochs} — "
                    f"loss: {train_mse:.6f}  val_loss: {val_mse:.6f}"
                )

            if val_mse < best_val - 1e-6:
                best_val = val_mse
                best_weights = self._get_weights()
                patience_count = 0
            else:
                patience_count += 1
                if patience_count >= patience:
                    print(f"  [EarlyStopping] Epoch {epoch}'de duruldu.")
                    break

        if best_weights is not None:
            self._set_weights(best_weights)

        # Eşik yalnızca normal validation kayıtlarından hesaplanır.
        normal_errors = self.reconstruction_error(X_val_normal)
        self.threshold_ = float(
            np.percentile(normal_errors, self.threshold_percentile)
        )

        print("[✓] Autoencoder eğitimi tamamlandı.")
        print(
            f"[✓] Anomali eşiği (MSE): {self.threshold_:.6f} "
            f"({self.threshold_percentile}. persentil)"
        )
        return self

    def reconstruction_error(self, X):
        """Her örnek için ortalama kare yeniden yapılandırma hatası."""
        X = np.asarray(X, dtype=float)
        _, _, _, _, _, out = self._forward(X)
        return np.mean((X - out) ** 2, axis=1)

    def predict(self, X):
        """1=Anomali, 0=Normal."""
        if self.threshold_ is None:
            raise RuntimeError("Model henüz eğitilmedi; önce fit() çağırın.")
        return (self.reconstruction_error(X) > self.threshold_).astype(int)

    def anomaly_scores(self, X):
        return self.reconstruction_error(X)

    def evaluate(self, X_test, y_test):
        y_test = np.asarray(y_test).astype(int)
        y_pred = self.predict(X_test)
        scores = self.anomaly_scores(X_test)

        report = classification_report(
            y_test, y_pred, output_dict=True, zero_division=0
        )
        cm = confusion_matrix(y_test, y_pred, labels=[0, 1])

        roc_auc = (
            roc_auc_score(y_test, scores)
            if len(np.unique(y_test)) == 2 else float("nan")
        )
        avg_precision = (
            average_precision_score(y_test, scores)
            if len(np.unique(y_test)) == 2 else float("nan")
        )

        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        tpr = tp / (tp + fn) if (tp + fn) else 0.0
        positive = report.get("1", {})

        results = {
            "model": "Autoencoder",
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

    def _print_results(self, results, cm):
        print("\n" + "=" * 60)
        print("  AUTOENCODER (NumPy) - SONUÇLAR")
        print("=" * 60)
        print(f"  Reconstruction Threshold: {results['threshold']:.6f}")
        print(f"  Precision:           {results['precision']:.4f}")
        print(f"  Recall:              {results['recall']:.4f}")
        print(f"  F1-Score:            {results['f1_score']:.4f}")
        print(f"  Accuracy:            {results['accuracy']:.4f}")
        print(f"  ROC-AUC:             {results['roc_auc']:.4f}")
        print(f"  Avg Precision:       {results['avg_precision']:.4f}")
        print(f"  False Positive Rate: {results['false_positive_rate']:.4f}")
        print("\n  Confusion Matrix:")
        print(f"  TN={cm[0, 0]:5d}  FP={cm[0, 1]:5d}")
        print(f"  FN={cm[1, 0]:5d}  TP={cm[1, 1]:5d}")
        print("=" * 60 + "\n")

    def save(self, path="results/autoencoder_model.pkl"):
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        if path.endswith(".keras"):
            path = path[:-6] + ".pkl"
        joblib.dump(self, path)
        print(f"[✓] Autoencoder modeli kaydedildi: {path}")

    @classmethod
    def load(cls, path):
        return joblib.load(path)
