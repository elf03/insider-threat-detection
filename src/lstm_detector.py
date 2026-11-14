"""NumPy tabanli, gercek LSTM ile kullanici davranisi anomali tespiti.

Her ornek: ayni kullanicinin onceki `sequence_length` saatindeki 13 ozellik.
Hedef: bir sonraki saatin 13 ozelligini tahmin etmek. Yuksek tahmin hatasi
anomali skorudur. Model yalnizca normal egitim hedefleriyle egitilir.
"""
from __future__ import annotations

import os
import joblib
import numpy as np
from sklearn.metrics import (
    precision_score, recall_score, f1_score, accuracy_score,
    roc_auc_score, average_precision_score, confusion_matrix,
)


def build_user_sequences(df, sequence_length=24, scaler_path="results/scaler.pkl"):
    """Mevcut preprocessing mantigini tekrar kullanarak kullanici dizileri kurar.

    preprocess() ile ayni kronolojik %70/%10/%20 bolme ve egitimde fit edilen
    ozellik istatistikleri/scaler kullanilir. Bolmeler arasinda gecmis baglami
    korunur; hedef kayitlar kendi bolmelerinde kalir.
    """
    from src.data_preprocessing import (
        _validate_columns, _fit_user_statistics, _fit_activity_mapping,
        _transform_features, FEATURE_COLUMNS,
    )

    if sequence_length < 2:
        raise ValueError("sequence_length en az 2 olmali.")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler bulunamadi: {scaler_path}. Once mevcut main.py calismali.")

    data = df.copy()
    _validate_columns(data)
    data["timestamp"] = __import__("pandas").to_datetime(data["timestamp"], errors="coerce")
    if data["timestamp"].isna().any():
        raise ValueError("Gecersiz timestamp kayitlari var.")
    data["is_anomaly"] = data["is_anomaly"].astype(int)
    data = data.sort_values(["timestamp", "user_id"], kind="mergesort").reset_index(drop=True)

    n = len(data)
    train_end = int(n * 0.70)
    val_end = int(n * 0.80)
    train_df = data.iloc[:train_end].copy()
    val_df = data.iloc[train_end:val_end].copy()
    test_df = data.iloc[val_end:].copy()

    user_stats, global_mean, global_std = _fit_user_statistics(train_df)
    activity_mapping = _fit_activity_mapping(train_df)
    transformed = _transform_features(
        data, user_stats, global_mean, global_std, activity_mapping
    )
    scaler = joblib.load(scaler_path)
    features = scaler.transform(transformed[FEATURE_COLUMNS].to_numpy(dtype=np.float64)).astype(np.float32)
    labels = data["is_anomaly"].to_numpy(dtype=np.int8)

    # Sinirlar hedef satir indeksidir. Her kullanici icin sadece ayni kullanicinin
    # onceki kayitlari kullanilir; validation/test baglami onceki donemlerden gelebilir.
    train_idx_end, val_idx_end = train_end, val_end
    user_positions = {
        user: group.index.to_numpy()
        for user, group in data.groupby("user_id", sort=False)
    }
    splits = {"train": (0, train_idx_end), "val": (train_idx_end, val_idx_end), "test": (val_idx_end, n)}
    outputs = {}
    metadata = {}

    for split_name, (start, end) in splits.items():
        seqs, ys, records = [], [], []
        for positions in user_positions.values():
            # Bu kullanicinin hedef satirlari ilgili bolumde olan her kaydi gez.
            local = positions[(positions >= start) & (positions < end)]
            for target_idx in local:
                loc = np.searchsorted(positions, target_idx)
                if loc < sequence_length:
                    continue
                history_idx = positions[loc-sequence_length:loc]
                if len(history_idx) != sequence_length:
                    continue

                seqs.append(
                    np.concatenate(
                        [
                            features[history_idx],
                            features[target_idx][None, :]
                        ],
                        axis=0
                    )
                )
                ys.append(labels[target_idx])

                target_row = data.iloc[target_idx]

                records.append({
                    "user_id": str(target_row["user_id"]),
                    "timestamp": target_row["timestamp"].isoformat(),
                    "is_anomaly": int(target_row["is_anomaly"]),
                    "data_volume_mb": float(target_row["data_volume_mb"]),
                    "hour_of_day": int(target_row["hour_of_day"]),
                    "failed_logins": int(target_row["failed_logins"]),
                    "unique_resources": int(target_row["unique_resources"])
                })
        if not seqs:
            raise ValueError(f"{split_name} icin LSTM dizisi olusturulamadi.")

        outputs[split_name] = (
            np.stack(seqs),
            np.asarray(ys, dtype=np.int8)
        )

        metadata[split_name] = records

    return (
        *outputs["train"],
        *outputs["val"],
        *outputs["test"],
        metadata
    )


class LSTMAnomalyDetector:
    """Kucuk boyutlu, NumPy ile egitilen tek katmanli LSTM tahmin modeli."""
    def __init__(self, input_dim, hidden_dim=12, epochs=8, batch_size=128,
                 learning_rate=0.001, threshold_percentile=95.0, seed=42,
                 gradient_clip=1.0):
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.threshold_percentile = float(threshold_percentile)
        self.seed = int(seed)
        self.gradient_clip = float(gradient_clip)
        self.params = None
        self.threshold_ = None
        self.history_ = []
        self.rng = np.random.default_rng(self.seed)

    @staticmethod
    def _sigmoid(x):
        x = np.clip(x, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-x))

    def _initialize(self):
        h, d = self.hidden_dim, self.input_dim
        scale = np.sqrt(2.0 / (d + h))
        self.params = {
            "W": self.rng.normal(0, scale, (d+h, 4*h)).astype(np.float32),
            "b": np.zeros(4*h, dtype=np.float32),
            "Why": self.rng.normal(0, np.sqrt(2.0/(h+d)), (h, d)).astype(np.float32),
            "by": np.zeros(d, dtype=np.float32),
        }
        # Forget gate biasini pozitif baslat.
        self.params["b"][h:2*h] = 1.0

    def _forward(self, x, cache=False):
        p = self.params
        batch, steps, _ = x.shape
        hdim = self.hidden_dim
        h = np.zeros((batch, hdim), dtype=np.float32)
        c = np.zeros_like(h)
        caches = []
        for t in range(steps):
            xt = x[:, t, :]
            z = np.concatenate([xt, h], axis=1) @ p["W"] + p["b"]
            zi, zf, zo, zg = np.split(z, 4, axis=1)
            i, f, o = self._sigmoid(zi), self._sigmoid(zf), self._sigmoid(zo)
            g = np.tanh(zg)
            c_prev = c
            c = f*c + i*g
            tanhc = np.tanh(c)
            h_prev = h
            h = o*tanhc
            if cache:
                caches.append((xt, h_prev, c_prev, i, f, o, g, c, tanhc))
        y = h @ p["Why"] + p["by"]
        return (y, caches) if cache else y

    def _batch_gradients(self, x, target):
        p = self.params
        pred, caches = self._forward(x, cache=True)
        batch = x.shape[0]
        dy = (2.0 / (batch*self.input_dim)) * (pred-target)
        grads = {k: np.zeros_like(v) for k, v in p.items()}
        last_hidden = caches[-1][5] * caches[-1][8]
        grads["Why"] = last_hidden.T @ dy
        grads["by"] = dy.sum(axis=0)
        dh = dy @ p["Why"].T
        dc_next = np.zeros((batch, self.hidden_dim), dtype=np.float32)

        for xt, h_prev, c_prev, i, f, o, g, c, tanhc in reversed(caches):
            do = dh*tanhc
            dc = dh*o*(1.0-tanhc*tanhc) + dc_next
            df = dc*c_prev
            di = dc*g
            dg = dc*i
            dc_next = dc*f
            dzi = di*i*(1.0-i)
            dzf = df*f*(1.0-f)
            dzo = do*o*(1.0-o)
            dzg = dg*(1.0-g*g)
            dz = np.concatenate([dzi, dzf, dzo, dzg], axis=1)
            joined = np.concatenate([xt, h_prev], axis=1)
            grads["W"] += joined.T @ dz
            grads["b"] += dz.sum(axis=0)
            dh = dz @ p["W"].T[:, self.input_dim:]

        for key in grads:
            np.clip(grads[key], -self.gradient_clip, self.gradient_clip, out=grads[key])
        return float(np.mean((pred-target)**2)), grads

    def _predict_batches(self, X):
        chunks = []
        for start in range(0, len(X), self.batch_size):
            chunks.append(self._forward(X[start:start+self.batch_size]))
        return np.concatenate(chunks, axis=0)

    def anomaly_scores(self, X):
        X = np.asarray(X, dtype=np.float32)
        pred = self._predict_batches(X[:, :-1, :])
        return np.mean((pred-X[:, -1, :])**2, axis=1)

    def fit(self, X_train, y_train, X_val, y_val):
        X_train = np.asarray(X_train, dtype=np.float32)
        X_val = np.asarray(X_val, dtype=np.float32)
        y_train = np.asarray(y_train, dtype=np.int8)
        y_val = np.asarray(y_val, dtype=np.int8)
        if X_train.ndim != 3 or X_train.shape[2] != self.input_dim:
            raise ValueError("X_train (ornek, zaman, ozellik) biciminde olmali.")
        normal = np.flatnonzero(y_train == 0)
        val_normal = np.flatnonzero(y_val == 0)
        if len(normal) == 0 or len(val_normal) == 0:
            raise ValueError("LSTM egitimi ve kalibrasyonu icin normal kayit gerekli.")
        Xn = X_train[normal]
        target = Xn[:, -1, :].copy()
        Xn_history = Xn[:, :-1, :]
        self._initialize()
        best_loss, best_params, patience = np.inf, None, 0

        for epoch in range(self.epochs):
            order = self.rng.permutation(len(Xn_history))
            losses = []
            for start in range(0, len(order), self.batch_size):
                idx = order[start:start+self.batch_size]
                xb, yb = Xn_history[idx], target[idx]
                loss, grads = self._batch_gradients(xb, yb)
                for key in self.params:
                    self.params[key] -= self.learning_rate*grads[key]
                losses.append(loss)
            val_scores = self.anomaly_scores(X_val[val_normal])
            val_loss = float(np.mean(val_scores))
            self.history_.append({"epoch": epoch+1, "loss": float(np.mean(losses)), "val_loss": val_loss})
            print(f"  Epoch {epoch+1:02d}/{self.epochs} â€” loss: {np.mean(losses):.6f}  val_loss: {val_loss:.6f}")
            if val_loss < best_loss - 1e-7:
                best_loss, best_params, patience = val_loss, {k:v.copy() for k,v in self.params.items()}, 0
            else:
                patience += 1
                if patience >= 3:
                    print("  [*] Early stopping: validation kaybi iyilesmedi.")
                    break
        if best_params is not None:
            self.params = best_params
        self.threshold_ = float(np.percentile(self.anomaly_scores(X_val[val_normal]), self.threshold_percentile))
        print(f"[âœ“] LSTM anomali esigi kalibre edildi: {self.threshold_:.6f}")
        return self

    def predict(self, X):
        if self.threshold_ is None:
            raise RuntimeError("Once fit() cagrilmali.")
        return (self.anomaly_scores(X) >= self.threshold_).astype(int)

    def evaluate(self, X, y_true):
        y_true = np.asarray(y_true, dtype=np.int8)
        scores = self.anomaly_scores(X)
        pred = (scores >= self.threshold_).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0,1]).ravel()
        return {
            "precision": float(precision_score(y_true,pred,zero_division=0)),
            "recall": float(recall_score(y_true,pred,zero_division=0)),
            "f1_score": float(f1_score(y_true,pred,zero_division=0)),
            "accuracy": float(accuracy_score(y_true,pred)),
            "roc_auc": float(roc_auc_score(y_true,scores)) if len(np.unique(y_true)) > 1 else float("nan"),
            "avg_precision": float(average_precision_score(y_true,scores)) if len(np.unique(y_true)) > 1 else float("nan"),
            "false_positive_rate": float(fp/(fp+tn)) if fp+tn else 0.0,
            "true_positive_rate": float(tp/(tp+fn)) if tp+fn else 0.0,
            "confusion_matrix": np.array([[tn,fp],[fn,tp]], dtype=int),
            "y_pred": pred,
            "y_scores": scores,
            "threshold": float(self.threshold_),
        }

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        joblib.dump({"params": self.params, "input_dim": self.input_dim,
                     "hidden_dim": self.hidden_dim, "threshold": self.threshold_,
                     "sequence_length": getattr(self, "sequence_length", None),
                     "history": self.history_}, path)
        print(f"[âœ“] LSTM modeli kaydedildi: {path}")

    @classmethod
    def load(cls, path):
        payload = joblib.load(path)
        model = cls(payload["input_dim"], hidden_dim=payload["hidden_dim"])
        model.params = payload["params"]
        model.threshold_ = payload["threshold"]
        model.history_ = payload.get("history", [])
        return model
