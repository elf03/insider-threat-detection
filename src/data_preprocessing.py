"""
data_preprocessing.py
----------------------
Kullanıcı davranış verisini model eğitimine hazırlar.
Özellik mühendisliği, normalizasyon ve train/test bölme işlemlerini içerir.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, List
import joblib
import os


FEATURE_COLUMNS = [
    "data_volume_mb",
    "hour_of_day",
    "is_working_hours",
    "old_activity_score",
    "new_activity_score",
    "failed_logins",
    "unique_resources",
    "activity_type_encoded",
    # Türetilmiş özellikler
    "volume_zscore",
    "off_hours_volume",
    "login_failure_rate",
    "resource_intensity",
    "balance_drop_ratio",
]


def load_data(filepath: str = "data/user_behavior_logs.csv") -> pd.DataFrame:
    """CSV veri setini yükler."""
    df = pd.read_csv(filepath, parse_dates=["timestamp"])
    print(f"[✓] Veri yüklendi: {len(df):,} satır, {len(df.columns)} sütun")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Özellik mühendisliği:
    - Kullanıcı başına istatistiksel z-score
    - Mesai dışı yoğun transfer oranı
    - Kaynak erişim yoğunluğu
    """
    df = df.copy()

    # Aktivite türü sayısal kodlama
    le = LabelEncoder()
    df["activity_type_encoded"] = le.fit_transform(df["activity_type"])

    # Kullanıcı bazında z-score (normal profilden sapma)
    user_stats = df.groupby("user_id")["data_volume_mb"].agg(["mean", "std"]).reset_index()
    user_stats.columns = ["user_id", "user_mean_vol", "user_std_vol"]
    df = df.merge(user_stats, on="user_id", how="left")
    df["user_std_vol"] = df["user_std_vol"].fillna(1.0).replace(0, 1.0)
    df["volume_zscore"] = (df["data_volume_mb"] - df["user_mean_vol"]) / df["user_std_vol"]

    # Mesai dışı büyük transfer
    df["off_hours_volume"] = df["data_volume_mb"] * (1 - df["is_working_hours"])

    # Başarısız giriş oranı (pencereleme)
    df["login_failure_rate"] = df["failed_logins"] / (df["unique_resources"] + 1)

    # Kaynak erişim yoğunluğu
    df["resource_intensity"] = df["unique_resources"] / (df["data_volume_mb"] + 1)

    # Bakiye düşme oranı (aktivite skoru değişimi)
    df["balance_drop_ratio"] = (
        (df["old_activity_score"] - df["new_activity_score"]) /
        (df["old_activity_score"] + 1e-6)
    )

    print(f"[✓] Özellik mühendisliği tamamlandı: {len(FEATURE_COLUMNS)} özellik")
    return df


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    scaler_path: str = "results/scaler.pkl"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Veriyi normalize eder ve train/test olarak böler.

    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    df = engineer_features(df)

    # NaN temizliği
    df[FEATURE_COLUMNS] = df[FEATURE_COLUMNS].fillna(0)

    X = df[FEATURE_COLUMNS].values
    y = df["is_anomaly"].values

    # Stratified split (anomali dengesini koru)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # StandardScaler - sadece train üzerinde fit
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Scaler'ı kaydet
    os.makedirs(os.path.dirname(scaler_path) if os.path.dirname(scaler_path) else ".", exist_ok=True)
    joblib.dump(scaler, scaler_path)

    print(f"[✓] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[✓] Train anomali oranı: {100*y_train.mean():.2f}%")
    print(f"[✓] Test anomali oranı: {100*y_test.mean():.2f}%")
    return X_train, X_test, y_train, y_test, FEATURE_COLUMNS


def get_user_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Her kullanıcı için davranış profili özeti üretir."""
    df = engineer_features(df)
    profiles = df.groupby("user_id").agg(
        total_records=("step", "count"),
        mean_volume=("data_volume_mb", "mean"),
        max_volume=("data_volume_mb", "max"),
        mean_failed_logins=("failed_logins", "mean"),
        off_hours_ratio=("is_working_hours", lambda x: 1 - x.mean()),
        anomaly_count=("is_anomaly", "sum"),
        mean_zscore=("volume_zscore", lambda x: abs(x).mean()),
    ).reset_index()
    profiles["anomaly_rate"] = profiles["anomaly_count"] / profiles["total_records"]
    return profiles.sort_values("anomaly_rate", ascending=False)
