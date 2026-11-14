
"""
data_preprocessing.py
---------------------
Kullanıcı davranış verisini model eğitimine hazırlar.
Özellik mühendisliği, kronolojik veri bölme ve normalizasyon işlemlerini içerir.
"""

import os
from typing import Tuple, List, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


FEATURE_COLUMNS = [
    "data_volume_mb",
    "hour_of_day",
    "is_working_hours",
    "old_activity_score",
    "new_activity_score",
    "failed_logins",
    "unique_resources",
    "activity_type_encoded",
    "volume_zscore",
    "off_hours_volume",
    "login_failure_rate",
    "resource_intensity",
    "balance_drop_ratio",
]


REQUIRED_COLUMNS = [
    "user_id",
    "activity_type",
    "data_volume_mb",
    "hour_of_day",
    "is_working_hours",
    "old_activity_score",
    "new_activity_score",
    "failed_logins",
    "unique_resources",
    "is_anomaly",
    "timestamp",
]


def load_data(filepath: str = "data/user_behavior_logs.csv") -> pd.DataFrame:
    """CSV veri setini yükler ve timestamp alanını tarihe dönüştürür."""

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Veri dosyası bulunamadı: {filepath}")

    df = pd.read_csv(filepath)

    if "timestamp" not in df.columns:
        raise ValueError("Veri setinde 'timestamp' sütunu bulunamadı.")

    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    invalid_dates = df["timestamp"].isna().sum()
    if invalid_dates:
        raise ValueError(
            f"{invalid_dates} satırın timestamp değeri geçersiz. "
            "Önce tarih alanlarını düzeltmelisin."
        )

    print(
        f"[✓] Veri yüklendi: {len(df):,} satır, "
        f"{len(df.columns)} sütun"
    )

    return df


def _validate_columns(df: pd.DataFrame) -> None:
    """Model için gerekli sütunların varlığını kontrol eder."""

    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing:
        raise ValueError(
            "Veri setinde eksik sütunlar var: " + ", ".join(missing)
        )


def _fit_user_statistics(
    train_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, float, float]:
    """
    Kullanıcı istatistiklerini yalnızca eğitim verisinden öğrenir.
    Varsa sadece normal kayıtları kullanır.
    """

    normal_df = train_df[train_df["is_anomaly"] == 0]

    # Eğitim verisinde normal kayıt yoksa tüm eğitim kayıtlarını kullan.
    if normal_df.empty:
        normal_df = train_df

    stats = (
        normal_df.groupby("user_id")["data_volume_mb"]
        .agg(["mean", "std"])
        .reset_index()
    )

    stats.columns = ["user_id", "user_mean_vol", "user_std_vol"]

    global_mean = float(normal_df["data_volume_mb"].mean())
    global_std = float(normal_df["data_volume_mb"].std())

    if not np.isfinite(global_mean):
        global_mean = 0.0

    if not np.isfinite(global_std) or global_std == 0:
        global_std = 1.0

    stats["user_mean_vol"] = stats["user_mean_vol"].fillna(global_mean)
    stats["user_std_vol"] = (
        stats["user_std_vol"]
        .replace(0, np.nan)
        .fillna(global_std)
    )

    return stats, global_mean, global_std


def _fit_activity_mapping(train_df: pd.DataFrame) -> Dict[str, int]:
    """Aktivite kodlarını yalnızca eğitim verisinden oluşturur."""

    categories = sorted(
        train_df["activity_type"].fillna("UNKNOWN").astype(str).unique()
    )

    return {category: index for index, category in enumerate(categories)}


def _transform_features(
    df: pd.DataFrame,
    user_stats: pd.DataFrame,
    global_mean: float,
    global_std: float,
    activity_mapping: Dict[str, int],
) -> pd.DataFrame:
    """Öğrenilmiş eğitim istatistiklerini kullanarak özellik üretir."""

    result = df.copy()

    result["activity_type"] = (
        result["activity_type"].fillna("UNKNOWN").astype(str)
    )

    # Eğitimde görülmeyen kategoriler -1 olarak kodlanır.
    result["activity_type_encoded"] = (
        result["activity_type"]
        .map(activity_mapping)
        .fillna(-1)
        .astype(int)
    )

    result = result.merge(user_stats, on="user_id", how="left")

    # Eğitimde bulunmayan kullanıcılar için genel eğitim istatistikleri.
    result["user_mean_vol"] = result["user_mean_vol"].fillna(global_mean)
    result["user_std_vol"] = (
        result["user_std_vol"]
        .replace(0, np.nan)
        .fillna(global_std)
    )

    result["volume_zscore"] = (
        (result["data_volume_mb"] - result["user_mean_vol"])
        / result["user_std_vol"]
    )

    result["off_hours_volume"] = (
        result["data_volume_mb"] * (1 - result["is_working_hours"])
    )

    result["login_failure_rate"] = (
        result["failed_logins"] / (result["unique_resources"] + 1)
    )

    result["resource_intensity"] = (
        result["unique_resources"] / (result["data_volume_mb"] + 1)
    )

    result["balance_drop_ratio"] = (
        (result["old_activity_score"] - result["new_activity_score"])
        / (result["old_activity_score"] + 1e-6)
    )

    # Sayısal olmayan ve sonsuz değerleri güvenli biçimde temizle.
    result[FEATURE_COLUMNS] = result[FEATURE_COLUMNS].apply(
        pd.to_numeric, errors="coerce"
    )

    result[FEATURE_COLUMNS] = (
        result[FEATURE_COLUMNS]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    return result


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Veri setinin tamamı üzerinde betimsel özellik üretir.

    Not: Bu fonksiyon geçmiş verileri incelemek ve profil özeti
    çıkarmak içindir. Model eğitiminde preprocess() kullanılır.
    """

    df = df.copy()

    if "is_anomaly" not in df.columns:
        raise ValueError("'is_anomaly' sütunu bulunamadı.")

    stats, global_mean, global_std = _fit_user_statistics(df)
    mapping = _fit_activity_mapping(df)

    return _transform_features(
        df,
        stats,
        global_mean,
        global_std,
        mapping,
    )


def preprocess(
    df: pd.DataFrame,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    scaler_path: str = "results/scaler.pkl",
) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, List[str]
]:
    """
    Veriyi kronolojik olarak eğitim, doğrulama ve test kümelerine ayırır.

    Varsayılan oranlar:
        Eğitim: %70
        Doğrulama: %10
        Test: %20

    Returns:
        X_train, X_val, X_test,
        y_train, y_val, y_test, feature_names
    """

    if not 0 < test_size < 1:
        raise ValueError("test_size 0 ile 1 arasında olmalıdır.")

    if not 0 < validation_size < 1 - test_size:
        raise ValueError("validation_size geçerli bir aralıkta olmalıdır.")

    _validate_columns(df)

    df = df.copy()
    df["timestamp"] = pd.to_datetime(
        df["timestamp"], errors="coerce"
    )

    if df["timestamp"].isna().any():
        raise ValueError("Geçersiz timestamp değerleri mevcut.")

    df = df.sort_values(
    ["timestamp", "user_id"],
    kind="mergesort"
).reset_index(drop=True)

    # Etiketleri doğrula.
    df["is_anomaly"] = pd.to_numeric(
        df["is_anomaly"], errors="coerce"
    )

    if df["is_anomaly"].isna().any():
        raise ValueError("is_anomaly sütununda geçersiz etiketler var.")

    if not set(df["is_anomaly"].unique()).issubset({0, 1}):
        raise ValueError("is_anomaly yalnızca 0 ve 1 değerlerini içermeli.")

    df["is_anomaly"] = df["is_anomaly"].astype(int)

    # Sayısal sütunları doğrula.
    numeric_columns = [
        "data_volume_mb",
        "hour_of_day",
        "is_working_hours",
        "old_activity_score",
        "new_activity_score",
        "failed_logins",
        "unique_resources",
    ]

    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df[numeric_columns] = (
        df[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(0)
    )

    # Kronolojik üçlü bölme.
    train_end = int(len(df) * (1 - test_size - validation_size))
    val_end = int(len(df) * (1 - test_size))

    if train_end <= 0 or val_end <= train_end or val_end >= len(df):
        raise ValueError("Eğitim/doğrulama/test bölmesi için yeterli kayıt yok.")

    train_df = df.iloc[:train_end].copy()
    val_df = df.iloc[train_end:val_end].copy()
    test_df = df.iloc[val_end:].copy()

    # İstatistikler ve kategori eşlemesi yalnızca eğitim verisinden öğrenilir.
    user_stats, global_mean, global_std = _fit_user_statistics(train_df)
    activity_mapping = _fit_activity_mapping(train_df)

    # Aynı öğrenilmiş istatistikleri üç kümeye uygula.
    train_features = _transform_features(
        train_df, user_stats, global_mean, global_std, activity_mapping
    )
    val_features = _transform_features(
        val_df, user_stats, global_mean, global_std, activity_mapping
    )
    test_features = _transform_features(
        test_df, user_stats, global_mean, global_std, activity_mapping
    )

    X_train = train_features[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    X_val = val_features[FEATURE_COLUMNS].to_numpy(dtype=np.float64)
    X_test = test_features[FEATURE_COLUMNS].to_numpy(dtype=np.float64)

    y_train = train_df["is_anomaly"].to_numpy(dtype=int)
    y_val = val_df["is_anomaly"].to_numpy(dtype=int)
    y_test = test_df["is_anomaly"].to_numpy(dtype=int)

    # Scaler yalnızca eğitim verisinde fit edilir.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    output_dir = os.path.dirname(scaler_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    joblib.dump(scaler, scaler_path)

    print("[✓] Kronolojik üçlü veri bölme tamamlandı.")
    print(
        f"[✓] Eğitim dönemi: {train_df['timestamp'].min()} - "
        f"{train_df['timestamp'].max()}"
    )
    print(
        f"[✓] Doğrulama dönemi: {val_df['timestamp'].min()} - "
        f"{val_df['timestamp'].max()}"
    )
    print(
        f"[✓] Test dönemi: {test_df['timestamp'].min()} - "
        f"{test_df['timestamp'].max()}"
    )

    print(f"[✓] Train: {X_train.shape}")
    print(f"[✓] Validation: {X_val.shape}")
    print(f"[✓] Test: {X_test.shape}")

    print(f"[✓] Train anomali oranı: %{100 * y_train.mean():.2f}")
    print(f"[✓] Validation anomali oranı: %{100 * y_val.mean():.2f}")
    print(f"[✓] Test anomali oranı: %{100 * y_test.mean():.2f}")

    print(f"[✓] Özellik sayısı: {len(FEATURE_COLUMNS)}")

    return (
        X_train, X_val, X_test,
        y_train, y_val, y_test,
        FEATURE_COLUMNS
    )


def get_user_profiles(df: pd.DataFrame) -> pd.DataFrame:
    """Her kullanıcı için geriye dönük davranış profili özeti üretir."""

    df = engineer_features(df)

    # step sütunu bulunmayan gerçek loglarda da çalışabilsin.
    if "step" in df.columns:
        total_records = ("step", "count")
    else:
        total_records = ("user_id", "size")

    profiles = df.groupby("user_id").agg(
        total_records=total_records,
        mean_volume=("data_volume_mb", "mean"),
        max_volume=("data_volume_mb", "max"),
        mean_failed_logins=("failed_logins", "mean"),
        off_hours_ratio=(
            "is_working_hours",
            lambda x: 1 - x.mean(),
        ),
        anomaly_count=("is_anomaly", "sum"),
        mean_zscore=("volume_zscore", lambda x: abs(x).mean()),
    ).reset_index()

    profiles["anomaly_rate"] = (
        profiles["anomaly_count"] / profiles["total_records"]
    )

    return profiles.sort_values(
        "anomaly_rate",
        ascending=False,
    )
