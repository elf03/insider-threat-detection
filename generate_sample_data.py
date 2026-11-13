"""
generate_sample_data.py
-----------------------
PaySim (Synthetic Financial Datasets For Fraud Detection) veri setini temel alarak
Kullanıcı Davranış Analizi için uyarlanmış sentetik bir iç tehdit veri seti üretir.

Kaynak: https://www.kaggle.com/datasets/ealaxi/paysim1/data

PaySim sütun eşleştirmeleri:
  step          → Zaman adımı (saat cinsinden, 1-744)
  amount        → Veri aktarım boyutu (MB)
  type          → Aktivite türü (LOGIN, FILE_ACCESS, DATA_TRANSFER, PRINT, REMOTE_ACCESS)
  nameOrig      → Kullanıcı ID
  isFraud       → Anomali etiketi (0=normal, 1=iç tehdit)
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

np.random.seed(42)


USERS = [f"USR_{i:04d}" for i in range(1, 101)]   # 100 kullanıcı
TOTAL_STEPS = 720                                    # 30 günlük saatlik veri
NORMAL_RATIO = 0.98                                  # %98 normal, %2 anomali
ANOMALY_USERS = np.random.choice(USERS, size=5, replace=False)  # 5 kötü niyetli kullanıcı

ACTIVITY_TYPES = ["LOGIN", "FILE_ACCESS", "DATA_TRANSFER", "PRINT", "REMOTE_ACCESS"]
ACTIVITY_WEIGHTS = [0.30, 0.35, 0.20, 0.10, 0.05]


def _working_hour(step: int) -> bool:
    """Verilen saat adımının mesai saatinde (09:00-18:00) olup olmadığını döndürür."""
    hour_of_day = step % 24
    return 9 <= hour_of_day <= 18


def generate_normal_session(user_id: str, step: int) -> dict:
    """Normal bir kullanıcı oturumu üretir."""
    hour = step % 24
    is_working = _working_hour(step)

    # Mesai saatinde aktivite yoğunluğu daha yüksek
    base_amount = np.random.lognormal(mean=3.5, sigma=0.8) if is_working else np.random.lognormal(mean=1.5, sigma=0.5)
    activity = np.random.choice(ACTIVITY_TYPES, p=ACTIVITY_WEIGHTS)

    # Kullanıcıya özgü sabit bir "normal davranış profili" skoru
    user_seed = int(user_id.split("_")[1])
    rng = np.random.default_rng(user_seed + step)
    old_balance = rng.uniform(100, 1000)
    new_balance = old_balance - base_amount * 0.1

    return {
        "step": step,
        "user_id": user_id,
        "activity_type": activity,
        "data_volume_mb": round(base_amount, 4),
        "hour_of_day": hour,
        "is_working_hours": int(is_working),
        "old_activity_score": round(old_balance, 4),
        "new_activity_score": round(max(new_balance, 0), 4),
        "failed_logins": int(np.random.poisson(0.1)),
        "unique_resources": int(np.random.poisson(3)),
        "is_anomaly": 0,
        "anomaly_type": "NORMAL"
    }


def generate_anomalous_session(user_id: str, step: int, anomaly_type: str) -> dict:
    """Kötü niyetli kullanıcı davranışı üretir (3 farklı iç tehdit senaryosu)."""
    hour = step % 24

    if anomaly_type == "DATA_EXFILTRATION":
        # Büyük veri sızıntısı: gece saatlerinde çok büyük transfer
        base_amount = np.random.lognormal(mean=7.0, sigma=0.5)   # ~1000+ MB
        activity = "DATA_TRANSFER"
        is_working = 0
    elif anomaly_type == "PRIVILEGE_ABUSE":
        # Yetki istismarı: mesai dışı yoğun dosya erişimi
        base_amount = np.random.lognormal(mean=4.5, sigma=0.6)
        activity = "FILE_ACCESS"
        is_working = 0
    else:  # CREDENTIAL_THEFT
        # Kimlik bilgisi çalma: çok sayıda başarısız giriş
        base_amount = np.random.lognormal(mean=2.0, sigma=0.3)
        activity = "LOGIN"
        is_working = int(_working_hour(step))

    old_balance = np.random.uniform(50, 200)
    new_balance = old_balance - base_amount * 0.5

    return {
        "step": step,
        "user_id": user_id,
        "activity_type": activity,
        "data_volume_mb": round(base_amount, 4),
        "hour_of_day": hour,
        "is_working_hours": is_working,
        "old_activity_score": round(old_balance, 4),
        "new_activity_score": round(max(new_balance, 0), 4),
        "failed_logins": int(np.random.poisson(5 if anomaly_type == "CREDENTIAL_THEFT" else 0.2)),
        "unique_resources": int(np.random.poisson(15)),
        "is_anomaly": 1,
        "anomaly_type": anomaly_type
    }


def generate_dataset(output_path: str = "data/user_behavior_logs.csv") -> pd.DataFrame:
    """Tam veri setini üretir ve CSV olarak kaydeder."""
    print("[*] Kullanıcı davranış veri seti üretiliyor (PaySim tabanlı)...")
    records = []

    # Her kullanıcı için her saatte 1 kayıt
    for user_id in USERS:
        is_malicious = user_id in ANOMALY_USERS
        anomaly_type = np.random.choice(
            ["DATA_EXFILTRATION", "PRIVILEGE_ABUSE", "CREDENTIAL_THEFT"]
        ) if is_malicious else None

        for step in range(1, TOTAL_STEPS + 1):
            # Anomalili kullanıcılar: rastgele %15 oranında kötü niyetli aktivite
            if is_malicious and np.random.random() < 0.15:
                record = generate_anomalous_session(user_id, step, anomaly_type)
            else:
                record = generate_normal_session(user_id, step)
            records.append(record)

    df = pd.DataFrame(records)

    # Zamansal özellik ekle
    base_time = datetime(2024, 1, 1, 0, 0, 0)
    df["timestamp"] = df["step"].apply(lambda s: base_time + timedelta(hours=int(s)))

    # Sütun sıralaması
    cols = [
        "timestamp", "step", "user_id", "activity_type",
        "data_volume_mb", "hour_of_day", "is_working_hours",
        "old_activity_score", "new_activity_score",
        "failed_logins", "unique_resources",
        "is_anomaly", "anomaly_type"
    ]
    df = df[cols]

    os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
    df.to_csv(output_path, index=False)

    total = len(df)
    anomalies = df["is_anomaly"].sum()
    print(f"[✓] Veri seti oluşturuldu: {total:,} kayıt | {anomalies:,} anomali ({100*anomalies/total:.2f}%)")
    print(f"[✓] Kötü niyetli kullanıcılar: {list(ANOMALY_USERS)}")
    print(f"[✓] Kaydedildi: {output_path}")
    return df


if __name__ == "__main__":
    df = generate_dataset()
    print(df.head(10).to_string())
