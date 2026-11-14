
from src.risk_engine import RiskEngine

engine = RiskEngine()

sample_threat = {
    "user_id": "USER_042",
    "failed_logins": 12,
    "hour_of_day": 3,
    "data_volume_mb": 850,
    "unique_resources": 28,
    "detected_by": [
        "Isolation Forest",
        "Autoencoder",
        "Hybrid",
        "LSTM"
    ]
}

result = engine.evaluate(sample_threat)

print("\n===== RİSK MOTORU TESTİ =====")
print("Kullanıcı:", sample_threat["user_id"])
print("Risk puanı:", result["risk_score"], "/ 100")
print("Risk seviyesi:", result["risk_level"])
print("Model sayısı:", result["model_count"])

print("\nTespit edilen bulgular:")
for item in result["evidence"]:
    print("-", item)
