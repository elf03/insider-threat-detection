
from src.risk_engine import RiskEngine
from src.threat_explainer import ThreatExplainer


# 1. Örnek tehdit verisi
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


# 2. Python risk motoru
risk_engine = RiskEngine()
risk_result = risk_engine.evaluate(sample_threat)

print("\n===== PYTHON RİSK DEĞERLENDİRMESİ =====")
print("Risk puanı:", risk_result["risk_score"], "/ 100")
print("Risk seviyesi:", risk_result["risk_level"])


# 3. Yerel LLM açıklaması
explainer = ThreatExplainer()

try:
    explanation = explainer.explain_threat(
        sample_threat,
        risk_result
    )

    print("\n===== YAPAY ZEKA TEHDİT RAPORU =====")

    print("Risk puanı:", explanation["risk_score"])
    print("Risk seviyesi:", explanation["risk_level"])

    print("\nOlay özeti:")
    print(explanation["summary"])

    print("\nTespit edilen bulgular:")
    for item in explanation["evidence"]:
        print("-", item)

    print("\nÖnerilen güvenlik önlemleri:")
    for action in explanation["recommended_actions"]:
        print("-", action)

    print("\nGüvenilirlik notu:")
    print(explanation["confidence_note"])

except Exception as error:
    print("\nTehdit analizi başarısız:", error)
