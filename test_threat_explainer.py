
from unittest.mock import patch

from src.risk_engine import RiskEngine
from src.threat_explainer import ThreatExplainer


def sample_threat():
    return {
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


def test_risk_engine_and_explainer_initialization():
    engine = RiskEngine()
    explainer = ThreatExplainer()

    assert engine is not None
    assert explainer is not None


def test_explainer_returns_expected_fields():
    threat = sample_threat()
    risk_result = RiskEngine().evaluate(threat)

    mocked_response = {
        "summary": "Şüpheli kullanıcı davranışı tespit edildi.",
        "recommended_actions": [
            "Kullanıcı hesabını inceleyin."
        ],
        "confidence_note": "Sentetik veri üzerinde oluşturulmuştur."
    }

    with patch.object(
        ThreatExplainer,
        "explain_threat",
        return_value={
            **risk_result,
            **mocked_response
        }
    ) as mocked_explain:

        explainer = ThreatExplainer()
        result = explainer.explain_threat(threat, risk_result)

        assert "risk_score" in result
        assert "risk_level" in result
        assert "summary" in result
        assert "evidence" in result
        assert "recommended_actions" in result
        assert "confidence_note" in result

        mocked_explain.assert_called_once()


def test_risk_result_is_preserved():
    threat = sample_threat()
    risk_result = RiskEngine().evaluate(threat)

    assert 0 <= risk_result["risk_score"] <= 100
    assert isinstance(risk_result["evidence"], list)
