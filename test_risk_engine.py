
import pytest
from src.risk_engine import RiskEngine


@pytest.fixture
def engine():
    return RiskEngine()


@pytest.fixture
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


def test_risk_engine_returns_result(engine, sample_threat):
    result = engine.evaluate(sample_threat)

    assert isinstance(result, dict)
    assert "risk_score" in result
    assert "risk_level" in result
    assert "model_count" in result
    assert "evidence" in result


def test_risk_score_is_within_valid_range(engine, sample_threat):
    result = engine.evaluate(sample_threat)

    assert 0 <= result["risk_score"] <= 100


def test_all_four_models_are_counted(engine, sample_threat):
    result = engine.evaluate(sample_threat)

    assert result["model_count"] == 4


def test_evidence_is_a_list(engine, sample_threat):
    result = engine.evaluate(sample_threat)

    assert isinstance(result["evidence"], list)
