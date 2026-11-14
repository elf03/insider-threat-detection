
import json
import requests


class ThreatExplainer:
    """
    Ollama üzerinden yerel LLM kullanarak
    tespit edilen tehditleri açıklayan modül.
    """

    def __init__(
        self,
        model="llama3.2:1b",
        api_url="http://localhost:11434/api/generate",
        timeout=120
    ):
        self.model = model
        self.api_url = api_url
        self.timeout = timeout


    def explain_threat(self, threat_data, risk_result):
        """
        Python RiskEngine sonucunu kullanarak
        Ollama üzerinden tehdit açıklaması üretir.
        """

        system_prompt = """
Sen deneyimli bir siber güvenlik ve iç tehdit analistisin.

Görevin, Python tarafından hesaplanmış risk sonucunu
ve tehdit bulgularını açıklamaktır.

KURALLAR:
1. Python tarafından verilen risk puanını değiştirme.
2. Python tarafından verilen risk seviyesini değiştirme.
3. Yeni risk puanı hesaplama.
4. Yalnızca verilen kanıtlara dayan.
5. Eksik bilgileri uydurma.
6. Kullanıcıyı doğrudan saldırgan olarak nitelendirme.
7. Somut ve uygulanabilir güvenlik önerileri sun.
8. Türkçe ve profesyonel bir dil kullan.

Yanıtını yalnızca geçerli JSON olarak döndür.

JSON yapısı:
{
    "summary": "Olayın açıklaması",
    "evidence": [
        "Tespit edilen bulgular"
    ],
    "recommended_actions": [
        "Önerilen güvenlik önlemleri"
    ],
    "confidence_note": "Analizin güvenilirliği ve sınırlamaları"
}
"""

        analysis_data = {
            "threat_data": threat_data,
            "python_risk_assessment": risk_result
        }

        user_prompt = (
            "Aşağıdaki tehdit verilerini ve Python risk "
            "değerlendirmesini açıkla:\n\n"
            + json.dumps(
                analysis_data,
                ensure_ascii=False,
                indent=2
            )
        )

        payload = {
            "model": self.model,
            "system": system_prompt,
            "prompt": user_prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": 0.2
            }
        }

        try:
            response = requests.post(
                self.api_url,
                json=payload,
                timeout=self.timeout
            )

            response.raise_for_status()

            result = response.json()
            llm_output = result.get("response", "")

            if not llm_output:
                raise ValueError("LLM boş yanıt döndürdü.")

            explanation = json.loads(llm_output)

            # Risk kararını ve teknik bulguları Python motorundan al.
            explanation["risk_score"] = risk_result["risk_score"]
            explanation["risk_level"] = risk_result["risk_level"]

            # Kanıtları Python motorunun bulgularıyla sabitle.
            explanation["evidence"] = risk_result["evidence"]

            # Güvenlik önerilerini Python motorundan al.
            explanation["recommended_actions"] = risk_result["recommended_actions"]

            # Nihai risk kararı Python motorundan gelir.
            explanation["risk_score"] = risk_result["risk_score"]
            explanation["risk_level"] = risk_result["risk_level"]

            return explanation

        except requests.exceptions.RequestException as error:
            raise RuntimeError(
                f"Ollama bağlantı hatası: {error}"
            ) from error

        except json.JSONDecodeError as error:
            raise RuntimeError(
                "LLM geçerli JSON formatında yanıt döndürmedi."
            ) from error

        except ValueError as error:
            raise RuntimeError(str(error)) from error
