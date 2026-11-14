
class RiskEngine:
    """
    Makine öğrenmesi sonuçları ve davranış göstergeleriyle
    deterministik risk puanı hesaplar.
    """

    def evaluate(self, threat_data):
        score = 0
        evidence = []

        # 1. Model uzlaşması
        detected_by = threat_data.get("detected_by", [])
        model_count = len(set(detected_by))

        model_score = min(model_count * 10, 40)
        score += model_score

        if model_count > 0:
            evidence.append(
                f"{model_count} farklı model şüpheli davranış tespit etti."
            )

        # 2. Başarısız girişler
        failed_logins = threat_data.get("failed_logins", 0)

        if failed_logins >= 10:
            score += 20
            evidence.append(
                f"{failed_logins} başarısız giriş denemesi tespit edildi."
            )
        elif failed_logins >= 5:
            score += 10
            evidence.append(
                f"{failed_logins} başarısız giriş denemesi tespit edildi."
            )

        # 3. Mesai dışı faaliyet
        hour = threat_data.get("hour_of_day")

        if hour is not None and (hour < 6 or hour >= 22):
            score += 10
            evidence.append(
                f"Faaliyet mesai dışı saatte gerçekleşti: {hour}:00."
            )

        # 4. Veri aktarım miktarı
        data_volume = threat_data.get("data_volume_mb", 0)

        if data_volume >= 1000:
            score += 20
            evidence.append(
                f"Yüksek veri aktarımı: {data_volume} MB."
            )
        elif data_volume >= 500:
            score += 10
            evidence.append(
                f"Yüksek olabilecek veri aktarımı: {data_volume} MB."
            )

        # 5. Kaynak erişim sayısı
        unique_resources = threat_data.get("unique_resources", 0)

        if unique_resources >= 25:
            score += 10
            evidence.append(
                f"{unique_resources} farklı kaynağa erişim gerçekleşti."
            )

        # Puan sınırı
        score = min(score, 100)

        # Risk seviyesi
        if score >= 80:
            risk_level = "Kritik"
        elif score >= 60:
            risk_level = "Yüksek"
        elif score >= 30:
            risk_level = "Orta"
        else:
            risk_level = "Düşük"


        recommended_actions = []

        if failed_logins >= 5:
            recommended_actions.append(
                "Başarısız giriş denemelerinin kaynak IP adreslerini ve "
                "kimlik doğrulama kayıtlarını incele."
            )

        if hour is not None and (hour < 6 or hour >= 22):
            recommended_actions.append(
                "Kullanıcının mesai dışı erişim yetkisini ve "
                "oturum geçmişini doğrula."
            )

        if data_volume >= 500:
            recommended_actions.append(
                "Aktarılan verinin hedefini, içeriğini ve "
                "yetkilendirme durumunu incele."
            )

        if unique_resources >= 25:
            recommended_actions.append(
                "Erişilen kaynakları ve kullanıcının erişim "
                "yetkilerini denetle."
            )

        if model_count >= 3:
            recommended_actions.append(
                "Birden fazla modelin ortak tespit ettiği olayı "
                "öncelikli güvenlik incelemesine al."
            )


        return {
            "risk_score": score,
            "risk_level": risk_level,
            "model_count": model_count,
            "evidence": evidence,
            "recommended_actions": recommended_actions
        }
