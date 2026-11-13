"""
visualization.py
-----------------
Tüm grafik ve görselleştirme fonksiyonları.
Matplotlib + Seaborn kullanılarak akademik kalitede grafikler üretir.
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # GUI olmadan kaydetmek için
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, confusion_matrix
from typing import Dict, List
import os

# Stil ayarları
plt.rcParams.update({
    "figure.dpi": 150,
    "font.family": "DejaVu Sans",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
})
COLORS = {
    "normal": "#2ecc71",
    "anomaly": "#e74c3c",
    "if": "#3498db",
    "ae": "#9b59b6",
    "rule": "#f39c12",
    "bg": "#0d1117",
    "text": "#ffffff",
}

PLOT_DIR = "results/plots"


def _save(fig, name: str):
    os.makedirs(PLOT_DIR, exist_ok=True)
    path = os.path.join(PLOT_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"[✓] Graf kaydedildi: {path}")
    return path


# ─────────────────────────────────────────────────────────────
# 1. Veri Dağılımı
# ─────────────────────────────────────────────────────────────
def plot_data_distribution(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), facecolor="#0f1923")
    fig.suptitle("Kullanıcı Davranış Veri Seti Dağılımı", fontsize=16, color="white", fontweight="bold", y=1.01)

    for ax in axes.flat:
        ax.set_facecolor("#1a2332")
        for spine in ax.spines.values():
            spine.set_color("#2d3f55")

    # (1) Anomali/Normal dağılımı (pasta grafik)
    ax = axes[0, 0]
    counts = df["is_anomaly"].value_counts()
    wedges, texts, autotexts = ax.pie(
        counts, labels=["Normal", "Anomali"], colors=[COLORS["normal"], COLORS["anomaly"]],
        autopct="%1.2f%%", startangle=90, textprops={"color": "white"},
        wedgeprops={"edgecolor": "white", "linewidth": 1.5}
    )
    ax.set_title("Veri Seti Dengesizliği", color="white", pad=10)

    # (2) Veri hacmi dağılımı
    ax = axes[0, 1]
    normal_vol = df[df["is_anomaly"] == 0]["data_volume_mb"]
    anomaly_vol = df[df["is_anomaly"] == 1]["data_volume_mb"]
    ax.hist(np.log1p(normal_vol), bins=50, color=COLORS["normal"], alpha=0.7, label="Normal", density=True)
    ax.hist(np.log1p(anomaly_vol), bins=50, color=COLORS["anomaly"], alpha=0.7, label="Anomali", density=True)
    ax.set_xlabel("log(1 + Veri Hacmi MB)", color="white")
    ax.set_ylabel("Yoğunluk", color="white")
    ax.set_title("Veri Transfer Hacmi Dağılımı", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a2332", labelcolor="white")

    # (3) Saatlik aktivite yoğunluğu
    ax = axes[1, 0]
    hourly = df.groupby(["hour_of_day", "is_anomaly"]).size().unstack(fill_value=0)
    hours = range(24)
    if 0 in hourly.columns:
        ax.bar(hours, hourly.get(0, [0]*24), color=COLORS["normal"], alpha=0.7, label="Normal")
    if 1 in hourly.columns:
        ax.bar(hours, hourly.get(1, [0]*24), color=COLORS["anomaly"], alpha=0.7, label="Anomali", bottom=hourly.get(0, [0]*24))
    ax.set_xlabel("Saat", color="white")
    ax.set_ylabel("Kayıt Sayısı", color="white")
    ax.set_title("Saatlik Aktivite Dağılımı", color="white")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a2332", labelcolor="white")
    ax.axvspan(0, 9, alpha=0.1, color="red", label="Mesai Dışı")
    ax.axvspan(18, 24, alpha=0.1, color="red")

    # (4) Aktivite türü dağılımı
    ax = axes[1, 1]
    act_counts = df[df["is_anomaly"] == 1]["activity_type"].value_counts()
    bars = ax.barh(act_counts.index, act_counts.values, color=COLORS["anomaly"], alpha=0.8)
    ax.set_xlabel("Anomali Sayısı", color="white")
    ax.set_title("Anomali Türlerine Göre Aktivite", color="white")
    ax.tick_params(colors="white")
    for bar, val in zip(bars, act_counts.values):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2, str(val),
                va="center", color="white", fontsize=9)

    plt.tight_layout()
    return _save(fig, "01_data_distribution.png")


# ─────────────────────────────────────────────────────────────
# 2. ROC-AUC Karşılaştırma
# ─────────────────────────────────────────────────────────────
def plot_roc_curves(y_test: np.ndarray, if_results: Dict, ae_results: Dict) -> str:
    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#0f1923")
    ax.set_facecolor("#1a2332")
    for spine in ax.spines.values():
        spine.set_color("#2d3f55")

    for results, color, label in [
        (if_results, COLORS["if"], "Isolation Forest"),
        (ae_results, COLORS["ae"], "Autoencoder"),
    ]:
        fpr_curve, tpr_curve, _ = roc_curve(y_test, results["y_scores"])
        auc = results["roc_auc"]
        ax.plot(fpr_curve, tpr_curve, color=color, lw=2.5, label=f"{label} (AUC={auc:.4f})")
        ax.fill_between(fpr_curve, tpr_curve, alpha=0.08, color=color)

    ax.plot([0, 1], [0, 1], "w--", lw=1, alpha=0.4, label="Rastgele Tahmin (AUC=0.50)")
    ax.set_xlabel("Yanlış Pozitif Oranı (FPR)", color="white", fontsize=12)
    ax.set_ylabel("Doğru Pozitif Oranı (TPR)", color="white", fontsize=12)
    ax.set_title("ROC-AUC Eğrisi Karşılaştırması", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a2332", labelcolor="white", fontsize=11)
    ax.grid(True, alpha=0.2, color="white")
    return _save(fig, "02_roc_curves.png")


# ─────────────────────────────────────────────────────────────
# 3. Precision-Recall Eğrisi
# ─────────────────────────────────────────────────────────────
def plot_precision_recall(y_test: np.ndarray, if_results: Dict, ae_results: Dict) -> str:
    fig, ax = plt.subplots(figsize=(9, 7), facecolor="#0f1923")
    ax.set_facecolor("#1a2332")

    for results, color, label in [
        (if_results, COLORS["if"], "Isolation Forest"),
        (ae_results, COLORS["ae"], "Autoencoder"),
    ]:
        precision, recall, _ = precision_recall_curve(y_test, results["y_scores"])
        ap = results["avg_precision"]
        ax.plot(recall, precision, color=color, lw=2.5, label=f"{label} (AP={ap:.4f})")
        ax.fill_between(recall, precision, alpha=0.08, color=color)

    baseline = y_test.mean()
    ax.axhline(baseline, color="white", ls="--", lw=1, alpha=0.4, label=f"Baseline (Anomali Oranı={baseline:.3f})")
    ax.set_xlabel("Recall", color="white", fontsize=12)
    ax.set_ylabel("Precision", color="white", fontsize=12)
    ax.set_title("Precision-Recall Eğrisi", color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="white")
    ax.legend(facecolor="#1a2332", labelcolor="white", fontsize=11)
    ax.set_facecolor("#1a2332")
    return _save(fig, "03_precision_recall.png")


# ─────────────────────────────────────────────────────────────
# 4. Confusion Matrix
# ─────────────────────────────────────────────────────────────
def plot_confusion_matrices(y_test: np.ndarray, if_results: Dict, ae_results: Dict) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#0f1923")
    fig.suptitle("Confusion Matrix Karşılaştırması", color="white", fontsize=15, fontweight="bold")

    for ax, results, title in [
        (axes[0], if_results, "Isolation Forest"),
        (axes[1], ae_results, "Autoencoder"),
    ]:
        cm = results["confusion_matrix"]
        ax.set_facecolor("#1a2332")
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=["Normal (0)", "Anomali (1)"],
            yticklabels=["Normal (0)", "Anomali (1)"],
            ax=ax, cbar=True,
            linewidths=0.5, linecolor="#0f1923",
            annot_kws={"size": 14, "weight": "bold"}
        )
        ax.set_title(title, color="white", fontsize=13, pad=12)
        ax.set_xlabel("Tahmin", color="white")
        ax.set_ylabel("Gerçek", color="white")
        ax.tick_params(colors="white")

        prec = results["precision"]
        rec = results["recall"]
        fpr = results["false_positive_rate"]
        ax.text(0.5, -0.18, f"P={prec:.3f}  R={rec:.3f}  FPR={fpr:.3f}",
                ha="center", transform=ax.transAxes, color="lightyellow", fontsize=10)

    plt.tight_layout()
    return _save(fig, "04_confusion_matrices.png")


# ─────────────────────────────────────────────────────────────
# 5. False Positive Karşılaştırması (Kongre Bildirisi Odak)
# ─────────────────────────────────────────────────────────────
def plot_false_positive_comparison(comparison_df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), facecolor="#0f1923")
    fig.suptitle(
        "Kural Tabanlı vs. Yapay Zeka Tabanlı Anomali Tespiti\n"
        "Kongre Bildirisi: False Positive Yorgunluğunun Azaltılması",
        color="white", fontsize=13, fontweight="bold"
    )

    methods = comparison_df["Yöntem"].tolist()
    bar_colors = [COLORS["rule"], COLORS["if"], COLORS["ae"]]

    # (1) FPR karşılaştırma
    ax = axes[0]
    ax.set_facecolor("#1a2332")
    fprs = [float(x) if x != "N/A" else 0 for x in comparison_df["False Positive Rate"]]
    bars = ax.bar(methods, fprs, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=1.2)
    ax.set_ylabel("False Positive Rate", color="white", fontsize=12)
    ax.set_title("Yanlış Alarm Oranı", color="white", fontsize=12)
    ax.tick_params(colors="white", axis="y")
    ax.tick_params(colors="white", axis="x", rotation=10)
    for bar, val in zip(bars, fprs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"%{val*100:.1f}", ha="center", color="white", fontsize=12, fontweight="bold")

    # Hedef çizgisi
    ax.axhline(0.15, color="lime", ls="--", lw=1.5, alpha=0.7, label="Hedef FPR < %15")
    ax.legend(facecolor="#1a2332", labelcolor="white")

    # (2) F1-Score karşılaştırma
    ax = axes[1]
    ax.set_facecolor("#1a2332")
    f1s = [float(x) if x != "N/A" else 0 for x in comparison_df["F1-Score"]]
    bars2 = ax.bar(methods, f1s, color=bar_colors, alpha=0.85, edgecolor="white", linewidth=1.2)
    ax.set_ylabel("F1-Score", color="white", fontsize=12)
    ax.set_title("Tespit Başarısı (F1-Score)", color="white", fontsize=12)
    ax.tick_params(colors="white", axis="y")
    ax.tick_params(colors="white", axis="x", rotation=10)
    for bar, val in zip(bars2, f1s):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", color="white", fontsize=12, fontweight="bold")

    plt.tight_layout()
    return _save(fig, "05_false_positive_comparison.png")


# ─────────────────────────────────────────────────────────────
# 6. Anomali Skor Zaman Serisi
# ─────────────────────────────────────────────────────────────
def plot_anomaly_timeline(df: pd.DataFrame, if_results: Dict, ae_results: Dict, X_test: np.ndarray) -> str:
    n = min(500, len(X_test))
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), facecolor="#0f1923", sharex=True)
    fig.suptitle("Anomali Skor Zaman Serisi (Test Verisi)", color="white", fontsize=14, fontweight="bold")

    titles = ["Isolation Forest Anomali Skoru", "Autoencoder Yeniden Yapılandırma Hatası"]
    scores_list = [if_results["y_scores"][:n], ae_results["y_scores"][:n]]
    colors_list = [COLORS["if"], COLORS["ae"]]

    for ax, title, scores, color in zip(axes, titles, scores_list, colors_list):
        ax.set_facecolor("#1a2332")
        t = np.arange(n)
        ax.plot(t, scores, color=color, lw=1.2, alpha=0.8, label="Anomali Skoru")
        
        # Gerçek anomalileri işaretle
        y_true_n = if_results["y_pred"][:n]
        anomaly_idx = np.where(y_true_n == 1)[0]
        ax.scatter(anomaly_idx, scores[anomaly_idx], color=COLORS["anomaly"], 
                   s=30, zorder=5, alpha=0.8, label="Tespit Edilen Anomali")

        # Eşik çizgisi
        threshold = np.percentile(scores, 95)
        ax.axhline(threshold, color="yellow", ls="--", lw=1.5, alpha=0.7, label=f"Eşik (P95)")
        ax.set_title(title, color="white", fontsize=12)
        ax.set_ylabel("Skor", color="white")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#1a2332", labelcolor="white", fontsize=9)

    axes[-1].set_xlabel("Zaman Adımı (Test Verisi)", color="white")
    plt.tight_layout()
    return _save(fig, "06_anomaly_timeline.png")


# ─────────────────────────────────────────────────────────────
# 7. Kullanıcı Risk Profili
# ─────────────────────────────────────────────────────────────
def plot_user_risk_heatmap(profiles: pd.DataFrame) -> str:
    top_n = 20
    top_profiles = profiles.head(top_n).copy()

    fig, ax = plt.subplots(figsize=(12, 8), facecolor="#0f1923")
    ax.set_facecolor("#1a2332")

    heatmap_data = top_profiles[["anomaly_rate", "mean_volume", "mean_failed_logins", "off_hours_ratio"]].copy()
    heatmap_data.index = top_profiles["user_id"]
    heatmap_data.columns = ["Anomali Oranı", "Ort. Veri Hacmi", "Ort. Başarısız Giriş", "Mesai Dışı Oranı"]

    # Normalize
    heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-9)

    sns.heatmap(
        heatmap_norm, annot=False, cmap="YlOrRd",
        ax=ax, cbar=True, linewidths=0.3, linecolor="#0f1923",
        yticklabels=heatmap_data.index
    )
    ax.set_title(f"Kullanıcı Risk Profili Haritası (İlk {top_n} Riskli Kullanıcı)",
                 color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    ax.set_xlabel("Risk Faktörleri", color="white")
    ax.set_ylabel("Kullanıcı ID", color="white")

    plt.tight_layout()
    return _save(fig, "07_user_risk_heatmap.png")


# ─────────────────────────────────────────────────────────────
# 8. Autoencoder Eğitim Grafiği
# ─────────────────────────────────────────────────────────────
def plot_training_history(history) -> str:
    if history is None:
        return ""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), facecolor="#0f1923")
    fig.suptitle("Autoencoder Eğitim Süreci", color="white", fontsize=14, fontweight="bold")

    for ax in axes:
        ax.set_facecolor("#1a2332")
        for spine in ax.spines.values():
            spine.set_color("#2d3f55")

    # Loss
    axes[0].plot(history.history["loss"], color=COLORS["ae"], lw=2, label="Train Loss")
    if "val_loss" in history.history:
        axes[0].plot(history.history["val_loss"], color=COLORS["rule"], lw=2, label="Val Loss")
    axes[0].set_title("MSE Kaybı", color="white")
    axes[0].set_xlabel("Epoch", color="white")
    axes[0].set_ylabel("MSE", color="white")
    axes[0].tick_params(colors="white")
    axes[0].legend(facecolor="#1a2332", labelcolor="white")

    # MAE
    if "mae" in history.history:
        axes[1].plot(history.history["mae"], color=COLORS["ae"], lw=2, label="Train MAE")
        if "val_mae" in history.history:
            axes[1].plot(history.history["val_mae"], color=COLORS["rule"], lw=2, label="Val MAE")
        axes[1].set_title("Ortalama Mutlak Hata (MAE)", color="white")
        axes[1].set_xlabel("Epoch", color="white")
        axes[1].set_ylabel("MAE", color="white")
        axes[1].tick_params(colors="white")
        axes[1].legend(facecolor="#1a2332", labelcolor="white")

    plt.tight_layout()
    return _save(fig, "08_autoencoder_training.png")
