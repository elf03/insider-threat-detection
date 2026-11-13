/**
 * app.js — İç Tehdit Tespit Sistemi Dashboard
 * Chart.js ile interaktif grafikler
 * Önce data.json'dan yüklenir, bulunamazsa demo veri kullanır
 */

// ─── CHART.JS GLOBAL DEFAULTS ───────────────────────────
Chart.defaults.color = "#8aabb5";
Chart.defaults.borderColor = "rgba(30,58,90,0.5)";
Chart.defaults.font.family = "Inter, system-ui, sans-serif";

// ─── DEMO DATA (main.py çalıştırılmadan önce) ───────────
const DEMO_DATA = {
  summary: {
    total_records: 72000,
    total_anomalies: 1620,
    anomaly_rate: 0.0225,
    total_users: 100,
    malicious_users_detected: 5
  },
  if_metrics: {
    precision: 0.8734,
    recall: 0.9512,
    f1: 0.9107,
    roc_auc: 0.9821,
    fpr: 0.0821
  },
  ae_metrics: {
    precision: 0.8921,
    recall: 0.9345,
    f1: 0.9128,
    roc_auc: 0.9756,
    fpr: 0.0643,
    threshold: 0.00342
  },
  model_comparison: [
    { "Yöntem": "Kural Tabanlı (SIEM)", "Precision": 0.2341, "Recall": 0.9120, "F1-Score": 0.3745, "False Positive Rate": 0.6723, "Yanlış Alarm Oranı (%)": 67.2 },
    { "Yöntem": "Isolation Forest (AI)", "Precision": 0.8734, "Recall": 0.9512, "F1-Score": 0.9107, "False Positive Rate": 0.0821, "Yanlış Alarm Oranı (%)": 8.2 },
    { "Yöntem": "Autoencoder (AI)", "Precision": 0.8921, "Recall": 0.9345, "F1-Score": 0.9128, "False Positive Rate": 0.0643, "Yanlış Alarm Oranı (%)": 6.4 }
  ],
  hourly_activity: Array.from({length:24}, (_,h) => ({
    hour_of_day: h,
    sum: h >= 9 && h <= 18 ? Math.round(20 + Math.random()*30) : Math.round(60 + Math.random()*80),
    rate: h >= 9 && h <= 18 ? 0.008 + Math.random()*0.01 : 0.03 + Math.random()*0.05
  })),
  top_risky_users: [
    { user_id: "USR_0023", anomaly_rate: 0.1532, mean_volume: 847.2, anomaly_count: 110 },
    { user_id: "USR_0067", anomaly_rate: 0.1341, mean_volume: 723.5, anomaly_count: 97  },
    { user_id: "USR_0089", anomaly_rate: 0.1218, mean_volume: 1024.1, anomaly_count: 88 },
    { user_id: "USR_0012", anomaly_rate: 0.1104, mean_volume: 612.8, anomaly_count: 80  },
    { user_id: "USR_0045", anomaly_rate: 0.0932, mean_volume: 589.3, anomaly_count: 67  },
    { user_id: "USR_0031", anomaly_rate: 0.0412, mean_volume: 234.1, anomaly_count: 30  },
    { user_id: "USR_0074", anomaly_rate: 0.0287, mean_volume: 198.4, anomaly_count: 21  },
    { user_id: "USR_0053", anomaly_rate: 0.0201, mean_volume: 165.2, anomaly_count: 15  },
  ],
  timeline: {
    if_scores:   Array.from({length:200}, () => Math.random() * 0.3 + (Math.random() < 0.05 ? 0.7 : 0)),
    ae_scores:   Array.from({length:200}, () => Math.random() * 0.002 + (Math.random() < 0.05 ? 0.008 : 0)),
    true_labels: Array.from({length:200}, () => Math.random() < 0.05 ? 1 : 0),
    if_preds:    Array.from({length:200}, () => Math.random() < 0.06 ? 1 : 0),
    ae_preds:    Array.from({length:200}, () => Math.random() < 0.055 ? 1 : 0),
  }
};

// ─── DATA LOADING ────────────────────────────────────────
async function loadData() {
  try {
    const res = await fetch("data.json");
    if (!res.ok) throw new Error("data.json bulunamadı");
    const data = await res.json();
    console.log("[✓] data.json yüklendi");
    return data;
  } catch (e) {
    console.warn("[!] data.json yüklenemedi, demo veri kullanılıyor:", e.message);
    return DEMO_DATA;
  }
}

// ─── HELPERS ─────────────────────────────────────────────
function fmt(val, dec = 4) {
  if (val === null || val === undefined || val === "N/A") return "N/A";
  return Number(val).toFixed(dec);
}
function pct(val) {
  return (Number(val) * 100).toFixed(2) + "%";
}
function setBar(id, val) {
  const el = document.getElementById(id);
  if (el) setTimeout(() => el.style.width = (Math.min(val, 1) * 100) + "%", 100);
}
function setText(id, text) {
  const el = document.getElementById(id);
  if (el) el.textContent = text;
}

function getRiskLevel(rate) {
  if (rate > 0.10) return ['risk-critical', 'KRİTİK'];
  if (rate > 0.05) return ['risk-high', 'YÜKSEK'];
  if (rate > 0.02) return ['risk-medium', 'ORTA'];
  return ['risk-low', 'DÜŞÜK'];
}

// ─── KPI CARDS ───────────────────────────────────────────
function renderKPIs(data) {
  setText("kpi-users", data.summary.total_users.toLocaleString("tr-TR"));
  setText("kpi-anomalies", data.summary.total_anomalies.toLocaleString("tr-TR"));
  setText("kpi-rate", (data.summary.anomaly_rate * 100).toFixed(2) + "%");
  setText("kpi-malicious", data.summary.malicious_users_detected);
}

// ─── METRICS ─────────────────────────────────────────────
function renderMetrics(data) {
  const IF = data.if_metrics;
  const AE = data.ae_metrics;

  setText("if-prec", fmt(IF.precision)); setBar("if-prec-bar", IF.precision);
  setText("if-rec",  fmt(IF.recall));    setBar("if-rec-bar",  IF.recall);
  setText("if-f1",   fmt(IF.f1));        setBar("if-f1-bar",   IF.f1);
  setText("if-auc",  fmt(IF.roc_auc));   setBar("if-auc-bar",  IF.roc_auc);
  setText("if-fpr",  pct(IF.fpr));       setBar("if-fpr-bar",  IF.fpr);

  setText("ae-prec", fmt(AE.precision)); setBar("ae-prec-bar", AE.precision);
  setText("ae-rec",  fmt(AE.recall));    setBar("ae-rec-bar",  AE.recall);
  setText("ae-f1",   fmt(AE.f1));        setBar("ae-f1-bar",   AE.f1);
  setText("ae-auc",  fmt(AE.roc_auc));   setBar("ae-auc-bar",  AE.roc_auc);
  setText("ae-fpr",  pct(AE.fpr));       setBar("ae-fpr-bar",  AE.fpr);
}

// ─── TIMELINE CHART ──────────────────────────────────────
function renderTimeline(data) {
  const tl = data.timeline;
  const n = tl.if_scores.length;
  const labels = Array.from({length: n}, (_, i) => `T+${i}`);

  // Normalize AE scores to [0,1] for display
  const aeMax = Math.max(...tl.ae_scores);
  const aeNorm = tl.ae_scores.map(v => v / (aeMax || 1));

  new Chart(document.getElementById("timelineChart"), {
    type: "line",
    data: {
      labels,
      datasets: [
        {
          label: "Isolation Forest Skoru",
          data: tl.if_scores,
          borderColor: "#00aaff",
          backgroundColor: "rgba(0,170,255,0.06)",
          borderWidth: 1.5,
          fill: true,
          tension: 0.3,
          pointRadius: 0,
          pointHitRadius: 10,
        },
        {
          label: "Autoencoder Hatası (norm.)",
          data: aeNorm,
          borderColor: "#9b59ff",
          backgroundColor: "rgba(155,89,255,0.06)",
          borderWidth: 1.5,
          fill: true,
          tension: 0.3,
          pointRadius: 0,
          pointHitRadius: 10,
        },
        {
          label: "Gerçek Anomali",
          data: tl.true_labels.map((v, i) => v === 1 ? tl.if_scores[i] : null),
          backgroundColor: "rgba(255,68,68,0.9)",
          borderColor: "transparent",
          type: "scatter",
          pointRadius: 5,
          pointStyle: "triangle",
          showLine: false,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      animation: { duration: 1200 },
      plugins: {
        legend: { labels: { usePointStyle: true, padding: 16, font: { size: 11 } } },
        tooltip: { mode: "index", intersect: false }
      },
      scales: {
        x: { ticks: { maxTicksLimit: 10 }, grid: { color: "rgba(255,255,255,0.04)" } },
        y: { min: 0, grid: { color: "rgba(255,255,255,0.04)" } }
      }
    }
  });
}

// ─── HOURLY CHART ─────────────────────────────────────────
function renderHourly(data) {
  const hourly = data.hourly_activity;
  const labels = hourly.map(h => `${String(h.hour_of_day).padStart(2,'0')}:00`);
  const rates = hourly.map(h => (h.rate * 100).toFixed(3));
  const counts = hourly.map(h => h.sum || 0);

  // Color bars: working hours green, off hours red
  const bgColors = hourly.map(h => {
    const hr = h.hour_of_day;
    return (hr >= 9 && hr <= 18)
      ? "rgba(0,230,118,0.5)"
      : "rgba(255,68,68,0.5)";
  });

  new Chart(document.getElementById("hourlyChart"), {
    type: "bar",
    data: {
      labels,
      datasets: [{
        label: "Anomali Oranı (%)",
        data: rates,
        backgroundColor: bgColors,
        borderColor: bgColors.map(c => c.replace("0.5", "0.9")),
        borderWidth: 1,
        borderRadius: 4,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: true, labels: { font: { size: 11 } } },
        tooltip: {
          callbacks: {
            label: ctx => `Anomali Oranı: %${ctx.parsed.y}`,
            afterLabel: (ctx) => {
              const h = hourly[ctx.dataIndex].hour_of_day;
              return (h >= 9 && h <= 18) ? "🟢 Mesai Saati" : "🔴 Mesai Dışı";
            }
          }
        }
      },
      scales: {
        x: { ticks: { maxTicksLimit: 12, font: { size: 10 } }, grid: { color: "rgba(255,255,255,0.04)" } },
        y: { title: { display: true, text: "Anomali Oranı (%)", color: "#8aabb5" }, grid: { color: "rgba(255,255,255,0.04)" } }
      }
    }
  });
}

// ─── FALSE POSITIVE COMPARISON CHART ─────────────────────
function renderFPComparison(data) {
  const methods = data.model_comparison.map(d => d["Yöntem"]);
  const fprs = data.model_comparison.map(d => +(d["Yanlış Alarm Oranı (%)"] || (d["False Positive Rate"] * 100)).toFixed(1));
  const colors = ["rgba(255,152,0,0.75)", "rgba(0,170,255,0.75)", "rgba(155,89,255,0.75)"];
  const borders = ["#ff9800", "#00aaff", "#9b59ff"];

  new Chart(document.getElementById("fpCompareChart"), {
    type: "bar",
    data: {
      labels: methods,
      datasets: [{
        label: "Yanlış Alarm Oranı (%)",
        data: fprs,
        backgroundColor: colors,
        borderColor: borders,
        borderWidth: 2,
        borderRadius: 8,
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: { display: false },
        tooltip: {
          callbacks: {
            label: ctx => `Yanlış Alarm Oranı: %${ctx.parsed.y}`,
            afterLabel: (ctx) => {
              if (ctx.dataIndex === 0) return "⚠️ Yüksek FP — Analist yorgunluğu!";
              return "✅ AI ile azaltılmış FP";
            }
          }
        },
        annotation: {}
      },
      scales: {
        x: { ticks: { font: { size: 11 } }, grid: { display: false } },
        y: {
          title: { display: true, text: "Yanlış Alarm Oranı (%)", color: "#8aabb5" },
          grid: { color: "rgba(255,255,255,0.04)" },
          max: Math.max(...fprs) * 1.3
        }
      }
    },
    plugins: [{
      id: "datalabels",
      afterDatasetsDraw(chart) {
        const ctx2 = chart.ctx;
        chart.data.datasets[0].data.forEach((val, i) => {
          const meta = chart.getDatasetMeta(0);
          const bar = meta.data[i];
          ctx2.fillStyle = "#e8f4ff";
          ctx2.font = "bold 13px Inter";
          ctx2.textAlign = "center";
          ctx2.fillText(`%${val}`, bar.x, bar.y - 8);
        });
      }
    }]
  });
}

// ─── F1 SCORE CHART ──────────────────────────────────────
function renderF1Chart(data) {
  const methods = data.model_comparison.map(d => d["Yöntem"]);
  const f1s = data.model_comparison.map(d => +Number(d["F1-Score"]).toFixed(4));
  const colors = ["rgba(255,152,0,0.7)", "rgba(0,170,255,0.7)", "rgba(155,89,255,0.7)"];

  new Chart(document.getElementById("f1Chart"), {
    type: "bar",
    data: {
      labels: methods,
      datasets: [{
        label: "F1-Score",
        data: f1s,
        backgroundColor: colors,
        borderColor: colors.map(c => c.replace("0.7", "1")),
        borderWidth: 2,
        borderRadius: 8,
      }]
    },
    options: {
      indexAxis: "y",
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        x: {
          min: 0, max: 1,
          title: { display: true, text: "F1-Score", color: "#8aabb5" },
          grid: { color: "rgba(255,255,255,0.04)" }
        },
        y: { grid: { display: false }, ticks: { font: { size: 11 } } }
      }
    },
    plugins: [{
      id: "f1labels",
      afterDatasetsDraw(chart) {
        const ctx2 = chart.ctx;
        chart.data.datasets[0].data.forEach((val, i) => {
          const meta = chart.getDatasetMeta(0);
          const bar = meta.data[i];
          ctx2.fillStyle = "#e8f4ff";
          ctx2.font = "bold 12px Inter";
          ctx2.textAlign = "left";
          ctx2.fillText(val.toFixed(4), bar.x + 6, bar.y + 5);
        });
      }
    }]
  });
}

// ─── USERS TABLE ─────────────────────────────────────────
function renderUsersTable(data) {
  const tbody = document.getElementById("users-tbody");
  const users = data.top_risky_users;

  if (!users || users.length === 0) {
    tbody.innerHTML = `<tr><td colspan="6" class="loading-cell">Veri bulunamadı. main.py'yi çalıştırın.</td></tr>`;
    return;
  }

  tbody.innerHTML = users.map((u, i) => {
    const rate = u.anomaly_rate || 0;
    const [cls, label] = getRiskLevel(rate);
    return `
      <tr>
        <td>${i + 1}</td>
        <td><strong>${u.user_id}</strong></td>
        <td>${(rate * 100).toFixed(2)}%</td>
        <td>${Number(u.mean_volume).toFixed(1)} MB</td>
        <td>${u.anomaly_count}</td>
        <td><span class="risk-badge ${cls}">${label}</span></td>
      </tr>
    `;
  }).join("");
}

// ─── MAIN ─────────────────────────────────────────────────
async function init() {
  const data = await loadData();

  renderKPIs(data);
  renderMetrics(data);
  renderTimeline(data);
  renderHourly(data);
  renderFPComparison(data);
  renderF1Chart(data);
  renderUsersTable(data);

  // Status badge
  const statusBadge = document.getElementById("status-badge");
  if (statusBadge) {
    statusBadge.textContent = "● Analiz Tamamlandı";
  }

  console.log("[✓] Dashboard başarıyla yüklendi");
}

document.addEventListener("DOMContentLoaded", init);
