/* İç Tehdit Tespit Sistemi — canlı dashboard */
Chart.defaults.color = "#8aabb5";
Chart.defaults.borderColor = "rgba(30,58,90,.5)";
Chart.defaults.font.family = "Inter, system-ui, sans-serif";

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 4) => v == null || v === "N/A" || !Number.isFinite(Number(v)) ? "N/A" : Number(v).toFixed(d);
const pct = (v) => v == null || !Number.isFinite(Number(v)) ? "N/A" : `${(Number(v) * 100).toFixed(2)}%`;
const safe = (v, fallback = "—") => v == null || v === "" ? fallback : String(v);
const setText = (id, value) => { const el = $(id); if (el) el.textContent = value; };
const setBar = (id, value) => {
  const el = $(id);
  if (el) el.style.width = `${Math.max(0, Math.min(1, Number(value) || 0)) * 100}%`;
};

async function loadData() {
  const response = await fetch(`data.json?v=${Date.now()}`, { cache: "no-store" });
  if (!response.ok) throw new Error(`data.json yüklenemedi: HTTP ${response.status}`);
  return response.json();
}

function metricCard(key, title, color, metrics, badge) {
  const rows = [
    ["Precision", "precision", false],
    ["Recall", "recall", false],
    ["F1-Score", "f1", false],
    ["ROC-AUC", "roc_auc", false],
    ["Yanlış Alarm (FPR)", "fpr", true]
  ];
  const body = rows.map(([label, field, isFpr]) => {
    const value = metrics?.[field];
    const display = isFpr ? pct(value) : fmt(value);
    const width = Number.isFinite(Number(value)) ? Math.max(0, Math.min(1, Number(value))) * 100 : 0;
    return `<div class="metric-row ${isFpr ? "metric-fpr" : ""}">
      <span class="metric-label">${label}</span>
      <div class="metric-bar-wrap"><div class="metric-bar" style="width:${width}%;background:${color}"></div></div>
      <span class="metric-val">${display}</span>
    </div>`;
  }).join("");
  return `<article class="metrics-card" style="border-color:${color}">
    <div class="metrics-header"><span class="metrics-model-name">${title}</span><span class="metrics-badge">${badge}</span></div>
    <div class="metrics-body">${body}</div>
  </article>`;
}

function renderKPIs(data) {
  const s = data.summary || {};
  setText("kpi-users", Number(s.total_users || 0).toLocaleString("tr-TR"));
  setText("kpi-anomalies", Number(s.total_anomalies || 0).toLocaleString("tr-TR"));
  setText("kpi-rate", `${(Number(s.anomaly_rate || 0) * 100).toFixed(2)}%`);
  const reports = data.threat_reports || [];
  const users = data.top_risky_users || [];
  const top = [...reports].sort((a,b) => Number(b.risk_score || 0) - Number(a.risk_score || 0))[0];
  const topUser = top?.user_id || users[0]?.user_id;
  const riskValue = $("kpi-malicious");
  if (riskValue) {
    riskValue.textContent = topUser
      ? `${topUser} · ${top?.risk_score ?? "—"}/100`
      : safe(s.malicious_users_detected, "Bulunamadı");
    riskValue.style.display = "block";
    riskValue.style.visibility = "visible";
    riskValue.style.fontSize = "clamp(17px, 2vw, 25px)";
    riskValue.style.lineHeight = "1.25";
    riskValue.style.overflowWrap = "anywhere";
  }
}

function renderMetrics(data) {
  const grid = $("metrics-grid");
  if (!grid) return;
  const models = [
    ["if_metrics", "🌲 Isolation Forest", "#00aaff", "Unsupervised"],
    ["ae_metrics", "🧠 Autoencoder", "#9b59ff", "Deep Learning"],
    ["hybrid_metrics", "⚡ Hybrid (IF + AE)", "#16c79a", "Ensemble"],
    ["lstm_metrics", "⏱ LSTM", "#ff9f43", "Temporal AI"]
  ];
  grid.innerHTML = models.map(([key, title, color, badge]) =>
    metricCard(key, title, color, data[key] || {}, badge)
  ).join("");
}

function makeChart(canvasId, config) {
  const canvas = $(canvasId);
  if (!canvas) return;
  if (canvas._chart) canvas._chart.destroy();
  canvas._chart = new Chart(canvas, config);
}

function renderTimeline(data) {
  const tl = data.timeline || {};
  const ifScores = tl.if_scores || [];
  const aeScores = tl.ae_scores || [];
  const labels = ifScores.map((_, i) => `Kayıt ${i + 1}`);
  const maxAE = Math.max(1e-12, ...aeScores.map(Number));
  const aeNorm = aeScores.map(v => Number(v) / maxAE);
  makeChart("timelineChart", {
    type: "line",
    data: { labels, datasets: [
      { label: "Isolation Forest skoru", data: ifScores, borderColor: "#00aaff", pointRadius: 0, borderWidth: 1.5, tension: .2 },
      { label: "Autoencoder hatası (normalize)", data: aeNorm, borderColor: "#9b59ff", pointRadius: 0, borderWidth: 1.5, tension: .2 },
      { label: "Gerçek anomali", data: (tl.true_labels || []).map((v,i) => Number(v) === 1 ? ifScores[i] : null), type: "scatter", pointRadius: 4, pointStyle: "triangle", backgroundColor: "#ff4444" }
    ]},
    options: { responsive: true, maintainAspectRatio: false, interaction: { mode: "index", intersect: false } }
  });
}

function renderHourly(data) {
  const rows = data.hourly_activity || [];
  makeChart("hourlyChart", {
    type: "bar",
    data: { labels: rows.map(x => `${String(x.hour_of_day).padStart(2,"0")}:00`), datasets: [{
      label: "Anomali oranı (%)",
      data: rows.map(x => Number(x.rate || 0) * 100),
      backgroundColor: rows.map(x => Number(x.hour_of_day) >= 9 && Number(x.hour_of_day) <= 18 ? "rgba(0,230,118,.55)" : "rgba(255,68,68,.55)"),
      borderRadius: 4
    }]},
    options: { responsive: true, maintainAspectRatio: false, scales: { y: { beginAtZero: true, title: { display: true, text: "Anomali oranı (%)" } } } }
  });
}

function normalizeComparison(data) {
  const rows = data.model_comparison || [];
  const names = ["Kural Tabanlı (SIEM)", "Isolation Forest (AI)", "Autoencoder (AI)", "Hibrit Model (IF + AE)", "LSTM (Zamansal AI)"];
  const modelKeys = ["if_metrics", "ae_metrics", "hybrid_metrics", "lstm_metrics"];
  const result = [];
  // SIEM satırı varsa onu koru.
  const siem = rows.find(r => String(r["Yöntem"] || r["YÃ¶ntem"] || "").toLowerCase().includes("siem"));
  if (siem) result.push({
    name: siem["Yöntem"] || siem["YÃ¶ntem"] || names[0],
    fpr: Number(siem["False Positive Rate"]),
    f1: Number(siem["F1-Score"])
  });
  modelKeys.forEach((key, i) => {
    const m = data[key] || {};
    result.push({ name: names[i + 1], fpr: Number(m.fpr), f1: Number(m.f1) });
  });
  return result;
}

function renderComparisons(data) {
  const rows = normalizeComparison(data);
  const labels = rows.map(x => x.name);
  const colors = ["#ff9800", "#00aaff", "#9b59ff", "#16c79a", "#ff9f43"];
  makeChart("fpCompareChart", {
    type: "bar",
    data: { labels, datasets: [{ label: "Yanlış alarm oranı (%)", data: rows.map(x => Number.isFinite(x.fpr) ? x.fpr * 100 : 0), backgroundColor: colors, borderRadius: 7 }] },
    options: { responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { y: { beginAtZero: true, title: { display: true, text: "FPR (%)" } }, x: { ticks: { maxRotation: 25, minRotation: 10 } } } }
  });
  makeChart("f1Chart", {
    type: "bar",
    data: { labels: rows.filter(x => Number.isFinite(x.f1)).map(x => x.name), datasets: [{ label: "F1-Score", data: rows.filter(x => Number.isFinite(x.f1)).map(x => x.f1), backgroundColor: colors, borderRadius: 7 }] },
    options: { indexAxis: "y", responsive: true, maintainAspectRatio: false, plugins: { legend: { display: false } }, scales: { x: { min: 0, max: 1 } } }
  });
}

function riskClass(level, score) {
  const value = String(level || "").toLocaleLowerCase("tr-TR");
  if (value.includes("kritik") || Number(score) >= 80) return "risk-critical";
  if (value.includes("yüksek") || Number(score) >= 60) return "risk-high";
  if (value.includes("orta") || Number(score) >= 30) return "risk-medium";
  return "risk-low";
}

function renderUsers(data) {
  const tbody = $("users-tbody");
  if (!tbody) return;
  const users = data.top_risky_users || [];
  if (!users.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="loading-cell">Riskli kullanıcı verisi bulunamadı.</td></tr>`;
    return;
  }
  tbody.innerHTML = users.map((u, i) => {
    const rate = Number(u.anomaly_rate || 0);
    const cls = rate > .1 ? "risk-critical" : rate > .05 ? "risk-high" : rate > .02 ? "risk-medium" : "risk-low";
    return `<tr><td>${i+1}</td><td><strong>${safe(u.user_id)}</strong></td><td>${(rate*100).toFixed(2)}%</td><td>${Number(u.mean_volume || 0).toFixed(1)} MB</td><td>${safe(u.anomaly_count, 0)}</td><td><span class="risk-badge ${cls}">${cls.replace("risk-","").toLocaleUpperCase("tr-TR")}</span></td></tr>`;
  }).join("");
}

function renderThreatReports(data) {
  const tbody = $("threat-reports-body");
  const count = $("threat-report-count");
  if (!tbody) return;
  const reports = [...(data.threat_reports || [])].sort((a,b) => Number(b.risk_score || 0) - Number(a.risk_score || 0));
  if (count) count.textContent = `${reports.length} rapor`;
  if (!reports.length) {
    tbody.innerHTML = `<tr><td colspan="6" class="loading-cell">Tehdit raporu bulunamadı. Önce main.py çalıştırılmalı.</td></tr>`;
    return;
  }
  tbody.innerHTML = reports.map(r => {
    const cls = riskClass(r.risk_level, r.risk_score);
    const detected = Array.isArray(r.detected_by) ? r.detected_by.join(", ") : safe(r.detected_by);
    const summary = r.explanation?.summary || r.explanation?.explanation || r.explanation?.reason || "Davranışsal anomali tespit edildi.";
    const evidence = r.explanation?.evidence || r.risk_assessment?.evidence || [];
    const actions = r.explanation?.recommended_actions || r.risk_assessment?.recommended_actions || [];
    const timestamp = formatTimestamp(r.timestamp);
    const evidenceHtml = evidence.length
      ? `<h4>Risk göstergeleri</h4><ul>${evidence.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ul>`
      : "";
    const actionsHtml = actions.length
      ? `<h4>Önerilen aksiyonlar</h4><ol>${actions.map(item => `<li>${escapeHtml(item)}</li>`).join("")}</ol>`
      : "";
    return `<tr>
      <td><strong>${escapeHtml(safe(r.user_id))}</strong></td>
      <td class="report-date">${escapeHtml(timestamp)}</td>
      <td><span class="risk-badge ${cls}">${escapeHtml(safe(r.risk_level, "Belirsiz"))}</span></td>
      <td><strong>${escapeHtml(safe(r.risk_score, 0))}/100</strong></td>
      <td>${escapeHtml(detected)}</td>
      <td class="report-details"><details><summary>Raporu aç</summary><div class="report-detail-body"><p>${escapeHtml(summary)}</p>${evidenceHtml}${actionsHtml}</div></details></td>
    </tr>`;
  }).join("");
}

function formatTimestamp(value) {
  if (!value) return "—";
  // ISO biçimindeki saat dilimsiz veri değerini yerel saat gibi kaydırmadan göster.
  const match = String(value).match(/^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2})(?::(\d{2}))?/);
  if (!match) return String(value);
  return `${match[3]}.${match[2]}.${match[1]} ${match[4]}:${match[5]}`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, c => ({ "&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;" }[c]));
}

function renderSource(data) {
  const s = data.summary || {};
  setText("source-records", `${Number(s.total_records || 0).toLocaleString("tr-TR")} kayıt`);
}

async function init() {
  try {
    const data = await loadData();
    renderKPIs(data);
    renderMetrics(data);
    renderTimeline(data);
    renderHourly(data);
    renderComparisons(data);
    renderUsers(data);
    renderThreatReports(data);
    renderSource(data);
    setText("status-badge", "● Analiz Tamamlandı");
    console.log("[✓] Gerçek dashboard verisi yüklendi", data);
  } catch (error) {
    console.error(error);
    setText("status-badge", "● Veri yüklenemedi");
    const target = $("dashboard-error");
    if (target) target.textContent = `${error.message}. Dashboard klasöründe python -m http.server 8000 komutuyla sunucuyu başlatın.`;
  }
}
document.addEventListener("DOMContentLoaded", init);
