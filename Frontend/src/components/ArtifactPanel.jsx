import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import axiosInstance, { API_BASE } from "../api/axiosstance";
import { cleanMarkdown, formatReportAsMarkdown } from "../utils/FormatReport";

// ─── helpers ─────────────────────────────────────────────────────────────────

function downloadBlob(content, filename, type) {
  const blob = new Blob([content], { type });
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url; link.download = filename; link.click();
  window.URL.revokeObjectURL(url);
}

async function imageUrlToDataUrl(url) {
  const res = await fetch(url);
  const blob = await res.blob();
  const format = blob.type.includes("png") ? "PNG" : "JPEG";
  const dataUrl = await new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onloadend = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsDataURL(blob);
  });
  return { dataUrl, format };
}

function fmt(val, isPercent = false) {
  if (val === null || val === undefined) return "—";
  if (typeof val === "number") {
    if (isPercent) return `${(val * 100).toFixed(1)}%`;
    return val % 1 === 0 ? String(val) : val.toFixed(4);
  }
  return String(val);
}

function fmtMape(val) {
  if (val === null || val === undefined) return "—";
  return `${Number(val).toFixed(1)}%`;
}

function getProblemConfig(type) {
  const configs = {
    classification: { label: "Classification", bg: "#e0f2fe", color: "#0369a1" },
    regression:     { label: "Regression",     bg: "#f0fdf4", color: "#15803d" },
    clustering:     { label: "Clustering",     bg: "#fdf4ff", color: "#7e22ce" },
    time_series:    { label: "Time Series",    bg: "#fff7ed", color: "#c2410c" },
  };
  return configs[type] || { label: type || "Analysis", bg: "#f3f4f6", color: "#374151" };
}

function getKeyMetric(conclusions, type) {
  if (!conclusions) return null;
  if (type === "classification") {
    const acc = conclusions.final_accuracy;
    if (acc !== null && acc !== undefined)
      return { label: "Accuracy", value: `${(acc * 100).toFixed(1)}%` };
  }
  if (type === "regression") {
    const r2 = conclusions.final_r2;
    if (r2 !== null && r2 !== undefined)
      return { label: "R² Score", value: r2.toFixed(3) };
  }
  if (type === "clustering") {
    const s = conclusions.final_silhouette_score;
    if (s !== null && s !== undefined)
      return { label: "Silhouette", value: s.toFixed(3) };
  }
  if (type === "time_series") {
    const mape = conclusions.mape;
    if (mape !== null && mape !== undefined)
      return { label: "MAPE", value: `${Number(mape).toFixed(1)}%` };
    const mae = conclusions.mae;
    if (mae !== null && mae !== undefined)
      return { label: "MAE", value: Number(mae).toFixed(4) };
  }
  return null;
}

function getQualityColor(metric, value) {
  if (metric === "MAPE") {
    if (value < 5) return "#15803d";
    if (value < 10) return "#ca8a04";
    if (value < 20) return "#ea580c";
    return "#dc2626";
  }
  if (metric === "Accuracy" || metric === "Silhouette" || metric === "R² Score") {
    if (value >= 0.9) return "#15803d";
    if (value >= 0.7) return "#ca8a04";
    return "#dc2626";
  }
  return "#1a1a1a";
}

// ─── sub-components ───────────────────────────────────────────────────────────

function StatTile({ label, value, accent }) {
  return (
    <div className="rounded-xl px-4 py-3 flex flex-col gap-1"
      style={{ background: "#fff7ed", border: "1px solid #fde8d0" }}>
      <span className="text-xs font-medium" style={{ color: "#888" }}>{label}</span>
      <span className="text-lg font-bold leading-tight" style={{ color: accent || "#cc785c" }}>
        {value}
      </span>
    </div>
  );
}

function SectionCard({ icon, title, status, children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  const skipped = status === "skipped" || !children;

  return (
    <div className="rounded-xl border overflow-hidden" style={{ borderColor: "#e8e8e8" }}>
      <button
        onClick={() => !skipped && setOpen(!open)}
        className="w-full flex items-center justify-between px-4 py-3 text-left transition-colors"
        style={{
          background: skipped ? "#fafafa" : open ? "#fff7f4" : "white",
          cursor: skipped ? "default" : "pointer",
        }}
      >
        <div className="flex items-center gap-2.5">
          <span className="text-base">{icon}</span>
          <span className="text-sm font-medium" style={{ color: skipped ? "#bbb" : "#1a1a1a" }}>
            {title}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs px-2 py-0.5 rounded-full"
            style={{
              background: skipped ? "#f5f5f5" : "#ecfdf5",
              color: skipped ? "#aaa" : "#15803d",
              border: `1px solid ${skipped ? "#e5e5e5" : "#bbf7d0"}`,
            }}>
            {skipped ? "Skipped" : "Done"}
          </span>
          {!skipped && (
            <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" style={{
              color: "#aaa", transform: open ? "rotate(180deg)" : "none", transition: "transform 0.2s"
            }} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={2}>
              <polyline points="6 9 12 15 18 9" />
            </svg>
          )}
        </div>
      </button>

      {open && !skipped && (
        <div className="px-4 py-3 text-sm border-t" style={{ borderColor: "#f0f0f0", background: "white" }}>
          {children}
        </div>
      )}
    </div>
  );
}

function KVTable({ data }) {
  if (!data || typeof data !== "object") return null;
  const entries = Object.entries(data).filter(([, v]) => v !== null && v !== undefined && typeof v !== "object");
  if (!entries.length) return null;
  return (
    <table className="w-full text-xs" style={{ borderCollapse: "collapse" }}>
      <tbody>
        {entries.map(([k, v]) => (
          <tr key={k} style={{ borderBottom: "1px solid #f5f5f5" }}>
            <td className="py-1.5 pr-4 font-medium" style={{ color: "#666", width: "40%" }}>
              {k.replace(/_/g, " ")}
            </td>
            <td className="py-1.5" style={{ color: "#1a1a1a" }}>{String(v)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

function MetricsBlock({ metrics }) {
  if (!metrics || !Object.keys(metrics).length) return null;
  return (
    <div className="grid grid-cols-2 gap-2 mt-2">
      {Object.entries(metrics).filter(([, v]) => v !== null && v !== undefined).map(([k, v]) => (
        <div key={k} className="rounded-lg px-3 py-2"
          style={{ background: "#f8f8f7", border: "1px solid #ebebeb" }}>
          <p className="text-xs" style={{ color: "#888" }}>{k.replace(/_/g, " ")}</p>
          <p className="text-sm font-semibold" style={{ color: "#1a1a1a" }}>
            {typeof v === "number" ? v.toFixed(4) : String(v)}
          </p>
        </div>
      ))}
    </div>
  );
}

function PipelineSteps({ tasks }) {
  if (!tasks?.length) return null;
  const labels = {
    cleaning: "Clean", eda: "EDA", ts_analysis: "TS Analysis",
    visualization: "Charts", feature_engineering: "Features",
    model_selection: "Train", hyperparameter_tuning: "Tune",
    evaluation: "Evaluate", reporting: "Report",
  };
  return (
    <div className="flex items-center gap-1 flex-wrap">
      {tasks.map((t, i) => (
        <div key={t} className="flex items-center gap-1">
          <div className="flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium"
            style={{ background: "#ecfdf5", color: "#15803d", border: "1px solid #bbf7d0" }}>
            <span style={{ width: 6, height: 6, borderRadius: "50%", background: "#22c55e", display: "inline-block" }} />
            {labels[t] || t}
          </div>
          {i < tasks.length - 1 && (
            <span style={{ color: "#ccc", fontSize: 10 }}>→</span>
          )}
        </div>
      ))}
    </div>
  );
}

// ─── main component ───────────────────────────────────────────────────────────

function ArtifactPanel({ threadId, onClose }) {
  const [report, setReport] = useState(null);
  const [charts, setCharts] = useState([]);
  const [tab, setTab] = useState("summary");
  const [errorMsg, setErrorMsg] = useState("");
  const [loading, setLoading] = useState(true);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [chartIdx, setChartIdx] = useState(0);
  const summaryRef = useRef(null);

  useEffect(() => {
    Promise.all([
      axiosInstance.get(`/api/runs/${threadId}/report`),
      axiosInstance.get(`/api/runs/${threadId}/charts`),
    ])
      .then(([r, c]) => { setReport(r.data.report); setCharts(c.data.charts); })
      .catch((err) => setErrorMsg(err.response?.data?.detail || "Could not load report."))
      .finally(() => setLoading(false));
  }, [threadId]);

  const markdown = report ? formatReportAsMarkdown(report, threadId) : "";
  const type = report?.problem_type;
  const typeConfig = getProblemConfig(type);
  const conclusions = report?.conclusions || {};
  const keyMetric = getKeyMetric(conclusions, type);
  const completed = report?.pipeline_status?.completed_tasks || [];
  const overview = report?.dataset_overview || {};
  const modelLikelyTrained = Boolean(report?.evaluation || report?.ts_evaluation_report);

  async function downloadFile(endpoint, filename) {
    try {
      const res = await axiosInstance.get(endpoint, { responseType: "blob" });
      const url = window.URL.createObjectURL(new Blob([res.data]));
      const link = document.createElement("a");
      link.href = url; link.download = filename; link.click();
      window.URL.revokeObjectURL(url);
    } catch { setErrorMsg("Download failed."); }
  }

  function copyMarkdown() {
    navigator.clipboard.writeText(markdown);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  async function downloadPdf() {
    setPdfLoading(true);
    try {
      const { default: JsPDF } = await import("jspdf");
      const doc = new JsPDF();
      let y = 20;
      const plain = markdown.replace(/[#*`|]/g, "");
      const lines = doc.splitTextToSize(plain, 180);
      doc.setFontSize(9);
      for (const line of lines) {
        if (y > 280) { doc.addPage(); y = 20; }
        doc.text(line, 10, y); y += 5;
      }
      for (const chart of charts) {
        doc.addPage(); y = 20;
        try {
          const { dataUrl, format } = await imageUrlToDataUrl(`${API_BASE}${chart.url}`);
          doc.addImage(dataUrl, format, 10, y, 180, 100);
        } catch { /* skip */ }
      }
      doc.save(`report-${threadId}.pdf`);
    } finally { setPdfLoading(false); }
  }

  // ── render ──────────────────────────────────────────────────────────────

  return (
    <div className="shrink-0 flex flex-col h-full"
      style={{ width: 480, background: "white", borderLeft: "1px solid #e8e8e8" }}>

      {/* Header */}
      <div className="shrink-0 flex items-center justify-between px-4 py-3"
        style={{ borderBottom: "1px solid #e8e8e8" }}>
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold" style={{ color: "#1a1a1a" }}>Analysis Report</span>
          {report && (
            <span className="text-xs px-2 py-0.5 rounded-full font-medium"
              style={{ background: typeConfig.bg, color: typeConfig.color }}>
              {typeConfig.label}
            </span>
          )}
        </div>
        <button onClick={onClose}
          className="w-7 h-7 rounded-lg flex items-center justify-center transition-colors"
          style={{ color: "#888" }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "#f5f5f5")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "transparent")}>
          <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 24 24"
            fill="none" stroke="currentColor" strokeWidth={2.5}>
            <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>
      </div>

      {/* Tabs */}
      <div className="shrink-0 flex gap-0 px-4" style={{ borderBottom: "1px solid #e8e8e8" }}>
        {["summary", "details", "charts"].map((t) => (
          <button key={t} onClick={() => setTab(t)}
            className="px-3 py-2.5 text-xs font-medium capitalize transition-colors"
            style={{
              color: tab === t ? "#cc785c" : "#888",
              borderBottom: tab === t ? "2px solid #cc785c" : "2px solid transparent",
            }}>
            {t === "charts" ? `Charts${charts.length ? ` (${charts.length})` : ""}` : t}
          </button>
        ))}
      </div>

      {/* Body */}
      <div className="flex-1 overflow-y-auto">
        {loading && (
          <div className="flex items-center justify-center h-32">
            <div className="flex gap-1.5">
              {[0,1,2].map(i => (
                <span key={i} className="w-2 h-2 rounded-full"
                  style={{ background: "#ddd", animation: `bounce 1.2s ease-in-out ${i*0.2}s infinite` }} />
              ))}
            </div>
          </div>
        )}
        {errorMsg && (
          <p className="text-sm m-4 p-3 rounded-lg"
            style={{ background: "#fef2f2", color: "#dc2626", border: "1px solid #fecaca" }}>
            {errorMsg}
          </p>
        )}

        {/* ── SUMMARY TAB ── */}
        {!loading && !errorMsg && tab === "summary" && report && (
          <div className="p-4 space-y-4" ref={summaryRef}>

            {/* Hero: key metric + recommendation */}
            <div className="rounded-2xl p-4 space-y-3"
              style={{ background: "linear-gradient(135deg, #fff7f4 0%, #fffbf9 100%)", border: "1px solid #fde8d0" }}>
              {keyMetric && (
                <div className="flex items-baseline gap-2">
                  <span className="text-4xl font-bold"
                    style={{ color: getQualityColor(keyMetric.label, parseFloat(keyMetric.value)) }}>
                    {keyMetric.value}
                  </span>
                  <span className="text-sm" style={{ color: "#888" }}>{keyMetric.label}</span>
                </div>
              )}
              <p className="text-sm leading-relaxed" style={{ color: "#444" }}>
                {conclusions.recommendation || "Analysis complete."}
              </p>
              {overview.final_shape && (
                <p className="text-xs" style={{ color: "#aaa" }}>
                  Dataset: {overview.final_shape[0].toLocaleString()} rows × {overview.final_shape[1]} columns
                </p>
              )}
            </div>

            {/* Metric tiles */}
            {type === "classification" && (
              <div className="grid grid-cols-2 gap-2">
                <StatTile label="Accuracy" value={fmt(conclusions.final_accuracy, true)} />
                <StatTile label="F1 Score" value={fmt(conclusions.final_f1, true)} />
                <StatTile label="Best Model" value={conclusions.best_model || "—"} accent="#1a1a1a" />
                <StatTile label="CV Score" value={fmt(conclusions.cv_score, true)} />
              </div>
            )}
            {type === "regression" && (
              <div className="grid grid-cols-2 gap-2">
                <StatTile label="R² Score" value={fmt(conclusions.final_r2)} />
                <StatTile label="Best Model" value={conclusions.best_model || "—"} accent="#1a1a1a" />
                <StatTile label="CV Score" value={fmt(conclusions.cv_score, true)} />
              </div>
            )}
            {type === "clustering" && (
              <div className="grid grid-cols-2 gap-2">
                <StatTile label="Clusters Found" value={fmt(conclusions.n_clusters)} />
                <StatTile label="Silhouette" value={fmt(conclusions.final_silhouette_score)} />
                <StatTile label="Algorithm" value={conclusions.best_algorithm || "—"} accent="#1a1a1a" />
                <StatTile label="Noise Points" value={fmt(conclusions.n_noise_points)} />
              </div>
            )}
            {type === "time_series" && (
              <div className="grid grid-cols-2 gap-2">
                <StatTile label="MAPE" value={fmtMape(conclusions.mape)} />
                <StatTile label="MAE" value={fmt(conclusions.mae)} />
                <StatTile label="RMSE" value={fmt(conclusions.rmse)} />
                <StatTile label="Model" value={conclusions.best_model || "—"} accent="#1a1a1a" />
              </div>
            )}

            {/* Pipeline steps */}
            {completed.length > 0 && (
              <div>
                <p className="text-xs font-medium mb-2" style={{ color: "#888" }}>PIPELINE COMPLETED</p>
                <PipelineSteps tasks={completed} />
              </div>
            )}

            {/* TS analysis summary */}
            {type === "time_series" && report.ts_analysis && (
              <div className="rounded-xl p-3 text-sm space-y-1"
                style={{ background: "#f8f8f7", border: "1px solid #e8e8e8" }}>
                <p className="font-medium text-xs mb-2" style={{ color: "#888" }}>TIME SERIES PROPERTIES</p>
                <KVTable data={{
                  time_column: report.ts_analysis.time_column,
                  target_column: report.ts_analysis.target_column,
                  frequency: report.ts_analysis.frequency,
                  stationary: report.ts_analysis.is_stationary ? "Yes" : "No (differenced)",
                  trend: report.ts_analysis.trend_detected ? "Detected" : "None",
                  seasonality: report.ts_analysis.seasonality_detected ? "Detected" : "None",
                }} />
              </div>
            )}
          </div>
        )}

        {/* ── DETAILS TAB ── */}
        {!loading && !errorMsg && tab === "details" && report && (
          <div className="p-4 space-y-2">
            <SectionCard icon="🧹" title="Data Cleaning" status={report.cleaning ? "done" : "skipped"} defaultOpen>
              {report.cleaning && <KVTable data={report.cleaning} />}
            </SectionCard>
            <SectionCard icon="🔍" title="Exploratory Analysis" status={report.eda ? "done" : "skipped"}>
              {report.eda && <KVTable data={report.eda} />}
            </SectionCard>
            {type === "time_series" && (
              <SectionCard icon="📈" title="Time Series Analysis" status={report.ts_analysis ? "done" : "skipped"}>
                {report.ts_analysis && <KVTable data={report.ts_analysis} />}
              </SectionCard>
            )}
            <SectionCard icon="⚙️" title="Feature Engineering" status={report.feature_engineering ? "done" : "skipped"}>
              {report.feature_engineering && <KVTable data={report.feature_engineering} />}
            </SectionCard>
            <SectionCard icon="🤖" title="Model Training" status={report.training ? "done" : "skipped"}>
              {report.training && (
                <>
                  <KVTable data={report.training} />
                  {report.training.metrics && <MetricsBlock metrics={report.training.metrics} />}
                </>
              )}
            </SectionCard>
            <SectionCard icon="🎛️" title="Hyperparameter Tuning" status={report.hyperparameter_tuning ? "done" : "skipped"}>
              {report.hyperparameter_tuning && <KVTable data={report.hyperparameter_tuning} />}
            </SectionCard>
            <SectionCard icon="📊" title="Evaluation" status={report.evaluation ? "done" : "skipped"}>
              {report.evaluation && (
                <>
                  <KVTable data={report.evaluation} />
                  {report.evaluation.metrics && <MetricsBlock metrics={report.evaluation.metrics} />}
                </>
              )}
            </SectionCard>
          </div>
        )}

        {/* ── CHARTS TAB ── */}
        {!loading && !errorMsg && tab === "charts" && (
          <div className="p-4">
            {charts.length === 0 ? (
              <p className="text-sm text-center py-8" style={{ color: "#aaa" }}>No charts for this run.</p>
            ) : (
              <div className="space-y-3">
                {/* Nav */}
                <div className="flex items-center justify-between">
                  <p className="text-xs font-medium" style={{ color: "#888" }}>
                    {chartIdx + 1} / {charts.length}
                  </p>
                  <div className="flex gap-1.5">
                    <button
                      onClick={() => setChartIdx(Math.max(0, chartIdx - 1))}
                      disabled={chartIdx === 0}
                      className="w-7 h-7 rounded-lg flex items-center justify-center disabled:opacity-30"
                      style={{ background: "#f0f0f0", color: "#555" }}>
                      ‹
                    </button>
                    <button
                      onClick={() => setChartIdx(Math.min(charts.length - 1, chartIdx + 1))}
                      disabled={chartIdx === charts.length - 1}
                      className="w-7 h-7 rounded-lg flex items-center justify-center disabled:opacity-30"
                      style={{ background: "#f0f0f0", color: "#555" }}>
                      ›
                    </button>
                  </div>
                </div>

                {/* Current chart */}
                {charts[chartIdx] && (
                  <div className="rounded-2xl overflow-hidden"
                    style={{ border: "1px solid #e8e8e8", boxShadow: "0 2px 8px rgba(0,0,0,0.05)" }}>
                    <img
                      src={`${API_BASE}${charts[chartIdx].url}`}
                      alt={charts[chartIdx].chart_type}
                      className="w-full"
                      style={{ display: "block" }}
                    />
                    <div className="px-4 py-2.5" style={{ borderTop: "1px solid #f0f0f0" }}>
                      <p className="text-sm font-medium capitalize" style={{ color: "#1a1a1a" }}>
                        {charts[chartIdx].chart_type?.replace(/_/g, " ")}
                      </p>
                    </div>
                  </div>
                )}

                {/* Thumbnail strip */}
                {charts.length > 1 && (
                  <div className="flex gap-2 overflow-x-auto pb-1">
                    {charts.map((c, i) => (
                      <button key={i} onClick={() => setChartIdx(i)}
                        className="shrink-0 rounded-lg overflow-hidden transition-all"
                        style={{
                          width: 72, height: 48,
                          border: `2px solid ${i === chartIdx ? "#cc785c" : "#e8e8e8"}`,
                        }}>
                        <img src={`${API_BASE}${c.url}`} alt={c.chart_type}
                          style={{ width: "100%", height: "100%", objectFit: "cover" }} />
                      </button>
                    ))}
                  </div>
                )}
              </div>
            )}
          </div>
        )}
      </div>

      {/* Footer action bar */}
      {!loading && !errorMsg && (
        <div className="shrink-0 flex items-center gap-2 px-4 py-3 flex-wrap"
          style={{ borderTop: "1px solid #e8e8e8" }}>
          <button onClick={copyMarkdown}
            className="text-xs px-3 py-1.5 rounded-lg transition-colors font-medium"
            style={{ background: "#f0f0f0", color: "#555" }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "#e5e5e5")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "#f0f0f0")}>
            {copied ? "Copied!" : "Copy"}
          </button>
          <button
            onClick={() => downloadBlob(markdown, `report-${threadId}.md`, "text/markdown")}
            className="text-xs px-3 py-1.5 rounded-lg transition-colors font-medium"
            style={{ background: "#f0f0f0", color: "#555" }}
            onMouseEnter={(e) => (e.currentTarget.style.background = "#e5e5e5")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "#f0f0f0")}>
            Markdown
          </button>
          <button onClick={downloadPdf} disabled={pdfLoading}
            className="text-xs px-3 py-1.5 rounded-lg transition-colors font-medium disabled:opacity-50"
            style={{ background: "#cc785c", color: "white" }}
            onMouseEnter={(e) => !pdfLoading && (e.currentTarget.style.background = "#b86647")}
            onMouseLeave={(e) => !pdfLoading && (e.currentTarget.style.background = "#cc785c")}>
            {pdfLoading ? "Exporting…" : "PDF"}
          </button>
          <button
            onClick={() => downloadFile(`/api/runs/${threadId}/model/download`, "model.pkl")}
            disabled={!modelLikelyTrained}
            title={!modelLikelyTrained ? "No trained model for this run" : ""}
            className="text-xs px-3 py-1.5 rounded-lg transition-colors font-medium disabled:opacity-40 disabled:cursor-not-allowed ml-auto"
            style={{ background: "#f0f0f0", color: "#555" }}
            onMouseEnter={(e) => modelLikelyTrained && (e.currentTarget.style.background = "#e5e5e5")}
            onMouseLeave={(e) => (e.currentTarget.style.background = "#f0f0f0")}>
            ↓ Model
          </button>
        </div>
      )}
    </div>
  );
}

export default ArtifactPanel;