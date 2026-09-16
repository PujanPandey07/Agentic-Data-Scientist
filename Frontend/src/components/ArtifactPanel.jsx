import { useEffect, useState } from "react";
import ReactMarkdown from "react-markdown";
import axiosInstance from "../api/axiosstance";
import jsPDF from "jspdf";
import { formatReportAsMarkdown } from "../utils/FormatReport";

const API_BASE = "http://127.0.0.1:8000";

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

function downloadBlob(content, filename, type) {
  const blob = new Blob([content], { type });
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  window.URL.revokeObjectURL(url);
}

function ArtifactPanel({ threadId, onClose }) {
  const [report, setReport] = useState(null);
  const [charts, setCharts] = useState([]);
  const [tab, setTab] = useState("report");
  const [errorMsg, setErrorMsg] = useState("");
  const [loading, setLoading] = useState(true);
  const [pdfLoading, setPdfLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    setLoading(true);
    setErrorMsg("");
    Promise.all([
      axiosInstance.get(`/api/runs/${threadId}/report`),
      axiosInstance.get(`/api/runs/${threadId}/charts`),
    ])
      .then(([r, c]) => {
        setReport(r.data.report);
        setCharts(c.data.charts);
      })
      .catch((error) => setErrorMsg(error.response?.data?.detail || "Could not load report."))
      .finally(() => setLoading(false));
  }, [threadId]);

  const markdown = report ? formatReportAsMarkdown(report, threadId) : "";

  async function downloadFile(endpoint, filename) {
    try {
      const response = await axiosInstance.get(endpoint, { responseType: "blob" });
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement("a");
      link.href = url;
      link.download = filename;
      link.click();
      window.URL.revokeObjectURL(url);
    } catch {
      setErrorMsg("Download failed.");
    }
  }

  function copyMarkdown() {
    navigator.clipboard.writeText(markdown);
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  async function downloadPdf() {
    setPdfLoading(true);
    try {
      const doc = new jsPDF();
      let y = 20;
      doc.setFontSize(16);
      doc.text("AI Data Scientist — Report", 10, y);
      y += 10;
      doc.setFontSize(10);
      doc.text(`Run: ${threadId}`, 10, y);
      y += 10;

      if (report.evaluation) {
        doc.setFontSize(12);
        doc.text("Evaluation", 10, y);
        y += 8;
        doc.setFontSize(9);
        for (const [key, value] of Object.entries(report.evaluation)) {
          if (typeof value === "object") continue;
          doc.text(`${key}: ${value}`, 10, y);
          y += 6;
          if (y > 270) { doc.addPage(); y = 20; }
        }
      }

      for (const chart of charts) {
        if (y > 200) { doc.addPage(); y = 20; }
        try {
          const { dataUrl, format } = await imageUrlToDataUrl(`${API_BASE}${chart.url}`);
          doc.addImage(dataUrl, format, 10, y, 180, 100);
          y += 110;
        } catch {
          // skip a chart that fails to convert rather than aborting the whole PDF
        }
      }

      doc.save(`report-${threadId}.pdf`);
    } finally {
      setPdfLoading(false);
    }
  }

  const modelLikelyTrained = Boolean(report?.evaluation);

  return (
    <div className="w-[440px] shrink-0 border-l border-muted/20 bg-white h-screen flex flex-col">
      <div className="flex items-center justify-between px-4 py-3 border-b border-muted/20">
        <p className="font-serif text-ink">Report</p>
        <button onClick={onClose} className="text-muted hover:text-ink text-sm">✕</button>
      </div>

      <div className="flex gap-4 px-4 pt-3 text-sm border-b border-muted/20">
        {["report", "charts"].map((t) => (
          <button
            key={t}
            onClick={() => setTab(t)}
            className={`pb-2 capitalize ${tab === t ? "text-ink border-b-2 border-accent" : "text-muted"}`}
          >
            {t}
          </button>
        ))}
      </div>

      <div className="flex-1 overflow-y-auto p-5">
        {loading && <p className="text-muted text-sm">Loading...</p>}
        {errorMsg && <p className="text-clay text-sm">{errorMsg}</p>}

        {!loading && !errorMsg && tab === "report" && report && (
          <article className="prose prose-sm max-w-none prose-headings:font-serif prose-headings:text-ink prose-p:text-ink prose-li:text-ink prose-strong:text-ink">
            <ReactMarkdown>{markdown}</ReactMarkdown>
          </article>
        )}

        {!loading && !errorMsg && tab === "charts" && (
          <div className="grid grid-cols-1 gap-3">
            {charts.map((chart, i) => (
              <div key={i} className="bg-paper rounded-md p-2 border border-muted/20">
                <img src={`${API_BASE}${chart.url}`} alt={chart.chart_type} className="w-full rounded" />
                <p className="text-xs text-muted mt-1">{chart.chart_type}</p>
              </div>
            ))}
            {charts.length === 0 && <p className="text-muted text-sm">No charts.</p>}
          </div>
        )}
      </div>

      {!loading && !errorMsg && tab === "report" && (
        <div className="border-t border-muted/20 p-3 flex flex-wrap gap-2">
          <button onClick={copyMarkdown}
            className="text-xs bg-panel text-paper px-3 py-1.5 rounded-md hover:bg-panel/80">
            {copied ? "Copied!" : "Copy"}
          </button>
          <button onClick={() => downloadBlob(markdown, `report-${threadId}.md`, "text/markdown")}
            className="text-xs bg-ink text-paper px-3 py-1.5 rounded-md hover:bg-panel">
            Markdown
          </button>
          <button onClick={downloadPdf} disabled={pdfLoading}
            className="text-xs bg-ink text-paper px-3 py-1.5 rounded-md hover:bg-panel disabled:opacity-50">
            {pdfLoading ? "..." : "PDF"}
          </button>
          <button onClick={() => downloadFile(`/api/runs/${threadId}/model/download`, "model.pkl")}
            disabled={!modelLikelyTrained}
            title={!modelLikelyTrained ? "No trained model for this run" : ""}
            className="text-xs bg-ink text-paper px-3 py-1.5 rounded-md hover:bg-panel disabled:opacity-40 disabled:cursor-not-allowed">
            Model
          </button>
        </div>
      )}
    </div>
  );
}

export default ArtifactPanel;