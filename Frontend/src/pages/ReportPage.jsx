import { useEffect, useState } from "react";
import { useParams } from "react-router-dom";
import axiosInstance from "../api/axiosstance";
import jsPDF from "jspdf";

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

function ReportPage() {
  const { threadId } = useParams();
  const [report, setReport] = useState(null);
  const [charts, setCharts] = useState([]);
  const [errorMsg, setErrorMsg] = useState("");
  const [loading, setLoading] = useState(true);
  const [pdfLoading, setPdfLoading] = useState(false);

  useEffect(() => {
    async function loadReport() {
      try {
        const [reportRes, chartsRes] = await Promise.all([
          axiosInstance.get(`/api/runs/${threadId}/report`),
          axiosInstance.get(`/api/runs/${threadId}/charts`),
        ]);
        setReport(reportRes.data.report);
        setCharts(chartsRes.data.charts);
      } catch (error) {
        setErrorMsg(error.response?.data?.detail || "Could not load report.");
      } finally {
        setLoading(false);
      }
    }
    loadReport();
  }, [threadId]);

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
          if (typeof value === "object") continue; // skip nested artifact paths
          const line = `${key}: ${value}`;
          doc.text(line, 10, y);
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
          // skip chart if it fails to load/convert — don't break the whole PDF
        }
      }

      doc.save(`report-${threadId}.pdf`);
    } finally {
      setPdfLoading(false);
    }
  }

  if (loading) return <div className="max-w-3xl mx-auto px-6 py-12 text-muted">Loading report...</div>;
  if (errorMsg) return <div className="max-w-3xl mx-auto px-6 py-12 text-clay">{errorMsg}</div>;
  if (!report) return null;

  // Proxy for "was a model actually trained" — the report doesn't expose
  // this directly, so we infer it from evaluation being present. If this
  // ever misfires (evaluation present but no model file), the backend's
  // own 404 on /model/download is still the real safety net.
  const modelLikelyTrained = Boolean(report.evaluation);

  return (
    <div className="max-w-3xl mx-auto px-6 py-8 space-y-8">
      <h1 className="font-serif text-2xl text-ink">Report — {threadId}</h1>

      <div className="flex flex-wrap gap-3">
        <button
          onClick={() => downloadFile(`/api/runs/${threadId}/report/download`, "report.json")}
          className="bg-ink text-paper px-4 py-2 rounded-md text-sm hover:bg-panel"
        >
          Download Report (JSON)
        </button>
        <button
          onClick={downloadPdf}
          disabled={pdfLoading}
          className="bg-ink text-paper px-4 py-2 rounded-md text-sm hover:bg-panel disabled:opacity-50"
        >
          {pdfLoading ? "Generating..." : "Download Report (PDF)"}
        </button>
        <button
          onClick={() => downloadFile(`/api/runs/${threadId}/model/download`, "model.pkl")}
          disabled={!modelLikelyTrained}
          title={!modelLikelyTrained ? "No trained model available for this run" : ""}
          className="bg-ink text-paper px-4 py-2 rounded-md text-sm hover:bg-panel disabled:opacity-40 disabled:cursor-not-allowed"
        >
          Download Model
        </button>
      </div>

      {report.evaluation && (
        <div>
          <h2 className="font-serif text-lg text-ink mb-2">Evaluation</h2>
          <pre className="text-xs text-muted bg-white p-3 rounded-md overflow-auto">
            {JSON.stringify(report.evaluation, null, 2)}
          </pre>
        </div>
      )}

      {charts.length > 0 && (
        <div>
          <h2 className="font-serif text-lg text-ink mb-3">Charts</h2>
          <div className="grid grid-cols-2 gap-4">
            {charts.map((chart, i) => (
              <div key={i} className="bg-white rounded-md p-2 border border-muted/20">
                <img src={`${API_BASE}${chart.url}`} alt={chart.chart_type} className="w-full" />
                <p className="text-xs text-muted mt-1">{chart.chart_type}</p>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

export default ReportPage;