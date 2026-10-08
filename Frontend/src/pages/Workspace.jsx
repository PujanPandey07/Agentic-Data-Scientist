import { useEffect, useRef, useState } from "react";
import { useParams, useNavigate, useLocation } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import axiosInstance, { getAccessToken, API_BASE } from "../api/axiosstance";
import ArtifactPanel from "../components/ArtifactPanel";
import { cleanMarkdown, formatAnalysisAsMarkdown } from "../utils/FormatReport";
import { useLayout } from "../context/LayoutContext";

// ─── helpers ────────────────────────────────────────────────────────────────

function resultToMessages(result) {
  if (result.direct_answer)
    return [{ role: "assistant", type: "text", text: result.direct_answer }];
  if (result.interrupted)
    return [{ role: "assistant", type: "interrupt", interrupt: result.interrupt }];
  return [{ role: "assistant", type: "done" }];
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

// ─── icons ───────────────────────────────────────────────────────────────────

function SendIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 24 24"
      fill="none" stroke="currentColor" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round">
      <line x1="22" y1="2" x2="11" y2="13" />
      <polygon points="22 2 15 22 11 13 2 9 22 2" />
    </svg>
  );
}

function PaperclipIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 24 24"
      fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="M21.44 11.05l-9.19 9.19a6 6 0 0 1-8.49-8.49l9.19-9.19a4 4 0 0 1 5.66 5.66L9.64 16.34a2 2 0 0 1-2.83-2.83l8.49-8.48" />
    </svg>
  );
}

function ChevronIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-3 w-3 ml-1" viewBox="0 0 24 24"
      fill="none" stroke="currentColor" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round">
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}

function SparkleIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 24 24"
      fill="currentColor">
      <path d="M12 2l2.4 7.6H22l-6.4 4.6 2.4 7.6L12 17.2l-6 4.6 2.4-7.6L2 9.6h7.6L12 2z" />
    </svg>
  );
}

function XIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" viewBox="0 0 24 24"
      fill="none" stroke="currentColor" strokeWidth={2.5} strokeLinecap="round" strokeLinejoin="round">
      <line x1="18" y1="6" x2="6" y2="18" />
      <line x1="6" y1="6" x2="18" y2="18" />
    </svg>
  );
}

// ─── model indicator ─────────────────────────────────────────────────────────

function ModelBadge({ onClick }) {
  const [modelLabel, setModelLabel] = useState(null);

  useEffect(() => {
    axiosInstance
      .get("/api/api-keys/status")
      .then((res) => {
        if (res.data.configured) {
          const name = res.data.model_name || res.data.provider;
          setModelLabel(name);
        }
      })
      .catch(() => {});
  }, []);

  const label = modelLabel || "Default model";

  return (
    <button
      onClick={onClick}
      className="flex items-center gap-1.5 text-xs px-3 py-1.5 rounded-full font-medium transition-colors"
      style={{
        background: "#f0f0f0",
        color: "#555",
        border: "1px solid #e0e0e0",
      }}
      onMouseEnter={(e) => {
        e.currentTarget.style.background = "#e8e8e8";
        e.currentTarget.style.borderColor = "#ccc";
      }}
      onMouseLeave={(e) => {
        e.currentTarget.style.background = "#f0f0f0";
        e.currentTarget.style.borderColor = "#e0e0e0";
      }}
    >
      <SparkleIcon />
      {label}
      <ChevronIcon />
    </button>
  );
}

// ─── interrupt card ───────────────────────────────────────────────────────────

function InterruptCard({ interrupt }) {
  const type = interrupt?.type;

  if (type === "job_status") {
    return (
      <div
        className="flex items-center gap-3 px-4 py-3 rounded-xl text-sm max-w-prose"
        style={{ background: "#fff7ed", border: "1px solid #f5d9c0", color: "#8a5c3a" }}
      >
        <div className="shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs"
          style={{ background: "#cc785c", color: "white" }}>
          <svg className="h-3 w-3 animate-spin" fill="none" viewBox="0 0 24 24">
            <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
            <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
          </svg>
        </div>
        <p>{interrupt.summary}</p>
      </div>
    );
  }

  if (type === "plan_review") {
    return (
      <div className="space-y-2 max-w-prose">
        <div className="px-4 py-3 rounded-xl text-sm"
          style={{ background: "#f8f8f7", border: "1px solid #e5e5e5" }}>
          <p className="font-semibold text-sm mb-2" style={{ color: "#1a1a1a" }}>
            Here&apos;s the plan — does this look right?
          </p>
          <p className="whitespace-pre-wrap text-sm leading-relaxed" style={{ color: "#444" }}>
            {interrupt.summary}
          </p>
          {interrupt.validation_problems?.length > 0 && (
            <div className="mt-3 px-3 py-2 rounded-lg text-xs"
              style={{ background: "#fef2f2", border: "1px solid #fecaca", color: "#7f1d1d" }}>
              <p className="font-semibold mb-1">Heads up:</p>
              <ul className="list-disc list-inside space-y-0.5">
                {interrupt.validation_problems.map((p, i) => <li key={i}>{p}</li>)}
              </ul>
            </div>
          )}
        </div>
      </div>
    );
  }

  if (type === "refinement") {
    return (
      <div className="px-4 py-3 rounded-xl text-sm max-w-prose"
        style={{ background: "#f8f8f7", border: "1px solid #e5e5e5" }}>
        <p className="font-semibold mb-2" style={{ color: "#1a1a1a" }}>Confirm this change?</p>
        <p className="whitespace-pre-wrap leading-relaxed" style={{ color: "#444" }}>
          {interrupt.summary}
        </p>
      </div>
    );
  }

  return (
    <div className="px-4 py-3 rounded-xl text-sm max-w-prose"
      style={{ background: "#f8f8f7", border: "1px solid #e5e5e5" }}>
      <p className="font-semibold mb-2" style={{ color: "#1a1a1a" }}>Needs your input:</p>
      <p className="whitespace-pre-wrap leading-relaxed" style={{ color: "#444" }}>
        {interrupt?.summary || JSON.stringify(interrupt, null, 2)}
      </p>
    </div>
  );
}

// ─── message bubble ───────────────────────────────────────────────────────────

function MessageBubble({ msg }) {
  const isUser = msg.role === "user";

  if (msg.type === "interrupt") {
    return (
      <div className="flex justify-start px-4 md:px-8 py-1">
        <div className="flex gap-3 max-w-2xl w-full">
          {/* AI avatar */}
          <div className="shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold mt-0.5"
            style={{ background: "#cc785c", color: "white" }}>
            AI
          </div>
          <div className="flex-1 min-w-0">
            <InterruptCard interrupt={msg.interrupt} />
          </div>
        </div>
      </div>
    );
  }

  if (msg.type === "done") {
    return (
      <div className="flex justify-start px-4 md:px-8 py-1">
        <div className="flex gap-3 max-w-2xl w-full">
          <div className="shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold mt-0.5"
            style={{ background: "#cc785c", color: "white" }}>
            AI
          </div>
          <p className="text-sm pt-1.5" style={{ color: "#888" }}>
            Pipeline step completed.
          </p>
        </div>
      </div>
    );
  }

  if (isUser) {
    return (
      <div className="flex justify-end px-4 md:px-8 py-1">
        <div
          className="max-w-[70%] px-4 py-2.5 rounded-2xl text-sm leading-relaxed"
          style={{ background: "#f0f0f0", color: "#1a1a1a" }}
        >
          <p className="whitespace-pre-wrap">{msg.text}</p>
        </div>
      </div>
    );
  }

  // AI text message
  return (
    <div className="flex justify-start px-4 md:px-8 py-1">
      <div className="flex gap-3 max-w-2xl w-full">
        {/* AI avatar */}
        <div className="shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold mt-1"
          style={{ background: "#cc785c", color: "white" }}>
          AI
        </div>
        <div className="flex-1 min-w-0 pt-0.5">
          <div className="ai-prose text-sm leading-relaxed" style={{ color: "#1a1a1a" }}>
            <ReactMarkdown remarkPlugins={[remarkGfm]}>
              {cleanMarkdown(msg.text)}
            </ReactMarkdown>
          </div>
        </div>
      </div>
    </div>
  );
}

// ─── loading dots ─────────────────────────────────────────────────────────────

function ThinkingDots() {
  return (
    <div className="flex justify-start px-4 md:px-8 py-1">
      <div className="flex gap-3 max-w-2xl w-full">
        <div className="shrink-0 w-7 h-7 rounded-full flex items-center justify-center text-xs font-bold mt-1"
          style={{ background: "#cc785c", color: "white" }}>
          AI
        </div>
        <div className="flex gap-1.5 items-center pt-2">
          {[0, 1, 2].map((i) => (
            <span key={i} className="w-1.5 h-1.5 rounded-full" style={{ background: "#ccc",
              animation: `bounce 1.2s ease-in-out ${i * 0.2}s infinite` }} />
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── chat input ───────────────────────────────────────────────────────────────

function ChatInput({ value, onChange, onSubmit, placeholder, disabled, children }) {
  const textareaRef = useRef(null);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = Math.min(el.scrollHeight, 200) + "px";
  }, [value]);

  function handleKeyDown(e) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      onSubmit(e);
    }
  }

  return (
    <div className="rounded-2xl border transition-colors"
      style={{ background: "white", borderColor: "#e5e5e5", boxShadow: "0 1px 4px rgba(0,0,0,0.06)" }}>
      <textarea
        ref={textareaRef}
        value={value}
        onChange={onChange}
        onKeyDown={handleKeyDown}
        placeholder={placeholder}
        disabled={disabled}
        rows={1}
        className="w-full px-4 pt-3 pb-1 text-sm bg-transparent focus:outline-none"
        style={{ color: "#1a1a1a", minHeight: "44px", maxHeight: "200px" }}
      />
      <div className="flex items-center justify-between px-3 pb-2.5 pt-1">
        <div className="flex gap-1">{children}</div>
        <button
          type="button"
          onClick={onSubmit}
          disabled={disabled || !value.trim()}
          className="w-8 h-8 rounded-full flex items-center justify-center transition-colors disabled:opacity-40"
          style={{
            background: disabled || !value.trim() ? "#e5e5e5" : "#cc785c",
            color: disabled || !value.trim() ? "#999" : "white",
          }}
        >
          <SendIcon />
        </button>
      </div>
    </div>
  );
}

// ─── composer (no threadId) ───────────────────────────────────────────────────

function Composer({ onStart }) {
  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [quickResult, setQuickResult] = useState(null);
  const [quickLoading, setQuickLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  const fileInputRef = useRef(null);

  function handleDrop(e) {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) setFile(e.dataTransfer.files[0]);
  }

  async function handleSubmit(e) {
    e?.preventDefault();
    setErrorMsg("");
    if (!file) { setErrorMsg("Please upload a file before starting a run."); return; }
    if (!query.trim()) { setErrorMsg("Please describe what you want to analyze."); return; }
    setLoading(true);
    try {
      await onStart(file, query);
    } catch (error) {
      setErrorMsg(error.response?.data?.detail || error.message || "Something went wrong.");
      setLoading(false);
    }
  }

  async function handleQuickAnalyze() {
    if (!file) return;
    const isSupported = /\.(csv|tsv|tab|xlsx|xls|parquet|pq|json|feather)$/i.test(file.name);
    if (!isSupported) { setErrorMsg("Please upload a supported tabular dataset (CSV, TSV, Excel, Parquet, JSON, or Feather)."); return; }
    setErrorMsg("");
    setQuickResult(null);
    setQuickLoading(true);
    try {
      const formData = new FormData();
      formData.append("file", file);
      const uploadRes = await axiosInstance.post("/api/upload", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });
      const analyzeRes = await axiosInstance.get(`/api/analyze/${uploadRes.data.dataset_id}`);
      setQuickResult(analyzeRes.data);
    } catch (error) {
      setErrorMsg(error.response?.data?.detail || "Could not analyze dataset.");
    } finally {
      setQuickLoading(false);
    }
  }

  function copyQuickResult() {
    navigator.clipboard.writeText(formatAnalysisAsMarkdown(quickResult));
    setCopied(true);
    setTimeout(() => setCopied(false), 1500);
  }

  function downloadQuickPdf() {
    import("jspdf").then(({ default: JsPDF }) => {
      const doc = new JsPDF();
      let y = 20;
      doc.setFontSize(16);
      doc.text("Dataset Summary", 10, y);
      y += 12;
      doc.setFontSize(9);
      const plain = formatAnalysisAsMarkdown(quickResult).replace(/[#*`|]/g, "");
      const lines = doc.splitTextToSize(plain, 180);
      for (const line of lines) {
        if (y > 280) { doc.addPage(); y = 20; }
        doc.text(line, 10, y);
        y += 5;
      }
      doc.save("dataset-summary.pdf");
    });
  }

  return (
    <div className="flex flex-col items-center justify-center h-full px-6">
      <div className="w-full max-w-xl">
        {/* Greeting */}
        <div className="text-center mb-8">
          <h1 className="text-2xl font-semibold mb-1" style={{ color: "#1a1a1a" }}>
            What would you like to analyze?
          </h1>
          <p className="text-sm" style={{ color: "#888" }}>
            Upload a dataset and describe your goal. I'll build you a full ML pipeline.
          </p>
        </div>

        {/* File drop zone */}
        <div
          onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
          onClick={() => fileInputRef.current?.click()}
          className="border-2 border-dashed rounded-2xl p-6 text-center cursor-pointer transition-colors mb-3"
          style={{
            borderColor: dragActive ? "#cc785c" : "#d8d8d8",
            background: dragActive ? "#fff7f5" : "#fafafa",
          }}
        >
          <input
            ref={fileInputRef}
            id="fileInput"
            type="file"
            accept=".csv,.tsv,.tab,.xlsx,.xls,.parquet,.pq,.json,.feather"
            onChange={(e) => { setFile(e.target.files[0]); setQuickResult(null); }}
            className="hidden"
          />
          {file ? (
            <div className="flex items-center justify-center gap-2">
              <div className="w-8 h-8 rounded-lg flex items-center justify-center"
                style={{ background: "#fff7ed", color: "#cc785c" }}>
                <PaperclipIcon />
              </div>
              <div className="text-left">
                <p className="text-sm font-medium" style={{ color: "#1a1a1a" }}>{file.name}</p>
                <p className="text-xs" style={{ color: "#888" }}>
                  {(file.size / 1024).toFixed(1)} KB · Click to replace
                </p>
              </div>
            </div>
          ) : (
            <>
              <p className="text-sm font-medium mb-1" style={{ color: "#555" }}>
                Drop your file here or click to browse
              </p>
              <p className="text-xs" style={{ color: "#aaa" }}>CSV, TSV, Excel, Parquet, JSON, or Feather files up to 100MB</p>
            </>
          )}
        </div>

        {/* Quick analyze */}
        {file && (
          <div className="flex justify-end mb-3">
            <button
              type="button"
              onClick={handleQuickAnalyze}
              disabled={!file || quickLoading}
              className="text-xs px-3 py-1.5 rounded-lg transition-colors disabled:opacity-50"
              style={{ background: "#f0f0f0", color: "#555", border: "1px solid #e0e0e0" }}
            >
              {quickLoading ? "Analyzing..." : "Quick dataset preview"}
            </button>
          </div>
        )}

        {/* Quick result */}
        {quickResult && (
          <div className="mb-3 rounded-2xl border overflow-hidden" style={{ borderColor: "#e5e5e5" }}>
            <div className="px-4 py-3 max-h-52 overflow-auto text-sm" style={{ background: "#fafafa" }}>
              <div className="ai-prose">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>
                  {cleanMarkdown(formatAnalysisAsMarkdown(quickResult))}
                </ReactMarkdown>
              </div>
            </div>
            <div className="flex gap-2 px-4 py-2" style={{ borderTop: "1px solid #e5e5e5" }}>
              <button onClick={copyQuickResult}
                className="text-xs px-2.5 py-1 rounded-lg"
                style={{ background: "#f0f0f0", color: "#555" }}>
                {copied ? "Copied!" : "Copy"}
              </button>
              <button
                onClick={() => downloadBlob(formatAnalysisAsMarkdown(quickResult), "dataset-summary.md", "text/markdown")}
                className="text-xs px-2.5 py-1 rounded-lg"
                style={{ background: "#f0f0f0", color: "#555" }}>
                Markdown
              </button>
              <button onClick={downloadQuickPdf}
                className="text-xs px-2.5 py-1 rounded-lg"
                style={{ background: "#f0f0f0", color: "#555" }}>
                PDF
              </button>
            </div>
          </div>
        )}

        {/* Divider */}
        {quickResult && (
          <div className="flex items-center gap-3 mb-3">
            <div className="flex-1 h-px" style={{ background: "#e5e5e5" }} />
            <span className="text-xs" style={{ color: "#aaa" }}>or run the full pipeline</span>
            <div className="flex-1 h-px" style={{ background: "#e5e5e5" }} />
          </div>
        )}

        {/* Chat input */}
        <ChatInput
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onSubmit={handleSubmit}
          placeholder={`e.g. "Clean the data, run EDA, and train a model to predict the target column"`}
          disabled={loading}
        >
          {loading && (
            <span className="text-xs px-1" style={{ color: "#aaa" }}>Starting pipeline…</span>
          )}
        </ChatInput>

        {errorMsg && (
          <p className="text-xs mt-2 text-center" style={{ color: "#e05252" }}>{errorMsg}</p>
        )}
      </div>
    </div>
  );
}

// ─── decision bar ─────────────────────────────────────────────────────────────

function DecisionBar({ onApprove, onReject, onEdit, loading }) {
  const [editText, setEditText] = useState("");
  const [showEdit, setShowEdit] = useState(false);

  function handleEdit(e) {
    e?.preventDefault();
    if (!editText.trim()) return;
    onEdit(editText);
    setEditText("");
    setShowEdit(false);
  }

  if (showEdit) {
    return (
      <div className="w-full max-w-2xl mx-auto space-y-2">
        <ChatInput
          value={editText}
          onChange={(e) => setEditText(e.target.value)}
          onSubmit={handleEdit}
          placeholder="Describe what you'd like changed…"
          disabled={loading}
        >
          <button
            onClick={() => setShowEdit(false)}
            className="text-xs px-2.5 py-1 rounded-lg"
            style={{ background: "#f0f0f0", color: "#555" }}
          >
            Cancel
          </button>
        </ChatInput>
      </div>
    );
  }

  return (
    <div className="w-full max-w-2xl mx-auto flex flex-wrap sm:flex-nowrap items-center gap-2">
      <button
        onClick={onApprove}
        disabled={loading}
        className="flex-1 min-w-[120px] py-2 rounded-xl text-sm font-medium transition-colors disabled:opacity-50"
        style={{ background: "#cc785c", color: "white" }}
        onMouseEnter={(e) => (e.currentTarget.style.background = "#b86647")}
        onMouseLeave={(e) => (e.currentTarget.style.background = "#cc785c")}
      >
        ✓ Approve
      </button>
      <button
        onClick={() => setShowEdit(true)}
        disabled={loading}
        className="flex-1 min-w-[120px] py-2 rounded-xl text-sm font-medium transition-colors disabled:opacity-50"
        style={{ background: "#f0f0f0", color: "#1a1a1a" }}
        onMouseEnter={(e) => (e.currentTarget.style.background = "#e5e5e5")}
        onMouseLeave={(e) => (e.currentTarget.style.background = "#f0f0f0")}
      >
        ✎ Suggest edit
      </button>
      <button
        onClick={onReject}
        disabled={loading}
        className="w-full sm:w-auto px-4 py-2 rounded-xl text-sm font-medium transition-colors disabled:opacity-50"
        style={{ background: "#fff0f0", color: "#c0392b", border: "1px solid #fecaca" }}
        onMouseEnter={(e) => (e.currentTarget.style.background = "#fee2e2")}
        onMouseLeave={(e) => (e.currentTarget.style.background = "#fff0f0")}
      >
        ✕ Reject
      </button>
    </div>
  );
}

// ─── main Workspace ───────────────────────────────────────────────────────────

function Workspace() {
  const { threadId } = useParams();
  const navigate = useNavigate();
  const location = useLocation();
  const freshResult = location.state?.freshResult;
  const freshQuery = location.state?.freshQuery;

  const [messages, setMessages] = useState([]);
  const [awaitingDecision, setAwaitingDecision] = useState(false);
  const [reportReady, setReportReady] = useState(false);
  const [showArtifact, setShowArtifact] = useState(false);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [loadingHistory, setLoadingHistory] = useState(Boolean(threadId));

  const bottomRef = useRef(null);
  const sseRef = useRef(null);

  // Scroll to bottom on new messages
  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, loading]);

  // Load conversation history or fresh result
  useEffect(() => {
    setShowArtifact(false);
    setErrorMsg("");

    if (!threadId) {
      setMessages([]);
      setAwaitingDecision(false);
      setReportReady(false);
      setLoadingHistory(false);
      return;
    }

    if (freshResult) {
      setMessages([
        { role: "user", type: "text", text: freshQuery },
        ...resultToMessages(freshResult),
      ]);
      setAwaitingDecision(Boolean(freshResult.interrupted));
      setReportReady(!freshResult.interrupted);
      setLoadingHistory(false);
      window.history.replaceState({}, "");
      return;
    }

    setLoadingHistory(true);
    setMessages([]);
    setAwaitingDecision(false);
    setReportReady(false);

    Promise.all([
      axiosInstance.get(`/api/conversations/${threadId}/messages`),
      axiosInstance.get(`/api/chat/${threadId}/pending`),
    ])
      .then(([msgRes, pendingRes]) => {
        const loaded = msgRes.data.messages.map((m) => ({
          role: m.role,
          type: "text",
          text: m.content,
        }));

        if (pendingRes.data.interrupted) {
          loaded.push({
            role: "assistant",
            type: "interrupt",
            interrupt: pendingRes.data.interrupt,
          });
          setAwaitingDecision(true);
        } else {
          setReportReady(loaded.length > 0);
        }

        setMessages(loaded);
      })
      .catch((err) => setErrorMsg(err.response?.data?.detail || "Could not load conversation."))
      .finally(() => setLoadingHistory(false));
  }, [threadId, freshResult, freshQuery]);

  // SSE — auto-resume when training job finishes
  useEffect(() => {
    const lastMsg = messages[messages.length - 1];
    const isJobWaiting =
      awaitingDecision &&
      lastMsg?.type === "interrupt" &&
      lastMsg.interrupt?.type === "job_status";

    if (isJobWaiting && !sseRef.current) {
      const jobId = lastMsg.interrupt.job_id;
      const token = getAccessToken();
      const url = `${API_BASE}/api/jobs/${jobId}/stream?token=${encodeURIComponent(token)}`;
      const es = new EventSource(url);
      sseRef.current = es;

      es.onmessage = (event) => {
        const data = JSON.parse(event.data);
        if (data.status === "complete") {
          es.close();
          sseRef.current = null;
          sendChat({ decision: { approved: true, edit_instruction: null } });
        } else if (data.status === "timeout") {
          es.close();
          sseRef.current = null;
          setErrorMsg("Training is taking longer than expected. Please check back later.");
          setAwaitingDecision(false);
        }
      };

      es.onerror = () => {
        es.close();
        sseRef.current = null;
        setErrorMsg("Lost connection while waiting for training. Please refresh.");
        setAwaitingDecision(false);
      };
    }

    if (!isJobWaiting && sseRef.current) {
      sseRef.current.close();
      sseRef.current = null;
    }

    return () => {
      if (sseRef.current) {
        sseRef.current.close();
        sseRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages, awaitingDecision]);

  // ── API calls ────────────────────────────────────────────────────────────

  async function startRun(file, query) {
    const trimmedQuery = query?.trim();
    if (!file) throw new Error("Please upload a file first.");
    if (!trimmedQuery) throw new Error("Please describe what you want to analyze.");

    const formData = new FormData();
    formData.append("file", file);
    const uploadRes = await axiosInstance.post("/api/upload", formData, {
      headers: { "Content-Type": "multipart/form-data" },
    });
    const runRes = await axiosInstance.post("/api/runs", {
      dataset_id: uploadRes.data.dataset_id,
      user_query: trimmedQuery,
    });

    navigate(`/c/${runRes.data.thread_id}`, {
      state: { freshResult: runRes.data, freshQuery: trimmedQuery },
    });
  }

  async function sendChat(body, userDisplayText) {
    setErrorMsg("");
    setLoading(true);
    if (userDisplayText) {
      setMessages((prev) => [...prev, { role: "user", type: "text", text: userDisplayText }]);
    }
    try {
      const response = await axiosInstance.post("/api/chat", { thread_id: threadId, ...body });
      setAwaitingDecision(Boolean(response.data.interrupted));
      setReportReady(!response.data.interrupted);
      setMessages((prev) => [...prev, ...resultToMessages(response.data)]);
      setInput("");
    } catch (error) {
      const detail = error.response?.data?.detail || "";
      if (error.response?.status === 400 && detail.includes("waiting for a decision")) {
        setAwaitingDecision(true);
      } else {
        setErrorMsg(detail || "Something went wrong.");
      }
    } finally {
      setLoading(false);
    }
  }

  function handleSendMessage(e) {
    e?.preventDefault();
    if (!input.trim()) return;
    sendChat({ user_query: input }, input);
  }

  function handleDecision(approved) {
    sendChat({ decision: { approved, edit_instruction: null } }, approved ? "Approved ✓" : "Rejected ✕");
  }

  function handleEditSubmit(editText) {
    sendChat({ decision: { approved: false, edit_instruction: editText } }, `Suggested edit: ${editText}`);
  }

  // ── render ───────────────────────────────────────────────────────────────

  // No threadId → show the composer
  if (!threadId) {
    return (
      <div className="flex-1 h-full overflow-hidden" style={{ background: "#f7f7f5" }}>
        <Composer onStart={startRun} />
      </div>
    );
  }

  if (loadingHistory) {
    return (
      <div className="flex-1 h-full flex items-center justify-center" style={{ background: "#f7f7f5" }}>
        <div className="flex gap-1.5">
          {[0, 1, 2].map((i) => (
            <span key={i} className="w-2 h-2 rounded-full" style={{
              background: "#ccc",
              animation: `bounce 1.2s ease-in-out ${i * 0.2}s infinite`,
            }} />
          ))}
        </div>
      </div>
    );
  }

  const lastMsg = messages[messages.length - 1];
  const isJobWaiting =
    awaitingDecision &&
    lastMsg?.type === "interrupt" &&
    lastMsg.interrupt?.type === "job_status";

  const { toggleSidebar } = useLayout();

  return (
    // The outermost element is a ROW on desktop. On mobile, ArtifactPanel slides in as an overlay.
    <div className="flex-1 flex flex-row h-full overflow-hidden relative" style={{ background: "#f7f7f5" }}>

      {/* Chat column */}
      <div className="flex-1 min-w-0 flex flex-col h-full overflow-hidden">
        {/* Header */}
        <div
          className="shrink-0 flex items-center justify-between px-3 md:px-4 py-3"
          style={{ borderBottom: "1px solid #e8e8e8", background: "white" }}
        >
          <div className="flex items-center gap-2 min-w-0">
            <button
              onClick={toggleSidebar}
              className="p-1.5 rounded-lg md:hidden text-gray-600 hover:text-black hover:bg-gray-100 transition-colors"
              aria-label="Open sidebar"
            >
              <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
              </svg>
            </button>
          </div>
          <div className="flex items-center gap-2">
            {reportReady && !awaitingDecision && (
              <button
                onClick={() => setShowArtifact(true)}
                className="text-xs px-3 py-1.5 rounded-full font-medium transition-colors"
                style={{ background: "#fff7ed", color: "#cc785c", border: "1px solid #f5d9c0" }}
                onMouseEnter={(e) => (e.currentTarget.style.background = "#feecd8")}
                onMouseLeave={(e) => (e.currentTarget.style.background = "#fff7ed")}
              >
                📊 View Report
              </button>
            )}
            <ModelBadge onClick={() => navigate("/settings/api-keys")} />
          </div>
        </div>

        {/* Messages */}
        <div className="flex-1 overflow-y-auto py-6 space-y-1">
          {messages.map((msg, i) => (
            <MessageBubble key={i} msg={msg} />
          ))}
          {loading && <ThinkingDots />}
          {errorMsg && (
            <div className="px-4 md:px-8 py-1">
              <div className="flex gap-3 max-w-2xl">
                <div className="w-7 shrink-0" />
                <p className="text-xs" style={{ color: "#e05252" }}>{errorMsg}</p>
              </div>
            </div>
          )}
          <div ref={bottomRef} />
        </div>

        {/* Bottom input area */}
        <div
          className="shrink-0 px-4 md:px-8 py-4"
          style={{ borderTop: "1px solid #e8e8e8", background: "white" }}
        >
          <div className="max-w-2xl mx-auto">
            {awaitingDecision && !isJobWaiting ? (
              <DecisionBar
                onApprove={() => handleDecision(true)}
                onReject={() => handleDecision(false)}
                onEdit={handleEditSubmit}
                loading={loading}
              />
            ) : isJobWaiting ? (
              <div className="text-center text-sm py-2" style={{ color: "#888" }}>
                <span className="inline-flex items-center gap-2">
                  <svg className="h-3.5 w-3.5 animate-spin" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
                  </svg>
                  Waiting for training to complete…
                </span>
              </div>
            ) : (
              <ChatInput
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onSubmit={handleSendMessage}
                placeholder="Message…"
                disabled={loading}
              />
            )}
            <p className="text-center text-xs mt-2" style={{ color: "#bbb" }}>
              AI Data Scientist · Results may require verification
            </p>
          </div>
        </div>
      </div>

      {/* CHANGED: the panel moved OUT of the chat column and is now its sibling in
          the row (it used to sit at the bottom of the column). */}
      {showArtifact && (
        <ArtifactPanel threadId={threadId} onClose={() => setShowArtifact(false)} />
      )}
    </div>
  );
}

export default Workspace;