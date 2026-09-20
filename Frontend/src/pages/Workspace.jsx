import { useEffect, useState, useRef } from "react";
import { useParams, useNavigate, useLocation } from "react-router-dom";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import axiosInstance from "../api/axiosstance";
import ArtifactPanel from "../components/ArtifactPanel";
import { cleanMarkdown, formatAnalysisAsMarkdown } from "../utils/FormatReport";

function resultToMessages(result) {
  if (result.direct_answer) return [{ role: "assistant", type: "text", text: result.direct_answer }];
  if (result.interrupted) return [{ role: "assistant", type: "interrupt", interrupt: result.interrupt }];
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

function Composer({ onStart }) {
  const [file, setFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [query, setQuery] = useState("");
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [quickResult, setQuickResult] = useState(null);
  const [quickLoading, setQuickLoading] = useState(false);
  const [copied, setCopied] = useState(false);

  function handleDrop(e) {
    e.preventDefault();
    setDragActive(false);
    if (e.dataTransfer.files?.[0]) setFile(e.dataTransfer.files[0]);
  }

  async function handleSubmit(e) {
    e.preventDefault();
    setErrorMsg("");
    if (!file) {
      setErrorMsg("Please upload a CSV file before starting a full run.");
      return;
    }
    if (!query.trim()) {
      setErrorMsg("Please describe what you want to analyze.");
      return;
    }
    setLoading(true);
    try {
      await onStart(file, query);
    } catch (error) {
      setErrorMsg(error.response?.data?.detail || "Something went wrong.");
      setLoading(false);
    }
  }

  async function handleQuickAnalyze() {
    if (!file) return;
      const isCsv = /\.csv$/i.test(file.name) || file.type?.includes("csv");
      if (!isCsv) {
        setErrorMsg("Only CSV files can be analyzed here.");
        return;
      }
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
    <div className="max-w-xl mx-auto px-6 py-16">
      <h1 className="font-serif text-2xl text-ink mb-6 text-center">What do you want to analyze?</h1>
      <form onSubmit={handleSubmit} className="space-y-5">
        <div
          onDragOver={(e) => { e.preventDefault(); setDragActive(true); }}
          onDragLeave={() => setDragActive(false)}
          onDrop={handleDrop}
          className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer transition-colors ${
            dragActive ? "border-accent bg-accent/5" : "border-muted/40"
          }`}
          onClick={() => document.getElementById("fileInput").click()}
        >
          <input id="fileInput" type="file" accept=".csv"
            onChange={(e) => { setFile(e.target.files[0]); setQuickResult(null); }} className="hidden" />
          {file ? (
            <p className="text-ink text-sm font-medium">{file.name}</p>
          ) : (
            <>
              <p className="text-ink text-sm font-medium mb-1">Drop your CSV here, or click to browse</p>
              <p className="text-muted text-xs">CSV files only</p>
            </>
          )}
        </div>

        <button
          type="button"
          onClick={handleQuickAnalyze}
          disabled={!file || quickLoading}
          className="w-full border border-ink text-ink py-2 rounded-md font-medium hover:bg-ink/5 transition-colors disabled:opacity-50"
        >
          {quickLoading ? "Analyzing..." : "Quick Analyze (no full pipeline)"}
        </button>

        {quickResult && (
          <div className="space-y-2">
            <div className="bg-white border border-muted/20 rounded-md p-4 max-h-64 overflow-auto">
              <article className="prose prose-sm max-w-none prose-headings:font-serif prose-headings:text-ink prose-p:text-ink prose-li:text-ink prose-strong:text-ink">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{cleanMarkdown(formatAnalysisAsMarkdown(quickResult))}</ReactMarkdown>
              </article>
            </div>
            <div className="flex gap-2">
              <button type="button" onClick={copyQuickResult}
                className="text-xs bg-panel text-paper px-3 py-1.5 rounded-md hover:bg-panel/80">
                {copied ? "Copied!" : "Copy"}
              </button>
              <button type="button"
                onClick={() => downloadBlob(formatAnalysisAsMarkdown(quickResult), "dataset-summary.md", "text/markdown")}
                className="text-xs bg-ink text-paper px-3 py-1.5 rounded-md hover:bg-panel">
                Markdown
              </button>
              <button type="button" onClick={downloadQuickPdf}
                className="text-xs bg-ink text-paper px-3 py-1.5 rounded-md hover:bg-panel">
                PDF
              </button>
            </div>
          </div>
        )}

        <div className="flex items-center gap-3">
          <div className="flex-1 h-px bg-muted/20" />
          <span className="text-xs text-muted">or run the full pipeline</span>
          <div className="flex-1 h-px bg-muted/20" />
        </div>

        <textarea
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="e.g. Clean the data, run EDA, and train a model to predict X"
          className="w-full px-3 py-2 border border-muted/40 rounded-md bg-white text-ink focus:outline-none focus:ring-2 focus:ring-accent"
          rows={3}
          required
        />
        {errorMsg && <p className="text-clay text-sm">{errorMsg}</p>}
        <button type="submit" disabled={loading || !file}
          className="w-full bg-ink text-paper py-2 rounded-md font-medium hover:bg-panel transition-colors disabled:opacity-50">
          {loading ? "Starting..." : "Start Full Run"}
        </button>
      </form>
    </div>
  );
}

function InterruptCard({ interrupt }) {
  const type = interrupt?.type;

  if (type === "job_status") {
    return (
      <div className="bg-white border border-muted/20 rounded-lg px-4 py-3 text-sm max-w-[85%] flex items-center gap-2">
        <span className="animate-pulse">⏳</span>
        <p className="text-ink">{interrupt.summary}</p>
      </div>
    );
  }

  if (type === "plan_review") {
    return (
      <div className="bg-white border border-muted/20 rounded-lg px-4 py-3 text-sm max-w-[85%]">
        <p className="font-medium text-ink mb-2">Here's the plan — does this look right?</p>
        <p className="whitespace-pre-wrap text-ink mb-3">{interrupt.summary}</p>
        {interrupt.validation_problems?.length > 0 && (
          <div className="bg-clay/10 border border-clay/30 rounded-md p-2 mb-2">
            <p className="text-clay text-xs font-medium mb-1">Heads up:</p>
            <ul className="text-clay text-xs list-disc list-inside">
              {interrupt.validation_problems.map((p, i) => <li key={i}>{p}</li>)}
            </ul>
          </div>
        )}
      </div>
    );
  }

  if (type === "refinement") {
    return (
      <div className="bg-white border border-muted/20 rounded-lg px-4 py-3 text-sm max-w-[85%]">
        <p className="font-medium text-ink mb-2">Confirm this change?</p>
        <p className="whitespace-pre-wrap text-ink">{interrupt.summary}</p>
      </div>
    );
  }

  return (
    <div className="bg-white border border-muted/20 rounded-lg px-4 py-3 text-sm max-w-[85%]">
      <p className="font-medium text-ink mb-2">Needs your input:</p>
      <p className="whitespace-pre-wrap text-ink">
        {interrupt?.summary || JSON.stringify(interrupt, null, 2)}
      </p>
    </div>
  );
}

function MessageBubble({ msg }) {
  const isUser = msg.role === "user";

  if (msg.type === "interrupt") {
    return <div className="flex justify-start"><InterruptCard interrupt={msg.interrupt} /></div>;
  }

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[80%] rounded-lg px-4 py-2 text-sm ${
        isUser ? "bg-ink text-paper" : "bg-white border border-muted/20 text-ink"
      }`}>
        {msg.type === "done" ? (
          <p>Pipeline step finished.</p>
        ) : isUser ? (
          <p className="whitespace-pre-wrap">{msg.text}</p>
        ) : (
          <article className="prose prose-sm max-w-none prose-headings:font-serif prose-headings:text-ink prose-p:text-ink prose-li:text-ink prose-strong:text-ink prose-table:text-ink prose-table:whitespace-nowrap">
            <div className="overflow-x-auto">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{cleanMarkdown(msg.text)}</ReactMarkdown>
            </div>
          </article>
        )}
      </div>
    </div>
  );
}

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
  const [editText, setEditText] = useState("");
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [loadingHistory, setLoadingHistory] = useState(Boolean(threadId));

  const pollIntervalRef = useRef(null);

  useEffect(() => {
    // Route changes must reset the visible conversation before loading its data.
    // oxlint-disable-next-line react/set-state-in-effect
    setShowArtifact(false);
    // oxlint-disable-next-line react/set-state-in-effect
    setErrorMsg("");

    if (!threadId) {
      setMessages([]);
      setAwaitingDecision(false);
      setReportReady(false);
      setLoadingHistory(false);
      return;
    }
    // Freshly started run — the live result is already in hand via
    // navigation state, so skip refetching (which would otherwise race
    // with this and clobber it with an incomplete DB snapshot).
    if (freshResult) {
      const result = freshResult;
      setMessages([
        { role: "user", type: "text", text: freshQuery },
        ...resultToMessages(result),
      ]);
      setAwaitingDecision(Boolean(result.interrupted));
      setReportReady(!result.interrupted);
      setLoadingHistory(false);
      // Clear so a future reload of this URL (no state left) correctly
      // falls through to the normal server-fetch path below.
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
          // Append — don't overwrite stored history, add the live
          // interrupt as additional current context.
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
      .catch((error) => setErrorMsg(error.response?.data?.detail || "Could not load conversation."))
      .finally(() => setLoadingHistory(false));
  }, [threadId, freshResult, freshQuery]);

  // Auto-poll: when the current pending interrupt is a "job_status" type
  // (a background job like training still running), automatically resend
  // a resume every 3 seconds — no human decision needed, this just asks
  // the graph to re-check whether the background job has finished yet.
  // Stops itself once the interrupt clears (job done) or changes type.
  useEffect(() => {
    const lastMsg = messages[messages.length - 1];
    const isJobWaiting =
      awaitingDecision &&
      lastMsg?.type === "interrupt" &&
      lastMsg.interrupt?.type === "job_status";

    if (isJobWaiting && !pollIntervalRef.current) {
      pollIntervalRef.current = setInterval(() => {
        sendChat({ decision: { approved: true, edit_instruction: null } });
      }, 3000);
    }

    if (!isJobWaiting && pollIntervalRef.current) {
      clearInterval(pollIntervalRef.current);
      pollIntervalRef.current = null;
    }

    return () => {
      if (pollIntervalRef.current) {
        clearInterval(pollIntervalRef.current);
        pollIntervalRef.current = null;
      }
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages, awaitingDecision]);

  async function startRun(file, query) {
    const trimmedQuery = query?.trim();
    if (!file) throw new Error("Please upload a CSV file before starting a full run.");
    const isCsv = /\.csv$/i.test(file.name) || file.type?.includes("csv");
    if (!isCsv) throw new Error("Only CSV files are supported.");
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
      setEditText("");
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
    e.preventDefault();
    if (!input.trim()) return;
    sendChat({ user_query: input }, input);
  }

  function handleDecision(approved) {
    sendChat({ decision: { approved, edit_instruction: null } }, approved ? "Approved" : "Rejected");
  }

  function handleEditSubmit(e) {
    e.preventDefault();
    if (!editText.trim()) return;
    sendChat({ decision: { approved: false, edit_instruction: editText } }, `Edit: ${editText}`);
  }

  if (!threadId) return <div className="bg-paper h-full overflow-y-auto"><Composer onStart={startRun} /></div>;
  if (loadingHistory) return <div className="max-w-2xl mx-auto px-6 py-12 text-muted">Loading...</div>;

  const lastMsg = messages[messages.length - 1];
  const isJobWaiting =
    awaitingDecision &&
    lastMsg?.type === "interrupt" &&
    lastMsg.interrupt?.type === "job_status";

  return (
    <div className="flex h-full">
      <div className="flex-1 flex flex-col">
        <div className="flex-1 overflow-y-auto px-6 py-6 max-w-2xl mx-auto w-full space-y-4">
          {messages.map((msg, i) => <MessageBubble key={i} msg={msg} />)}
          {errorMsg && <p className="text-clay text-sm">{errorMsg}</p>}
        </div>

        <div className="border-t border-muted/20 px-6 py-4 max-w-2xl mx-auto w-full space-y-3">
          {reportReady && !awaitingDecision && (
            <button onClick={() => setShowArtifact(true)} className="text-sm text-accent hover:underline">
              📊 View Report →
            </button>
          )}

          {awaitingDecision ? (
            isJobWaiting ? null : (
              <div className="space-y-3">
                <div className="flex gap-3">
                  <button onClick={() => handleDecision(true)} disabled={loading}
                    className="bg-accent text-ink px-4 py-2 rounded-md text-sm font-medium disabled:opacity-50">
                    Approve
                  </button>
                  <button onClick={() => handleDecision(false)} disabled={loading}
                    className="bg-clay text-white px-4 py-2 rounded-md text-sm font-medium disabled:opacity-50">
                    Reject
                  </button>
                </div>
                <form onSubmit={handleEditSubmit} className="flex gap-2">
                  <input value={editText} onChange={(e) => setEditText(e.target.value)}
                    placeholder="Or describe what to change instead..."
                    className="flex-1 px-3 py-2 border border-muted/40 rounded-md bg-white text-ink text-sm focus:outline-none focus:ring-2 focus:ring-accent" />
                  <button type="submit" disabled={loading}
                    className="bg-ink text-paper px-4 py-2 rounded-md text-sm font-medium disabled:opacity-50">
                    Send Edit
                  </button>
                </form>
              </div>
            )
          ) : (
            <form onSubmit={handleSendMessage} className="flex gap-2">
              <input value={input} onChange={(e) => setInput(e.target.value)}
                placeholder="Message..."
                className="flex-1 px-3 py-2 border border-muted/40 rounded-md bg-white text-ink text-sm focus:outline-none focus:ring-2 focus:ring-accent" />
              <button type="submit" disabled={loading}
                className="bg-ink text-paper px-4 py-2 rounded-md text-sm font-medium disabled:opacity-50">
                Send
              </button>
            </form>
          )}
        </div>
      </div>

      {showArtifact && <ArtifactPanel threadId={threadId} onClose={() => setShowArtifact(false)} />}
    </div>
  );
}

export default Workspace;