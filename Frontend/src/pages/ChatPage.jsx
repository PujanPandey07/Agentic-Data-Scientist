import { useEffect, useState } from "react";
import { useParams, useLocation, Link } from "react-router-dom";
import axiosInstance from "../api/axiosstance";

function resultToMessages(result) {
  if (result.direct_answer) {
    return [{ role: "assistant", type: "text", text: result.direct_answer }];
  }
  if (result.interrupted) {
    return [{ role: "assistant", type: "interrupt", interrupt: result.interrupt }];
  }
  return [{ role: "assistant", type: "text", text: "Pipeline step completed." }];
}

function ChatPage() {
  const { threadId } = useParams();
  const location = useLocation();

  const [messages, setMessages] = useState([]);
  const [latestResult, setLatestResult] = useState(location.state?.runResult || null);
  const [awaitingDecision, setAwaitingDecision] = useState(
    Boolean(location.state?.runResult?.interrupted)
  );
  const [input, setInput] = useState("");
  const [editText, setEditText] = useState("");
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [loadingHistory, setLoadingHistory] = useState(!location.state);

  useEffect(() => {
    if (location.state?.runResult) {
      const msgs = [];
      if (location.state.userQuery) {
        msgs.push({ role: "user", type: "text", text: location.state.userQuery });
      }
      msgs.push(...resultToMessages(location.state.runResult));
      setMessages(msgs);
      return;
    }

    async function loadHistory() {
      try {
        const res = await axiosInstance.get(`/api/conversations/${threadId}/messages`);
        setMessages(res.data.messages.map((m) => ({ role: m.role, type: "text", text: m.content })));
      } catch (error) {
        setErrorMsg(error.response?.data?.detail || "Could not load conversation.");
      } finally {
        setLoadingHistory(false);
      }
    }
    loadHistory();
  }, [threadId]);

  async function sendChat(body, userDisplayText) {
    setErrorMsg("");
    setLoading(true);
    if (userDisplayText) {
      setMessages((prev) => [...prev, { role: "user", type: "text", text: userDisplayText }]);
    }
    try {
      const response = await axiosInstance.post("/api/chat", { thread_id: threadId, ...body });
      setLatestResult(response.data);
      setAwaitingDecision(Boolean(response.data.interrupted));
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

  if (loadingHistory) {
    return <div className="max-w-2xl mx-auto px-6 py-12 text-muted">Loading...</div>;
  }

  return (
    <div className="max-w-2xl mx-auto px-6 py-8 flex flex-col min-h-[calc(100vh-56px)]">
      <div className="flex-1 space-y-4 pb-6">
        {messages.map((msg, i) => <MessageBubble key={i} msg={msg} />)}
        {errorMsg && <p className="text-clay text-sm">{errorMsg}</p>}
      </div>

      <div className="sticky bottom-0 bg-paper pt-4 border-t border-muted/20 space-y-3">
        {awaitingDecision ? (
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
              <input
                value={editText}
                onChange={(e) => setEditText(e.target.value)}
                placeholder="Or describe what to change instead..."
                className="flex-1 px-3 py-2 border border-muted/40 rounded-md bg-white text-ink text-sm focus:outline-none focus:ring-2 focus:ring-accent"
              />
              <button type="submit" disabled={loading}
                className="bg-ink text-paper px-4 py-2 rounded-md text-sm font-medium disabled:opacity-50">
                Send Edit
              </button>
            </form>
          </div>
        ) : (
          <form onSubmit={handleSendMessage} className="flex gap-2">
            <input
              value={input}
              onChange={(e) => setInput(e.target.value)}
              placeholder="Message..."
              className="flex-1 px-3 py-2 border border-muted/40 rounded-md bg-white text-ink text-sm focus:outline-none focus:ring-2 focus:ring-accent"
            />
            <button type="submit" disabled={loading}
              className="bg-ink text-paper px-4 py-2 rounded-md text-sm font-medium disabled:opacity-50">
              Send
            </button>
          </form>
        )}

        {latestResult && !latestResult.interrupted && (
          <Link to={`/report/${threadId}`} className="inline-block text-sm text-accent hover:underline">
            View Report →
          </Link>
        )}
      </div>
    </div>
  );
}

function MessageBubble({ msg }) {
  const isUser = msg.role === "user";
  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div className={`max-w-[80%] rounded-lg px-4 py-2 text-sm ${
        isUser ? "bg-ink text-paper" : "bg-white border border-muted/20 text-ink"
      }`}>
        {msg.type === "interrupt" ? (
          <>
            <p className="font-medium mb-2">Approval needed:</p>
            <pre className="text-xs whitespace-pre-wrap bg-paper p-2 rounded-md overflow-auto text-ink">
              {JSON.stringify(msg.interrupt, null, 2)}
            </pre>
          </>
        ) : (
          <p className="whitespace-pre-wrap">{msg.text}</p>
        )}
      </div>
    </div>
  );
}

export default ChatPage;