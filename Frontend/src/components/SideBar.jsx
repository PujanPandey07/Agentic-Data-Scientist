import { useEffect, useState } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router-dom";
import axiosInstance from "../api/axiosstance";
import { useAuth } from "../context/useAuth";

function formatDate(value) {
  const date = new Date(value);
  const now = new Date();
  const diffDays = Math.floor((now - date) / 86400000);
  if (diffDays === 0) return "Today";
  if (diffDays === 1) return "Yesterday";
  if (diffDays < 7) return date.toLocaleDateString(undefined, { weekday: "long" });
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function groupConversations(conversations) {
  const groups = {};
  for (const c of conversations) {
    const label = formatDate(c.created_at);
    if (!groups[label]) groups[label] = [];
    groups[label].push(c);
  }
  return groups;
}

function TrashIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-3.5 w-3.5" viewBox="0 0 24 24"
      fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
      <path d="M10 11v6M14 11v6" />
      <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
    </svg>
  );
}

function PlusIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 24 24"
      fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <line x1="12" y1="5" x2="12" y2="19" />
      <line x1="5" y1="12" x2="19" y2="12" />
    </svg>
  );
}

function GearIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 24 24"
      fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="3" />
      <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 1 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 1 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9z" />
    </svg>
  );
}

function LogoutIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" className="h-4 w-4" viewBox="0 0 24 24"
      fill="none" stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round">
      <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
      <polyline points="16 17 21 12 16 7" />
      <line x1="21" y1="12" x2="9" y2="12" />
    </svg>
  );
}

function Sidebar({ onCloseMobile }) {
  const [conversations, setConversations] = useState([]);
  const { threadId: activeThreadId } = useParams();
  const { logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    axiosInstance
      .get("/api/conversations")
      .then((res) => setConversations(res.data.conversations))
      .catch(() => {});
  }, [location.pathname]);

  async function handleDelete(e, threadId) {
    e.preventDefault();
    e.stopPropagation();
    const previous = conversations;
    setConversations((prev) => prev.filter((c) => c.thread_id !== threadId));
    if (activeThreadId === threadId) navigate("/");
    try {
      await axiosInstance.delete(`/api/conversations/${threadId}`);
    } catch {
      setConversations(previous);
    }
  }

  const groups = groupConversations(conversations);
  const groupKeys = Object.keys(groups);

  return (
    <aside
      className="w-64 md:w-60 shrink-0 h-full flex flex-col shadow-xl md:shadow-none"
      style={{ background: "#171717", color: "#ececec" }}
    >
      {/* Logo / App name + Mobile Close Button */}
      <div className="px-4 pt-5 pb-3 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div
            className="w-7 h-7 rounded-lg flex items-center justify-center text-xs font-bold shrink-0"
            style={{ background: "#cc785c", color: "#fff" }}
          >
            DS
          </div>
          <span className="font-semibold text-sm tracking-tight" style={{ color: "#ececec" }}>
            AI Data Scientist
          </span>
        </div>
        {onCloseMobile && (
          <button
            onClick={onCloseMobile}
            className="p-1 rounded md:hidden text-neutral-400 hover:text-white transition-colors"
            aria-label="Close sidebar"
          >
            <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
            </svg>
          </button>
        )}
      </div>

      {/* New Chat button */}
      <div className="px-3 pb-3">
        <button
          onClick={() => {
            navigate("/");
            onCloseMobile?.();
          }}
          className="w-full flex items-center gap-2 px-3 py-2 rounded-lg text-sm font-medium transition-colors"
          style={{ background: "#2a2a2a", color: "#ececec" }}
          onMouseEnter={(e) => (e.currentTarget.style.background = "#333")}
          onMouseLeave={(e) => (e.currentTarget.style.background = "#2a2a2a")}
        >
          <PlusIcon />
          New conversation
        </button>
      </div>

      {/* Conversation history */}
      <div className="flex-1 overflow-y-auto px-2 space-y-4">
        {groupKeys.length === 0 && (
          <p className="text-xs px-2 py-2" style={{ color: "#666" }}>
            No conversations yet.
          </p>
        )}
        {groupKeys.map((label) => (
          <div key={label}>
            <p className="text-xs font-medium px-2 py-1 uppercase tracking-wider" style={{ color: "#666" }}>
              {label}
            </p>
            <div className="space-y-0.5">
              {groups[label].map((c) => {
                const isActive = c.thread_id === activeThreadId;
                return (
                  <div key={c.thread_id} className="group relative">
                    <Link
                      to={`/c/${c.thread_id}`}
                      onClick={() => onCloseMobile?.()}
                      className="flex items-center justify-between px-2 py-2 rounded-lg text-sm transition-colors"
                      style={{
                        background: isActive ? "#2a2a2a" : "transparent",
                        color: isActive ? "#ececec" : "#aaa",
                      }}
                      onMouseEnter={(e) => {
                        if (!isActive) e.currentTarget.style.background = "#222";
                        e.currentTarget.style.color = "#ececec";
                      }}
                      onMouseLeave={(e) => {
                        if (!isActive) e.currentTarget.style.background = "transparent";
                        e.currentTarget.style.color = isActive ? "#ececec" : "#aaa";
                      }}
                    >
                      <span className="block truncate flex-1 pr-1 text-xs leading-5">
                        {c.title || "Untitled analysis"}
                      </span>
                      <button
                        onClick={(e) => handleDelete(e, c.thread_id)}
                        className="opacity-0 group-hover:opacity-100 shrink-0 p-1 rounded transition-opacity"
                        style={{ color: "#888" }}
                        onMouseEnter={(e) => (e.currentTarget.style.color = "#e05252")}
                        onMouseLeave={(e) => (e.currentTarget.style.color = "#888")}
                        title="Delete"
                        aria-label="Delete conversation"
                      >
                        <TrashIcon />
                      </button>
                    </Link>
                  </div>
                );
              })}
            </div>
          </div>
        ))}
      </div>

      {/* Bottom actions */}
      <div className="px-2 pb-4 pt-2" style={{ borderTop: "1px solid #2a2a2a" }}>
        <Link
          to="/settings/api-keys"
          className="flex items-center gap-2.5 px-2 py-2 rounded-lg text-sm transition-colors"
          style={{ color: "#aaa" }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "#222";
            e.currentTarget.style.color = "#ececec";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "transparent";
            e.currentTarget.style.color = "#aaa";
          }}
        >
          <GearIcon />
          AI Model settings
        </Link>
        <button
          onClick={logout}
          className="flex items-center gap-2.5 px-2 py-2 rounded-lg text-sm w-full transition-colors mt-0.5"
          style={{ color: "#aaa" }}
          onMouseEnter={(e) => {
            e.currentTarget.style.background = "#222";
            e.currentTarget.style.color = "#ececec";
          }}
          onMouseLeave={(e) => {
            e.currentTarget.style.background = "transparent";
            e.currentTarget.style.color = "#aaa";
          }}
        >
          <LogoutIcon />
          Log out
        </button>
      </div>
    </aside>
  );
}

export default Sidebar;