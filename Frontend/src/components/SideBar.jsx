import { useEffect, useState } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router-dom";
import axiosInstance from "../api/axiosstance";
import { useAuth } from "../context/useAuth";

function Sidebar() {
  const [conversations, setConversations] = useState([]);
  const { threadId: activeThreadId } = useParams();
  const { logout } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

  function formatDate(value) {
    return new Date(value).toLocaleDateString(undefined, {
      month: "short",
      day: "numeric",
    });
  }

  useEffect(() => {
    axiosInstance
      .get("/api/conversations")
      .then((res) => setConversations(res.data.conversations))
      .catch(() => {});
  }, [location.pathname]);

  async function handleDelete(e, threadId) {
    // Stop the click from also navigating via the parent <Link>.
    e.preventDefault();
    e.stopPropagation();

    // Optimistic update — remove from list immediately so the UI
    // feels instant. We'll put it back if the server call fails.
    const previous = conversations;
    setConversations((prev) => prev.filter((c) => c.thread_id !== threadId));

    // If the user is currently viewing the conversation being deleted,
    // send them home before it disappears.
    if (activeThreadId === threadId) {
      navigate("/");
    }

    try {
      await axiosInstance.delete(`/api/conversations/${threadId}`);
    } catch {
      // Restore the list if something went wrong on the server.
      setConversations(previous);
    }
  }

  return (
    <aside className="w-64 shrink-0 bg-ink text-paper h-full flex flex-col">
      <div className="px-4 py-4 border-b border-panel">
        <p className="font-serif text-lg">AI Data Scientist</p>
      </div>

      <div className="px-3 py-3">
        <button
          onClick={() => navigate("/")}
          className="w-full bg-panel text-paper py-2 rounded-md text-sm font-medium hover:bg-panel/80"
        >
          + New Chat
        </button>
      </div>

      <div className="flex-1 overflow-y-auto px-3 space-y-1">
        <p className="text-xs text-muted uppercase tracking-wide mb-1 px-1">History</p>
        {conversations.map((c) => (
          <div key={c.thread_id} className="group relative">
            <Link
              to={`/c/${c.thread_id}`}
              className={`flex items-start justify-between px-2 py-2 rounded-md text-sm ${
                c.thread_id === activeThreadId
                  ? "bg-panel text-paper"
                  : "text-muted hover:bg-panel/50 hover:text-paper"
              }`}
            >
              <div className="min-w-0 flex-1 pr-1">
                <span className="block truncate text-paper">
                  {c.title || "Untitled analysis"}
                </span>
                <span className="block truncate text-xs text-muted mt-0.5">
                  {formatDate(c.created_at)} · Dataset {c.dataset_id.slice(0, 8)}
                </span>
              </div>

              {/* Trash icon — only visible on hover, sits on the right */}
              <button
                onClick={(e) => handleDelete(e, c.thread_id)}
                className="opacity-0 group-hover:opacity-100 shrink-0 p-1 rounded hover:text-clay transition-opacity"
                title="Delete conversation"
                aria-label="Delete conversation"
              >
                <svg
                  xmlns="http://www.w3.org/2000/svg"
                  className="h-3.5 w-3.5"
                  viewBox="0 0 24 24"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth={2}
                  strokeLinecap="round"
                  strokeLinejoin="round"
                >
                  <polyline points="3 6 5 6 21 6" />
                  <path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" />
                  <path d="M10 11v6M14 11v6" />
                  <path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" />
                </svg>
              </button>
            </Link>
          </div>
        ))}
        {conversations.length === 0 && (
          <p className="text-xs text-muted px-1">No runs yet.</p>
        )}
      </div>

      <div className="px-3 py-3 border-t border-panel">
        <button onClick={logout} className="text-sm text-muted hover:text-clay">
          Log out
        </button>
      </div>
    </aside>
  );
}

export default Sidebar;