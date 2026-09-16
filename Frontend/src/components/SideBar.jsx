import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import axiosInstance from "../api/axiosstance";
import { useAuth } from "../context/AuthContext";

function Sidebar() {
  const [conversations, setConversations] = useState([]);
  const { threadId: activeThreadId } = useParams();
  const { logout } = useAuth();
  const navigate = useNavigate();

  useEffect(() => {
    axiosInstance
      .get("/api/conversations")
      .then((res) => setConversations(res.data.conversations))
      .catch(() => {});
  }, [activeThreadId]);

  return (
    <aside className="w-64 shrink-0 bg-ink text-paper h-screen flex flex-col">
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
          <Link
            key={c.thread_id}
            to={`/c/${c.thread_id}`}
            className={`block px-2 py-2 rounded-md text-sm truncate ${
              c.thread_id === activeThreadId
                ? "bg-panel text-paper"
                : "text-muted hover:bg-panel/50 hover:text-paper"
            }`}
          >
            {c.title || `Run ${c.thread_id.slice(0, 8)}`}
          </Link>
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