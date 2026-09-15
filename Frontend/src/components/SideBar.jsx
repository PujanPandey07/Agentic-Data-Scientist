import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import axiosInstance from "../api/axiosstance";

function Sidebar() {
  const [conversations, setConversations] = useState([]);
  const { threadId: activeThreadId } = useParams();

  useEffect(() => {
    axiosInstance
      .get("/api/conversations")
      .then((res) => setConversations(res.data.conversations))
      .catch(() => {});
  }, []);

  return (
    <aside className="w-64 shrink-0 bg-ink text-paper h-[calc(100vh-56px)] overflow-y-auto px-3 py-4">
      <Link
        to="/new-run"
        className="block w-full text-center bg-panel text-paper py-2 rounded-md text-sm font-medium mb-4 hover:bg-panel/80"
      >
        + New Run
      </Link>
      <p className="text-xs text-muted uppercase tracking-wide mb-2 px-1">History</p>
      <div className="space-y-1">
        {conversations.map((c) => (
          <Link
            key={c.thread_id}
            to={`/chat/${c.thread_id}`}
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
    </aside>
  );
}

export default Sidebar;