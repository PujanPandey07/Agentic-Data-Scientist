import { Link } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

function Dashboard() {
  const { logout } = useAuth();

  return (
    <div className="min-h-screen bg-paper p-8">
      <div className="flex justify-between items-center mb-8">
        <h1 className="font-serif text-2xl text-ink">Dashboard</h1>
        <button onClick={logout} className="text-sm text-muted hover:text-clay">
          Log out
        </button>
      </div>
      <Link
        to="/new-run"
        className="inline-block bg-ink text-paper px-4 py-2 rounded-md font-medium hover:bg-panel transition-colors"
      >
        + New Run
      </Link>
    </div>
  );
}

export default Dashboard;