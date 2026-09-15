import { Link, useLocation } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

function Navbar() {
  const { logout } = useAuth();
  const location = useLocation();

  function isActive(path) {
    return location.pathname === path ? "text-ink" : "text-muted";
  }

  return (
    <nav className="bg-white border-b border-muted/20 px-6 py-3 flex items-center justify-between">
      <Link to="/" className="font-serif text-lg text-ink">
        AI Data Scientist
      </Link>
      <div className="flex items-center gap-6 text-sm">
        <Link to="/" className={`hover:text-ink ${isActive("/")}`}>
          Dashboard
        </Link>
        <Link to="/new-run" className={`hover:text-ink ${isActive("/new-run")}`}>
          New Run
        </Link>
        <button onClick={logout} className="text-muted hover:text-clay">
          Log out
        </button>
      </div>
    </nav>
  );
}

export default Navbar;