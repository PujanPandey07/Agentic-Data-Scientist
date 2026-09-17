import { Navigate } from "react-router-dom";
import { useAuth } from "../context/useAuth";

function ProtectedRoute({ children }) {
  const { accessToken, loading } = useAuth();

  if (loading) return <p className="p-8 text-muted">Loading...</p>;
  if (!accessToken) return <Navigate to="/login" replace />;

  return children;
}

export default ProtectedRoute;