import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import axiosInstance from "../api/axiosstance";
import { useAuth } from "../context/useAuth";
import AuthLayout from "../components/authlayout";

function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const navigate = useNavigate();
  const { login } = useAuth();

  async function handleSubmit(e) {
    e.preventDefault();
    setErrorMsg("");
    try {
      const response = await axiosInstance.post("/api/auth/login", {
        email,
        password,
      });
      login(response.data.access_token);
      navigate("/");
    } catch (error) {
      setErrorMsg(error.response?.data?.detail || "Something went wrong.");
    }
  }

  return (
    <AuthLayout
      title="Welcome back"
      subtitle="Log in to continue your analysis."
      footer={
        <p>
          Don't have an account?{" "}
          <Link to="/register" className="text-accent hover:underline">
            Register
          </Link>
        </p>
      }
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="block text-sm text-ink mb-1">Email</label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            className="w-full px-3 py-2 border border-muted/40 rounded-md bg-white text-ink focus:outline-none focus:ring-2 focus:ring-accent"
            required
          />
        </div>
        <div>
          <label className="block text-sm text-ink mb-1">Password</label>
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full px-3 py-2 border border-muted/40 rounded-md bg-white text-ink focus:outline-none focus:ring-2 focus:ring-accent"
            required
          />
        </div>
        {errorMsg && <p className="text-clay text-sm">{errorMsg}</p>}
        <button
          type="submit"
          className="w-full bg-ink text-paper py-2 rounded-md font-medium hover:bg-panel transition-colors"
        >
          Log In
        </button>
      </form>
    </AuthLayout>
  );
}

export default LoginPage;