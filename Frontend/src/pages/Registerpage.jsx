import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import axiosInstance from "../api/axiosstance";
import AuthLayout from "../components/authlayout";

function checkPasswordStrength(password) {
  const checks = {
    length: password.length >= 8,
    upper: /[A-Z]/.test(password),
    lower: /[a-z]/.test(password),
    number: /[0-9]/.test(password),
    special: /[^A-Za-z0-9]/.test(password),
  };
  const score = Object.values(checks).filter(Boolean).length;
  return { checks, score, valid: score === 5 };
}

function RegisterPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [errorMsg, setErrorMsg] = useState("");
  const navigate = useNavigate();

  const { checks, score, valid } = checkPasswordStrength(password);
  const strengthLabels = ["Very weak", "Weak", "Fair", "Good", "Strong"];
  const strengthColors = ["bg-clay", "bg-clay", "bg-yellow-500", "bg-yellow-500", "bg-accent"];

  async function handleSubmit(e) {
    e.preventDefault();
    setErrorMsg("");
    if (!valid) {
      setErrorMsg("Password doesn't meet all requirements yet.");
      return;
    }
    try {
      await axiosInstance.post("/api/auth/register", { email, password });
      navigate("/login");
    } catch (error) {
      setErrorMsg(error.response?.data?.detail || "Something went wrong.");
    }
  }

  return (
    <AuthLayout
      title="Create your account"
      subtitle="Start building your first pipeline."
      footer={
        <p>
          Already have an account?{" "}
          <Link to="/login" className="text-accent hover:underline">
            Log in
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
          {password && (
            <div className="mt-2">
              <div className="flex gap-1 mb-1">
                {[0, 1, 2, 3, 4].map((i) => (
                  <div
                    key={i}
                    className={`h-1 flex-1 rounded ${i < score ? strengthColors[score - 1] : "bg-muted/20"}`}
                  />
                ))}
              </div>
              <p className="text-xs text-muted mb-1">{strengthLabels[Math.max(score - 1, 0)]}</p>
              <ul className="text-xs text-muted space-y-0.5">
                <li className={checks.length ? "text-accent" : ""}>• At least 8 characters</li>
                <li className={checks.upper ? "text-accent" : ""}>• One uppercase letter</li>
                <li className={checks.lower ? "text-accent" : ""}>• One lowercase letter</li>
                <li className={checks.number ? "text-accent" : ""}>• One number</li>
                <li className={checks.special ? "text-accent" : ""}>• One special character</li>
              </ul>
            </div>
          )}
        </div>
        {errorMsg && <p className="text-clay text-sm">{errorMsg}</p>}
        <button
          type="submit"
          disabled={!valid}
          className="w-full bg-ink text-paper py-2 rounded-md font-medium hover:bg-panel transition-colors disabled:opacity-50"
        >
          Create Account
        </button>
      </form>
    </AuthLayout>
  );
}

export default RegisterPage;