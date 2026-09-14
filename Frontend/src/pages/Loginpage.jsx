import { useState } from "react";
import axiosInstance, { setAccessToken } from "../api/axiosstance";

function LoginPage() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");

  async function handleSubmit(e) {
    e.preventDefault(); // stops the browser's default full-page-reload-on-submit behavior

    try {
      const response = await axiosInstance.post("/api/auth/login", {
        email,
        password,
      });

      // The backend returns { access_token: "..." } — store it in memory
      // via the setter we wrote earlier, so every future axios call
      // automatically attaches it.
      setAccessToken(response.data.access_token);

      // Placeholder for now — once a dashboard page exists, this becomes
      // a real navigate("/dashboard") call instead.
      console.log("Logged in! Token:", response.data.access_token);
    } catch (error) {
      // error.response is what axios attaches when the backend responds
      // with an error status (like our 401 "Invalid email or password").
      // error.response?.data?.detail matches your backend's HTTPException
      // detail field exactly.
      console.log("Login failed:", error.response?.data?.detail);
    }
  }

  return (
    <form onSubmit={handleSubmit}>
      <input
        type="email"
        value={email}
        onChange={(e) => setEmail(e.target.value)}
        placeholder="Email"
      />
      <input
        type="password"
        value={password}
        onChange={(e) => setPassword(e.target.value)}
        placeholder="Password"
      />
      <button type="submit">Log In</button>
    </form>
  );
}

export default LoginPage;