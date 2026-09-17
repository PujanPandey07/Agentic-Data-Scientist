import { useState, useEffect } from "react";
import axios from "axios";
import { setAccessToken as setAxiosAccessToken } from "../api/axiosstance";
import { AuthContext } from "./AuthContextValue";

export function AuthProvider({ children }) {
  const [accessToken, setAccessTokenState] = useState(null);
  const [loading, setLoading] = useState(true);

  function login(token) {
    setAccessTokenState(token);
    setAxiosAccessToken(token);
  }

  function logout() {
    setAccessTokenState(null);
    setAxiosAccessToken(null);
  }

  useEffect(() => {
    async function tryRefresh() {
      try {
        const response = await axios.post(
          "http://127.0.0.1:8000/api/auth/refresh",
          {},
          { withCredentials: true }
        );
        login(response.data.access_token);
      } catch {
        // no valid refresh cookie — not logged in, that's fine
      } finally {
        setLoading(false);
      }
    }
    tryRefresh();
  }, []);

  return (
    <AuthContext.Provider value={{ accessToken, login, logout, loading }}>
      {children}
    </AuthContext.Provider>
  );
}