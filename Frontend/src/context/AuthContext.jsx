import { useState, useEffect } from "react";
import axiosInstance, { setAccessToken as setAxiosAccessToken } from "../api/axiosstance";
import { AuthContext } from "./AuthContextValue";

export function AuthProvider({ children }) {
  const [accessToken, setAccessTokenState] = useState(null);
  const [loading, setLoading] = useState(true);

  function login(token) {
    setAccessTokenState(token);
    setAxiosAccessToken(token);
  }

  async function logout() {
    try {
      await axiosInstance.post("/api/auth/logout");
    } catch {
      // Backend logout failed or unreachable, continue clearing local state
    } finally {
      setAccessTokenState(null);
      setAxiosAccessToken(null);
    }
  }

  useEffect(() => {
    async function tryRefresh() {
      try {
        const response = await axiosInstance.post("/api/auth/refresh");
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