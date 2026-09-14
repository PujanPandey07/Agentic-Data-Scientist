import axios from "axios";

// This holds the current access token in memory only (a plain JS variable,
// not localStorage). It resets to null on every page refresh — that's
// intentional, since the backend's refresh token lives in an httpOnly
// cookie the browser sends automatically, so we can silently re-fetch
// a new access token on load instead of ever persisting it to disk.
let accessToken = null;

// Called from your login page after a successful /api/auth/login response,
// and later from the refresh logic below whenever a new token is issued.
export function setAccessToken(token) {
  accessToken = token;
}

// The actual axios instance every part of the app will import and use
// instead of calling axios directly — this is what makes the token
// attachment and refresh logic automatic and centralized.
const axiosInstance = axios.create({
  baseURL: "http://127.0.0.1:8000",
  withCredentials: true, // sends the httpOnly refresh cookie automatically on every request
});

// REQUEST interceptor: runs before every outgoing request.
// Attaches the current access token, if we have one.
axiosInstance.interceptors.request.use((config) => {
  if (accessToken) {
    config.headers.Authorization = `Bearer ${accessToken}`;
  }
  return config;
});

// RESPONSE interceptor: runs after every response comes back.
// If we get a 401 (access token expired/invalid), try to silently get a
// new access token using the refresh cookie, then retry the original
// request once. This is what makes token expiry invisible to the user.
axiosInstance.interceptors.response.use(
  (response) => response, // happy path — pass successful responses through untouched
  async (error) => {
    const originalRequest = error.config;

    // Only attempt this once per request (the `_retry` flag prevents an
    // infinite loop if refresh itself keeps failing).
    if (error.response?.status === 401 && !originalRequest._retry) {
      originalRequest._retry = true;

      try {
        // Calls your backend's refresh endpoint. withCredentials above
        // ensures the httpOnly refresh cookie gets sent along with this.
        const refreshResponse = await axios.post(
          "http://127.0.0.1:8000/api/auth/refresh",
          {},
          { withCredentials: true }
        );

        const newAccessToken = refreshResponse.data.access_token;
        setAccessToken(newAccessToken);

        // Update the header on the original failed request and retry it.
        originalRequest.headers.Authorization = `Bearer ${newAccessToken}`;
        return axiosInstance(originalRequest);
      } catch (refreshError) {
        // Refresh itself failed — the user's session is truly over
        // (refresh token expired/invalid). Clear the token and let the
        // error propagate so the app can redirect to login.
        setAccessToken(null);
        return Promise.reject(refreshError);
      }
    }

    return Promise.reject(error);
  }
);

export default axiosInstance;