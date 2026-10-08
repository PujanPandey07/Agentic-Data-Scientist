import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import ProtectedRoute from "./components/ProtectedRoute";
import Sidebar from "./components/SideBar";
import LoginPage from "./pages/Loginpage";
import RegisterPage from "./pages/Registerpage";
import Workspace from "./pages/Workspace";
import ApiKeySettings from "./pages/ApiKeySettings";

import { LayoutProvider, useLayout } from "./context/LayoutContext";

function Shell({ children }) {
  const { sidebarOpen, setSidebarOpen } = useLayout();

  return (
    <div className="flex h-screen overflow-hidden relative">
      {/* Mobile overlay backdrop */}
      {sidebarOpen && (
        <div
          onClick={() => setSidebarOpen(false)}
          className="fixed inset-0 bg-black/50 z-40 md:hidden transition-opacity"
        />
      )}

      {/* Sidebar - drawer on mobile, static on desktop */}
      <div
        className={`fixed inset-y-0 left-0 z-50 md:static md:z-auto transition-transform duration-200 ease-in-out ${
          sidebarOpen ? "translate-x-0" : "-translate-x-full md:translate-x-0"
        }`}
      >
        <Sidebar onCloseMobile={() => setSidebarOpen(false)} />
      </div>

      <div className="flex-1 overflow-hidden min-w-0">
        {children}
      </div>
    </div>
  );
}

function withSidebar(Page) {
  return (
    <ProtectedRoute>
      <LayoutProvider>
        <Shell>
          <Page />
        </Shell>
      </LayoutProvider>
    </ProtectedRoute>
  );
}

function App() {
  return (
    <AuthProvider>
      <BrowserRouter>
        <Routes>
          <Route path="/login" element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />
          <Route path="/" element={withSidebar(Workspace)} />
          <Route path="/c/:threadId" element={withSidebar(Workspace)} />
          <Route path="/settings/api-keys" element={withSidebar(ApiKeySettings)} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;