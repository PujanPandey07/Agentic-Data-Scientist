import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import ProtectedRoute from "./components/ProtectedRoute";
import AppLayout from "./components/AppLayout";
import LoginPage from "./pages/Loginpage";
import RegisterPage from "./pages/Registerpage";
import Dashboard from "./pages/Dashboard";
import CreateRunPage from "./pages/CreateRun";
import ChatPage from "./pages/ChatPage";
import ReportPage from "./pages/ReportPage";

function withLayout(Page) {
  return (
    <ProtectedRoute>
      <AppLayout>
        <Page />
      </AppLayout>
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
          <Route path="/" element={withLayout(Dashboard)} />
          <Route path="/new-run" element={withLayout(CreateRunPage)} />
          <Route path="/chat/:threadId" element={withLayout(ChatPage)} />
          <Route path="/report/:threadId" element={withLayout(ReportPage)} />
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;