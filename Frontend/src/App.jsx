import { BrowserRouter, Routes, Route } from "react-router-dom";
import { AuthProvider } from "./context/AuthContext";
import ProtectedRoute from "./components/ProtectedRoute";
import Sidebar from "./components/SideBar";
import LoginPage from "./pages/Loginpage";
import RegisterPage from "./pages/Registerpage";
import Workspace from "./pages/Workspace";

function withSidebar(Page) {
  return (
    <ProtectedRoute>
      <div className="flex">
        <Sidebar />
        <div className="flex-1">
          <Page />
        </div>
      </div>
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
        </Routes>
      </BrowserRouter>
    </AuthProvider>
  );
}

export default App;