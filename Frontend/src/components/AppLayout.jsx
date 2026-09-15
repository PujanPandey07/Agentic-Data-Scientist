import Navbar from "./Navbar";
import Sidebar from "./SideBar";

function AppLayout({ children }) {
  return (
    <div className="min-h-screen bg-paper">
      <Navbar />
      <div className="flex">
        <Sidebar />
        <main className="flex-1 overflow-y-auto">{children}</main>
      </div>
    </div>
  );
}

export default AppLayout;