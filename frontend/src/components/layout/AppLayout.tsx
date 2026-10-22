import { Link, useLocation, useNavigate } from "react-router-dom";
import { Shield, UploadCloud, Clock, LayoutDashboard, LogOut, Settings, User } from "lucide-react";
import { useAuthStore } from "@/store/authStore";
import toast from "react-hot-toast";
import { clsx } from "clsx";

const navItems = [
  { to: "/dashboard",  label: "Dashboard",  icon: LayoutDashboard },
  { to: "/detect",     label: "Detect",     icon: UploadCloud },
  { to: "/history",    label: "History",    icon: Clock },
  { to: "/settings",   label: "Settings",   icon: Settings },
];

export function AppLayout({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  const navigate = useNavigate();
  const { user, logout } = useAuthStore();

  const handleLogout = async () => {
    await logout();
    toast.success("Logged out");
    navigate("/login");
  };

  return (
    <div className="flex h-screen bg-gray-950">
      {/* Sidebar */}
      <aside className="flex w-64 flex-col border-r border-gray-800 bg-gray-900">
        {/* Logo */}
        <div className="flex items-center gap-3 border-b border-gray-800 px-6 py-5">
          <div className="flex h-9 w-9 items-center justify-center rounded-lg bg-brand-600">
            <Shield className="h-5 w-5 text-white" />
          </div>
          <span className="text-lg font-bold tracking-tight text-white">DeepGuard</span>
        </div>

        {/* Navigation */}
        <nav className="flex-1 space-y-1 px-3 py-4">
          {navItems.map(({ to, label, icon: Icon }) => (
            <Link
              key={to}
              to={to}
              className={clsx(
                "flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition",
                location.pathname === to
                  ? "bg-brand-600/20 text-brand-400"
                  : "text-gray-400 hover:bg-gray-800 hover:text-gray-100"
              )}
            >
              <Icon className="h-4 w-4" />
              {label}
            </Link>
          ))}
        </nav>

        {/* User footer */}
        <div className="border-t border-gray-800 p-4">
          <div className="flex items-center gap-3">
            <div className="flex h-9 w-9 items-center justify-center rounded-full bg-brand-600/20">
              <User className="h-4 w-4 text-brand-400" />
            </div>
            <div className="min-w-0 flex-1">
              <p className="truncate text-sm font-medium text-gray-100">{user?.username ?? "User"}</p>
              <p className="truncate text-xs text-gray-500">{user?.email ?? ""}</p>
            </div>
            <button
              onClick={handleLogout}
              title="Logout"
              className="rounded-lg p-1.5 text-gray-500 transition hover:bg-gray-800 hover:text-gray-300"
            >
              <LogOut className="h-4 w-4" />
            </button>
          </div>
        </div>
      </aside>

      {/* Main content */}
      <main className="flex-1 overflow-auto">
        {children}
      </main>
    </div>
  );
}
