import { useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import {
  Shield, UploadCloud, Clock, LayoutDashboard,
  Settings, LogOut, User, ChevronRight, Menu,
  Zap,
} from "lucide-react";
import { useAuthStore } from "@/store/authStore";
import toast from "react-hot-toast";

const navItems = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard, color: "#6366f1" },
  { to: "/detect",    label: "Detect",    icon: UploadCloud,     color: "#00f5ff" },
  { to: "/history",   label: "History",   icon: Clock,           color: "#bf00ff" },
  { to: "/settings",  label: "Settings",  icon: Settings,        color: "#00ff88" },
];

interface AppLayoutProps {
  children: React.ReactNode;
}

export function AppLayout({ children }: AppLayoutProps) {
  const location = useLocation();
  const navigate  = useNavigate();
  const { user, logout } = useAuthStore();
  const [mobileOpen, setMobileOpen] = useState(false);

  const handleLogout = async () => {
    await logout();
    toast.success("Signed out");
    navigate("/login");
  };

  const Sidebar = (
    <motion.aside
      initial={{ x: -20, opacity: 0 }}
      animate={{ x: 0, opacity: 1 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      className="flex h-full w-64 flex-col"
      style={{
        background: "rgba(10, 15, 30, 0.95)",
        borderRight: "1px solid rgba(255,255,255,0.06)",
        backdropFilter: "blur(20px)",
      }}
    >
      {/* Logo */}
      <div className="flex items-center gap-3 px-6 py-6" style={{ borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
        <motion.div
          animate={{ boxShadow: ["0 0 15px rgba(99,102,241,0.4)", "0 0 30px rgba(99,102,241,0.7)", "0 0 15px rgba(99,102,241,0.4)"] }}
          transition={{ duration: 2.5, repeat: Infinity }}
          className="flex h-10 w-10 items-center justify-center rounded-xl"
          style={{ background: "linear-gradient(135deg, #4f46e5, #7c3aed)" }}
        >
          <Shield className="h-5 w-5 text-white" />
        </motion.div>
        <div>
          <p className="text-base font-bold tracking-tight text-white">DeepGuard</p>
          <p className="text-xs text-gray-500">Detection Platform</p>
        </div>
      </div>

      {/* Nav */}
      <nav className="flex-1 space-y-1 px-3 py-5">
        {navItems.map(({ to, label, icon: Icon, color }, i) => {
          const isActive = location.pathname === to;
          return (
            <motion.div
              key={to}
              initial={{ x: -20, opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              transition={{ delay: i * 0.06, duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
            >
              <Link
                to={to}
                onClick={() => setMobileOpen(false)}
                className="group relative flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-all duration-200"
                style={
                  isActive
                    ? { background: `${color}18`, color: color, boxShadow: `0 0 20px ${color}20` }
                    : { color: "#64748b" }
                }
              >
                {isActive && (
                  <motion.div
                    layoutId="nav-indicator"
                    className="absolute inset-0 rounded-xl"
                    style={{ background: `${color}12`, border: `1px solid ${color}30` }}
                    transition={{ type: "spring", stiffness: 400, damping: 30 }}
                  />
                )}
                <div
                  className="relative z-10 flex h-7 w-7 items-center justify-center rounded-lg transition-all"
                  style={isActive ? { background: `${color}22` } : {}}
                >
                  <Icon
                    className="h-4 w-4 transition-all"
                    style={isActive ? { color, filter: `drop-shadow(0 0 6px ${color})` } : {}}
                  />
                </div>
                <span className="relative z-10 flex-1">{label}</span>
                {isActive && (
                  <ChevronRight
                    className="relative z-10 h-3.5 w-3.5 opacity-60"
                    style={{ color }}
                  />
                )}

                {/* Hover glow */}
                {!isActive && (
                  <div
                    className="absolute inset-0 rounded-xl opacity-0 transition-opacity duration-200 group-hover:opacity-100"
                    style={{ background: "rgba(255,255,255,0.04)" }}
                  />
                )}
              </Link>
            </motion.div>
          );
        })}
      </nav>

      {/* Status indicator */}
      <div className="px-4 py-3">
        <div
          className="flex items-center gap-2 rounded-xl px-3 py-2.5"
          style={{ background: "rgba(0,255,136,0.06)", border: "1px solid rgba(0,255,136,0.12)" }}
        >
          <motion.div
            className="h-2 w-2 rounded-full"
            style={{ background: "#00ff88" }}
            animate={{ opacity: [1, 0.4, 1] }}
            transition={{ duration: 1.5, repeat: Infinity }}
          />
          <Zap className="h-3.5 w-3.5" style={{ color: "#00ff88" }} />
          <span className="text-xs font-medium" style={{ color: "#00ff88" }}>System Online</span>
        </div>
      </div>

      {/* User footer */}
      <div className="px-3 py-4" style={{ borderTop: "1px solid rgba(255,255,255,0.05)" }}>
        <div className="flex items-center gap-3 rounded-xl px-3 py-3" style={{ background: "rgba(255,255,255,0.03)" }}>
          <div
            className="relative flex h-9 w-9 shrink-0 items-center justify-center rounded-full"
            style={{ background: "linear-gradient(135deg, #4f46e5, #7c3aed)" }}
          >
            <User className="h-4 w-4 text-white" />
            <motion.div
              className="absolute -bottom-0.5 -right-0.5 h-3 w-3 rounded-full border-2"
              style={{ background: "#00ff88", borderColor: "#0a0f1e" }}
              animate={{ scale: [1, 1.2, 1] }}
              transition={{ duration: 2, repeat: Infinity }}
            />
          </div>
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-semibold text-gray-100">{user?.username ?? "Guest"}</p>
            <p className="truncate text-xs text-gray-500">{user?.email ?? ""}</p>
          </div>
          <motion.button
            onClick={handleLogout}
            whileHover={{ scale: 1.1 }}
            whileTap={{ scale: 0.9 }}
            className="flex h-8 w-8 items-center justify-center rounded-lg transition-all"
            style={{ color: "#64748b" }}
            title="Sign out"
            aria-label="Sign out"
          >
            <LogOut className="h-3.5 w-3.5 hover:text-red-400 transition-colors" />
          </motion.button>
        </div>
      </div>
    </motion.aside>
  );

  return (
    <div
      className="flex h-screen overflow-hidden"
      style={{ background: "#030712" }}
    >
      {/* Desktop sidebar */}
      <div className="hidden md:block">
        {Sidebar}
      </div>

      {/* Mobile sidebar */}
      <AnimatePresence>
        {mobileOpen && (
          <>
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 z-40 bg-black/60 backdrop-blur-sm md:hidden"
              onClick={() => setMobileOpen(false)}
            />
            <motion.div
              initial={{ x: -280 }}
              animate={{ x: 0 }}
              exit={{ x: -280 }}
              transition={{ type: "spring", stiffness: 400, damping: 40 }}
              className="fixed inset-y-0 left-0 z-50 md:hidden"
            >
              {Sidebar}
            </motion.div>
          </>
        )}
      </AnimatePresence>

      {/* Main area */}
      <div className="flex flex-1 flex-col overflow-hidden">
        {/* Mobile top bar */}
        <div
          className="flex items-center gap-4 px-4 py-3 md:hidden"
          style={{ background: "rgba(10,15,30,0.95)", borderBottom: "1px solid rgba(255,255,255,0.06)" }}
        >
          <button onClick={() => setMobileOpen(true)} aria-label="Open menu">
            <Menu className="h-5 w-5 text-gray-400" />
          </button>
          <Shield className="h-5 w-5 text-brand-400" />
          <span className="text-sm font-bold text-white">DeepGuard</span>
        </div>

        {/* Content */}
        <main className="flex-1 overflow-auto">
          <motion.div
            key={location.pathname}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            className="min-h-full"
          >
            {children}
          </motion.div>
        </main>
      </div>
    </div>
  );
}
