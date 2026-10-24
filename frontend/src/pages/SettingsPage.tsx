import { useState, FormEvent } from "react";
import { motion } from "framer-motion";
import {
  User, Shield, Lock, Eye, EyeOff, Save,
  Calendar, Mail, BadgeCheck, Settings,
} from "lucide-react";
import { useAuthStore } from "@/store/authStore";
import { apiClient } from "@/lib/api";
import toast from "react-hot-toast";

export function SettingsPage() {
  const { user } = useAuthStore();
  const [passwords, setPasswords] = useState({
    current_password: "", new_password: "", confirm: "",
  });
  const [isChanging, setIsChanging] = useState(false);
  const [showCurrent, setShowCurrent] = useState(false);
  const [showNew, setShowNew] = useState(false);

  const handlePasswordChange = async (e: FormEvent) => {
    e.preventDefault();
    if (passwords.new_password !== passwords.confirm) {
      toast.error("New passwords do not match");
      return;
    }
    setIsChanging(true);
    try {
      await apiClient.post("/auth/change-password", {
        current_password: passwords.current_password,
        new_password:     passwords.new_password,
      });
      toast.success("Password updated");
      setPasswords({ current_password: "", new_password: "", confirm: "" });
    } catch {
      // handled by interceptor
    } finally {
      setIsChanging(false);
    }
  };

  const profileFields = [
    { icon: User,       label: "Username",    value: user?.username ?? "—" },
    { icon: Mail,       label: "Email",       value: user?.email ?? "—" },
    { icon: User,       label: "Full Name",   value: user?.full_name ?? "—" },
    { icon: BadgeCheck, label: "Role",        value: user?.is_admin ? "Administrator" : "User" },
    { icon: Calendar,   label: "Member since",value: user?.created_at ? new Date(user.created_at).toLocaleDateString(undefined, { month: "long", day: "numeric", year: "numeric" }) : "—" },
  ];

  return (
    <div
      className="min-h-full px-8 py-8"
      style={{ background: "linear-gradient(180deg, #0a0f1e 0%, #030712 100%)" }}
    >
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -15 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-8"
      >
        <div className="flex items-center gap-2 text-xs text-gray-600 mb-1">
          <Settings className="h-3.5 w-3.5" />
          <span>Account Settings</span>
        </div>
        <h1 className="text-3xl font-bold text-white">Settings</h1>
        <p className="mt-1 text-sm text-gray-500">Manage your account and preferences.</p>
      </motion.div>

      <div className="mx-auto max-w-2xl space-y-6">
        {/* Profile card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, duration: 0.5 }}
          className="card-glow relative overflow-hidden"
        >
          <div
            className="absolute inset-x-0 top-0 h-px"
            style={{ background: "linear-gradient(90deg, transparent, rgba(99,102,241,0.6), transparent)" }}
          />

          <div className="mb-5 flex items-center gap-3">
            <div
              className="flex h-9 w-9 items-center justify-center rounded-xl"
              style={{ background: "rgba(99,102,241,0.15)", border: "1px solid rgba(99,102,241,0.25)" }}
            >
              <User className="h-4 w-4" style={{ color: "#6366f1" }} />
            </div>
            <h2 className="font-bold text-gray-200">Profile Information</h2>
          </div>

          {/* Avatar */}
          <div className="mb-6 flex items-center gap-4">
            <motion.div
              whileHover={{ scale: 1.05 }}
              className="relative flex h-16 w-16 items-center justify-center rounded-2xl text-2xl font-black text-white"
              style={{ background: "linear-gradient(135deg, #4f46e5, #7c3aed)" }}
            >
              {user?.username?.[0]?.toUpperCase() ?? "?"}
              <motion.div
                className="absolute -bottom-1 -right-1 flex h-5 w-5 items-center justify-center rounded-full"
                style={{ background: "#00ff88", border: "2px solid #0a0f1e" }}
                animate={{ scale: [1, 1.2, 1] }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <Shield className="h-2.5 w-2.5 text-black" />
              </motion.div>
            </motion.div>
            <div>
              <p className="font-semibold text-white">{user?.username}</p>
              <p className="text-sm text-gray-500">{user?.email}</p>
              <div
                className="mt-1 inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-xs"
                style={{ background: "rgba(0,255,136,0.1)", border: "1px solid rgba(0,255,136,0.2)", color: "#00ff88" }}
              >
                <BadgeCheck className="h-3 w-3" />
                {user?.is_admin ? "Administrator" : "Verified User"}
              </div>
            </div>
          </div>

          {/* Fields */}
          <dl className="space-y-3">
            {profileFields.map(({ icon: Icon, label, value }, i) => (
              <motion.div
                key={label}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.15 + i * 0.05 }}
                className="flex items-center justify-between rounded-xl px-4 py-3"
                style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.04)" }}
              >
                <div className="flex items-center gap-2.5">
                  <Icon className="h-4 w-4 text-gray-600" />
                  <dt className="text-sm text-gray-500">{label}</dt>
                </div>
                <dd className="text-sm font-medium text-gray-200">{value}</dd>
              </motion.div>
            ))}
          </dl>
        </motion.div>

        {/* Change password */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.5 }}
          className="card-glow relative overflow-hidden"
        >
          <div
            className="absolute inset-x-0 top-0 h-px"
            style={{ background: "linear-gradient(90deg, transparent, rgba(0,245,255,0.4), transparent)" }}
          />

          <div className="mb-5 flex items-center gap-3">
            <div
              className="flex h-9 w-9 items-center justify-center rounded-xl"
              style={{ background: "rgba(0,245,255,0.08)", border: "1px solid rgba(0,245,255,0.15)" }}
            >
              <Lock className="h-4 w-4" style={{ color: "#00f5ff" }} />
            </div>
            <h2 className="font-bold text-gray-200">Change Password</h2>
          </div>

          <form onSubmit={handlePasswordChange} className="space-y-4">
            {/* Current */}
            <div>
              <label className="mb-1.5 block text-xs font-medium text-gray-500">Current Password</label>
              <div className="relative">
                <input
                  type={showCurrent ? "text" : "password"}
                  className="input pr-10"
                  value={passwords.current_password}
                  onChange={(e) => setPasswords({ ...passwords, current_password: e.target.value })}
                  required
                  autoComplete="current-password"
                />
                <button type="button" onClick={() => setShowCurrent((v) => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-600 hover:text-gray-400 transition-colors"
                  aria-label="Toggle">
                  {showCurrent ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>

            {/* New */}
            <div>
              <label className="mb-1.5 block text-xs font-medium text-gray-500">New Password</label>
              <div className="relative">
                <input
                  type={showNew ? "text" : "password"}
                  className="input pr-10"
                  value={passwords.new_password}
                  onChange={(e) => setPasswords({ ...passwords, new_password: e.target.value })}
                  required
                  autoComplete="new-password"
                />
                <button type="button" onClick={() => setShowNew((v) => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-600 hover:text-gray-400 transition-colors"
                  aria-label="Toggle">
                  {showNew ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>

            {/* Confirm */}
            <div>
              <label className="mb-1.5 block text-xs font-medium text-gray-500">Confirm New Password</label>
              <input
                type="password"
                className="input"
                value={passwords.confirm}
                onChange={(e) => setPasswords({ ...passwords, confirm: e.target.value })}
                required
                autoComplete="new-password"
              />
            </div>

            <motion.button
              type="submit"
              disabled={isChanging}
              className="btn-primary"
              whileHover={{ scale: 1.02 }}
              whileTap={{ scale: 0.97 }}
            >
              {isChanging ? (
                <motion.span
                  className="h-4 w-4 rounded-full border-2 border-white/30 border-t-white"
                  animate={{ rotate: 360 }}
                  transition={{ duration: 0.7, repeat: Infinity, ease: "linear" }}
                />
              ) : (
                <Save className="h-4 w-4" />
              )}
              {isChanging ? "Updating…" : "Update Password"}
            </motion.button>
          </form>
        </motion.div>
      </div>
    </div>
  );
}
