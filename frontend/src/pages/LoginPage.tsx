import { useState, FormEvent } from "react";
import { Link, useNavigate, useLocation } from "react-router-dom";
import { motion } from "framer-motion";
import { Shield, Eye, EyeOff, ArrowRight, Lock, Mail } from "lucide-react";
import { useAuthStore } from "@/store/authStore";
import { ParticleField } from "@/components/3d/ParticleField";
import toast from "react-hot-toast";

export function LoginPage() {
  const navigate  = useNavigate();
  const location  = useLocation();
  const { login, isLoading } = useAuthStore();
  const [form, setForm]     = useState({ email: "", password: "" });
  const [showPw, setShowPw] = useState(false);
  const [focused, setFocused] = useState<string | null>(null);

  const from = (location.state as { from?: { pathname?: string } })?.from?.pathname ?? "/dashboard";

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    try {
      await login(form.email, form.password);
      toast.success("Welcome back!");
      navigate(from, { replace: true });
    } catch {
      // handled by axios interceptor
    }
  };

  return (
    <div
      className="relative flex min-h-screen items-center justify-center overflow-hidden px-4"
      style={{ background: "#030712" }}
    >
      {/* 3D particle background */}
      <ParticleField />

      {/* Grid overlay */}
      <div className="pointer-events-none absolute inset-0 bg-grid opacity-40" />

      {/* Radial glow center */}
      <div
        className="pointer-events-none absolute inset-0"
        style={{
          background: "radial-gradient(ellipse 60% 60% at 50% 50%, rgba(99,102,241,0.12) 0%, transparent 70%)",
        }}
      />

      {/* Card */}
      <motion.div
        initial={{ opacity: 0, y: 40, scale: 0.95 }}
        animate={{ opacity: 1, y: 0, scale: 1 }}
        transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
        className="relative z-10 w-full max-w-md"
      >
        {/* Logo */}
        <motion.div
          initial={{ opacity: 0, y: -20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1, duration: 0.5 }}
          className="mb-8 flex flex-col items-center gap-4"
        >
          <motion.div
            animate={{
              boxShadow: [
                "0 0 20px rgba(99,102,241,0.4)",
                "0 0 50px rgba(99,102,241,0.7)",
                "0 0 20px rgba(99,102,241,0.4)",
              ],
            }}
            transition={{ duration: 3, repeat: Infinity }}
            className="flex h-16 w-16 items-center justify-center rounded-2xl"
            style={{ background: "linear-gradient(135deg, #4f46e5, #7c3aed)" }}
          >
            <Shield className="h-8 w-8 text-white" />
          </motion.div>
          <div className="text-center">
            <h1 className="text-3xl font-bold text-white">DeepGuard</h1>
            <p className="mt-1 text-sm" style={{ color: "#64748b" }}>
              Synthetic Media Detection Platform
            </p>
          </div>
        </motion.div>

        {/* Form card */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.5 }}
          className="card-glow overflow-hidden"
        >
          {/* Card top glow strip */}
          <div
            className="absolute inset-x-0 top-0 h-px"
            style={{ background: "linear-gradient(90deg, transparent, rgba(99,102,241,0.6), transparent)" }}
          />

          <h2 className="mb-6 text-xl font-semibold text-white">Sign in to your account</h2>

          <form onSubmit={handleSubmit} className="space-y-5">
            {/* Email */}
            <div>
              <label className="mb-2 block text-sm font-medium" style={{ color: "#94a3b8" }}>
                Email address
              </label>
              <div className="relative">
                <Mail
                  className="absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 transition-colors"
                  style={{ color: focused === "email" ? "#6366f1" : "#475569" }}
                />
                <input
                  type="email"
                  className="input pl-10"
                  placeholder="you@example.com"
                  value={form.email}
                  onChange={(e) => setForm({ ...form, email: e.target.value })}
                  onFocus={() => setFocused("email")}
                  onBlur={() => setFocused(null)}
                  required
                  autoComplete="email"
                />
              </div>
            </div>

            {/* Password */}
            <div>
              <label className="mb-2 block text-sm font-medium" style={{ color: "#94a3b8" }}>
                Password
              </label>
              <div className="relative">
                <Lock
                  className="absolute left-3.5 top-1/2 h-4 w-4 -translate-y-1/2 transition-colors"
                  style={{ color: focused === "password" ? "#6366f1" : "#475569" }}
                />
                <input
                  type={showPw ? "text" : "password"}
                  className="input pl-10 pr-10"
                  placeholder="••••••••"
                  value={form.password}
                  onChange={(e) => setForm({ ...form, password: e.target.value })}
                  onFocus={() => setFocused("password")}
                  onBlur={() => setFocused(null)}
                  required
                  autoComplete="current-password"
                />
                <button
                  type="button"
                  onClick={() => setShowPw((v) => !v)}
                  className="absolute right-3.5 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300 transition-colors"
                  aria-label={showPw ? "Hide password" : "Show password"}
                >
                  {showPw ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                </button>
              </div>
            </div>

            {/* Submit */}
            <motion.button
              type="submit"
              disabled={isLoading}
              className="btn-primary w-full justify-center py-3 text-base"
              whileHover={{ scale: 1.01 }}
              whileTap={{ scale: 0.98 }}
            >
              {isLoading ? (
                <span className="flex items-center gap-2">
                  <motion.span
                    className="h-4 w-4 rounded-full border-2 border-white/30 border-t-white"
                    animate={{ rotate: 360 }}
                    transition={{ duration: 0.7, repeat: Infinity, ease: "linear" }}
                  />
                  Authenticating…
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  Sign In
                  <ArrowRight className="h-4 w-4" />
                </span>
              )}
            </motion.button>
          </form>

          <p className="mt-6 text-center text-sm" style={{ color: "#475569" }}>
            Don't have an account?{" "}
            <Link
              to="/register"
              className="font-semibold transition-colors"
              style={{ color: "#818cf8" }}
            >
              Create one
            </Link>
          </p>
        </motion.div>

        {/* Footer badges */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.5 }}
          className="mt-6 flex items-center justify-center gap-4"
        >
          {["JWT Auth", "E2E Encrypted", "Multi-Model AI"].map((label) => (
            <span
              key={label}
              className="rounded-full px-3 py-1 text-xs"
              style={{
                background: "rgba(255,255,255,0.04)",
                border: "1px solid rgba(255,255,255,0.07)",
                color: "#475569",
              }}
            >
              {label}
            </span>
          ))}
        </motion.div>
      </motion.div>
    </div>
  );
}
