import { useState, FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { motion, AnimatePresence } from "framer-motion";
import { Shield, Eye, EyeOff, ArrowRight, User, Mail, Lock, CheckCircle2, XCircle } from "lucide-react";
import { useAuthStore } from "@/store/authStore";
import { ParticleField } from "@/components/3d/ParticleField";
import toast from "react-hot-toast";

interface PasswordStrength {
  score: number;     // 0–4
  label: string;
  color: string;
}

function getPasswordStrength(pw: string): PasswordStrength {
  let score = 0;
  if (pw.length >= 8)          score++;
  if (/[A-Z]/.test(pw))        score++;
  if (/[0-9]/.test(pw))        score++;
  if (/[^a-zA-Z0-9]/.test(pw)) score++;
  const levels: PasswordStrength[] = [
    { score: 0, label: "Too short",  color: "#475569" },
    { score: 1, label: "Weak",       color: "#ef4444" },
    { score: 2, label: "Fair",       color: "#f97316" },
    { score: 3, label: "Good",       color: "#eab308" },
    { score: 4, label: "Strong",     color: "#22c55e" },
  ];
  return levels[score];
}

function Req({ met, text }: { met: boolean; text: string }) {
  return (
    <div className="flex items-center gap-1.5 text-xs">
      {met
        ? <CheckCircle2 className="h-3.5 w-3.5 text-green-400" />
        : <XCircle className="h-3.5 w-3.5 text-gray-600" />}
      <span style={{ color: met ? "#86efac" : "#475569" }}>{text}</span>
    </div>
  );
}

export function RegisterPage() {
  const navigate  = useNavigate();
  const { register, isLoading } = useAuthStore();
  const [form, setForm] = useState({
    email: "", username: "", full_name: "", password: "", confirmPassword: "",
  });
  const [showPw, setShowPw]     = useState(false);
  const [focused, setFocused]   = useState<string | null>(null);
  const [errors, setErrors]     = useState<Record<string, string>>({});

  const strength = getPasswordStrength(form.password);

  const validate = () => {
    const e: Record<string, string> = {};
    if (form.password.length < 8)           e.password = "At least 8 characters";
    if (!/[A-Z]/.test(form.password))       e.password = "Must include uppercase";
    if (!/[0-9]/.test(form.password))       e.password = "Must include a number";
    if (form.password !== form.confirmPassword) e.confirmPassword = "Passwords do not match";
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!validate()) return;
    try {
      await register({
        email:     form.email,
        username:  form.username,
        full_name: form.full_name || undefined,
        password:  form.password,
      });
      toast.success("Account created! Sign in to continue.");
      navigate("/login");
    } catch {
      // handled by interceptor
    }
  };

  const inputProps = (field: string) => ({
    onFocus: () => setFocused(field),
    onBlur:  () => setFocused(null),
  });

  const iconColor = (field: string) => ({ color: focused === field ? "#6366f1" : "#475569" });

  return (
    <div
      className="relative flex min-h-screen items-center justify-center overflow-hidden px-4 py-10"
      style={{ background: "#030712" }}
    >
      <ParticleField />
      <div className="pointer-events-none absolute inset-0 bg-grid opacity-40" />
      <div
        className="pointer-events-none absolute inset-0"
        style={{ background: "radial-gradient(ellipse 60% 60% at 50% 40%, rgba(99,102,241,0.1) 0%, transparent 70%)" }}
      />

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
          className="mb-8 flex flex-col items-center gap-3"
        >
          <motion.div
            animate={{ boxShadow: ["0 0 20px rgba(99,102,241,0.4)", "0 0 50px rgba(99,102,241,0.7)", "0 0 20px rgba(99,102,241,0.4)"] }}
            transition={{ duration: 3, repeat: Infinity }}
            className="flex h-14 w-14 items-center justify-center rounded-2xl"
            style={{ background: "linear-gradient(135deg, #4f46e5, #7c3aed)" }}
          >
            <Shield className="h-7 w-7 text-white" />
          </motion.div>
          <h1 className="text-2xl font-bold text-white">Create your account</h1>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2, duration: 0.5 }}
          className="card-glow overflow-hidden"
        >
          <div
            className="absolute inset-x-0 top-0 h-px"
            style={{ background: "linear-gradient(90deg, transparent, rgba(99,102,241,0.6), transparent)" }}
          />

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Name row */}
            <div className="grid grid-cols-2 gap-3">
              <div>
                <label className="mb-1.5 block text-xs font-medium" style={{ color: "#94a3b8" }}>Username *</label>
                <div className="relative">
                  <User className="absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 transition-colors" style={iconColor("username")} />
                  <input
                    type="text"
                    className="input pl-9 text-sm"
                    placeholder="johndoe"
                    value={form.username}
                    onChange={(e) => setForm({ ...form, username: e.target.value })}
                    {...inputProps("username")}
                    required
                  />
                </div>
              </div>
              <div>
                <label className="mb-1.5 block text-xs font-medium" style={{ color: "#94a3b8" }}>Full Name</label>
                <input
                  type="text"
                  className="input text-sm"
                  placeholder="John Doe"
                  value={form.full_name}
                  onChange={(e) => setForm({ ...form, full_name: e.target.value })}
                  {...inputProps("full_name")}
                />
              </div>
            </div>

            {/* Email */}
            <div>
              <label className="mb-1.5 block text-xs font-medium" style={{ color: "#94a3b8" }}>Email *</label>
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 transition-colors" style={iconColor("email")} />
                <input
                  type="email"
                  className="input pl-9 text-sm"
                  placeholder="you@example.com"
                  value={form.email}
                  onChange={(e) => setForm({ ...form, email: e.target.value })}
                  {...inputProps("email")}
                  required
                />
              </div>
            </div>

            {/* Password */}
            <div>
              <label className="mb-1.5 block text-xs font-medium" style={{ color: "#94a3b8" }}>Password *</label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 transition-colors" style={iconColor("password")} />
                <input
                  type={showPw ? "text" : "password"}
                  className="input pl-9 pr-9 text-sm"
                  placeholder="Min 8 chars, uppercase, number"
                  value={form.password}
                  onChange={(e) => setForm({ ...form, password: e.target.value })}
                  {...inputProps("password")}
                  required
                />
                <button type="button" onClick={() => setShowPw((v) => !v)}
                  className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-500 hover:text-gray-300 transition-colors"
                  aria-label="Toggle password visibility">
                  {showPw ? <EyeOff className="h-3.5 w-3.5" /> : <Eye className="h-3.5 w-3.5" />}
                </button>
              </div>

              {/* Strength meter */}
              <AnimatePresence>
                {form.password.length > 0 && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: "auto" }}
                    exit={{ opacity: 0, height: 0 }}
                    className="mt-2"
                  >
                    <div className="mb-1.5 flex gap-1">
                      {[1, 2, 3, 4].map((i) => (
                        <motion.div
                          key={i}
                          className="h-1 flex-1 rounded-full"
                          animate={{ background: i <= strength.score ? strength.color : "rgba(255,255,255,0.08)" }}
                          transition={{ duration: 0.3 }}
                        />
                      ))}
                    </div>
                    <div className="grid grid-cols-2 gap-1">
                      <Req met={form.password.length >= 8}          text="8+ characters" />
                      <Req met={/[A-Z]/.test(form.password)}        text="Uppercase letter" />
                      <Req met={/[0-9]/.test(form.password)}        text="Number" />
                      <Req met={/[^a-zA-Z0-9]/.test(form.password)} text="Special character" />
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
              {errors.password && <p className="mt-1.5 text-xs text-red-400">{errors.password}</p>}
            </div>

            {/* Confirm password */}
            <div>
              <label className="mb-1.5 block text-xs font-medium" style={{ color: "#94a3b8" }}>Confirm Password *</label>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 h-3.5 w-3.5 -translate-y-1/2 transition-colors" style={iconColor("confirm")} />
                <input
                  type="password"
                  className="input pl-9 text-sm"
                  placeholder="••••••••"
                  value={form.confirmPassword}
                  onChange={(e) => setForm({ ...form, confirmPassword: e.target.value })}
                  {...inputProps("confirm")}
                  required
                />
                {form.confirmPassword.length > 0 && (
                  <div className="absolute right-3 top-1/2 -translate-y-1/2">
                    {form.password === form.confirmPassword
                      ? <CheckCircle2 className="h-4 w-4 text-green-400" />
                      : <XCircle className="h-4 w-4 text-red-400" />}
                  </div>
                )}
              </div>
              {errors.confirmPassword && <p className="mt-1.5 text-xs text-red-400">{errors.confirmPassword}</p>}
            </div>

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
                  Creating account…
                </span>
              ) : (
                <span className="flex items-center gap-2">
                  Create Account
                  <ArrowRight className="h-4 w-4" />
                </span>
              )}
            </motion.button>
          </form>

          <p className="mt-5 text-center text-sm" style={{ color: "#475569" }}>
            Already have an account?{" "}
            <Link to="/login" className="font-semibold" style={{ color: "#818cf8" }}>
              Sign in
            </Link>
          </p>
        </motion.div>
      </motion.div>
    </div>
  );
}
