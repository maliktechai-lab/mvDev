import { useState, FormEvent } from "react";
import { Link, useNavigate } from "react-router-dom";
import { Shield } from "lucide-react";
import { useAuthStore } from "@/store/authStore";
import toast from "react-hot-toast";

export function RegisterPage() {
  const navigate = useNavigate();
  const { register, isLoading } = useAuthStore();
  const [form, setForm] = useState({
    email: "",
    username: "",
    full_name: "",
    password: "",
    confirmPassword: "",
  });
  const [errors, setErrors] = useState<Record<string, string>>({});

  const validate = () => {
    const e: Record<string, string> = {};
    if (form.password !== form.confirmPassword) e.confirmPassword = "Passwords do not match";
    if (form.password.length < 8) e.password = "At least 8 characters";
    if (!/[A-Z]/.test(form.password)) e.password = "Must include uppercase letter";
    if (!/[0-9]/.test(form.password)) e.password = "Must include a number";
    setErrors(e);
    return Object.keys(e).length === 0;
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    if (!validate()) return;
    try {
      await register({
        email: form.email,
        username: form.username,
        full_name: form.full_name || undefined,
        password: form.password,
      });
      toast.success("Account created! Please sign in.");
      navigate("/login");
    } catch {
      // error toast handled by interceptor
    }
  };

  return (
    <div className="flex min-h-screen items-center justify-center bg-gray-950 px-4 py-10">
      <div className="w-full max-w-md">
        {/* Logo */}
        <div className="mb-8 flex flex-col items-center gap-3">
          <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-brand-600 shadow-lg shadow-brand-600/30">
            <Shield className="h-7 w-7 text-white" />
          </div>
          <h1 className="text-2xl font-bold text-white">Create your account</h1>
        </div>

        <div className="card">
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="mb-1.5 block text-sm font-medium text-gray-300">Username *</label>
                <input
                  type="text"
                  className="input"
                  placeholder="johndoe"
                  value={form.username}
                  onChange={(e) => setForm({ ...form, username: e.target.value })}
                  required
                />
              </div>
              <div>
                <label className="mb-1.5 block text-sm font-medium text-gray-300">Full Name</label>
                <input
                  type="text"
                  className="input"
                  placeholder="John Doe"
                  value={form.full_name}
                  onChange={(e) => setForm({ ...form, full_name: e.target.value })}
                />
              </div>
            </div>

            <div>
              <label className="mb-1.5 block text-sm font-medium text-gray-300">Email *</label>
              <input
                type="email"
                className="input"
                placeholder="you@example.com"
                value={form.email}
                onChange={(e) => setForm({ ...form, email: e.target.value })}
                required
              />
            </div>

            <div>
              <label className="mb-1.5 block text-sm font-medium text-gray-300">Password *</label>
              <input
                type="password"
                className="input"
                placeholder="Min 8 chars, 1 uppercase, 1 number"
                value={form.password}
                onChange={(e) => setForm({ ...form, password: e.target.value })}
                required
              />
              {errors.password && <p className="mt-1 text-xs text-red-400">{errors.password}</p>}
            </div>

            <div>
              <label className="mb-1.5 block text-sm font-medium text-gray-300">Confirm Password *</label>
              <input
                type="password"
                className="input"
                placeholder="••••••••"
                value={form.confirmPassword}
                onChange={(e) => setForm({ ...form, confirmPassword: e.target.value })}
                required
              />
              {errors.confirmPassword && <p className="mt-1 text-xs text-red-400">{errors.confirmPassword}</p>}
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className="btn-primary w-full justify-center py-2.5"
            >
              {isLoading ? "Creating account…" : "Create account"}
            </button>
          </form>

          <p className="mt-5 text-center text-sm text-gray-500">
            Already have an account?{" "}
            <Link to="/login" className="font-medium text-brand-400 hover:text-brand-300">
              Sign in
            </Link>
          </p>
        </div>
      </div>
    </div>
  );
}
