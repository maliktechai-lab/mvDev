import { useState, FormEvent } from "react";
import { useAuthStore } from "@/store/authStore";
import { apiClient } from "@/lib/api";
import toast from "react-hot-toast";

export function SettingsPage() {
  const { user } = useAuthStore();
  const [passwords, setPasswords] = useState({
    current_password: "",
    new_password: "",
    confirm: "",
  });
  const [isChanging, setIsChanging] = useState(false);

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
        new_password: passwords.new_password,
      });
      toast.success("Password updated");
      setPasswords({ current_password: "", new_password: "", confirm: "" });
    } finally {
      setIsChanging(false);
    }
  };

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">Settings</h1>
        <p className="mt-1 text-sm text-gray-400">Manage your account preferences.</p>
      </div>

      <div className="mx-auto max-w-xl space-y-6">
        {/* Profile info */}
        <div className="card">
          <h2 className="mb-4 text-lg font-semibold text-white">Profile</h2>
          <dl className="space-y-3">
            <div className="flex justify-between">
              <dt className="text-sm text-gray-400">Username</dt>
              <dd className="text-sm font-medium text-gray-100">{user?.username}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-sm text-gray-400">Email</dt>
              <dd className="text-sm font-medium text-gray-100">{user?.email}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-sm text-gray-400">Full Name</dt>
              <dd className="text-sm font-medium text-gray-100">{user?.full_name ?? "—"}</dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-sm text-gray-400">Role</dt>
              <dd className="text-sm font-medium text-gray-100">
                {user?.is_admin ? "Administrator" : "User"}
              </dd>
            </div>
            <div className="flex justify-between">
              <dt className="text-sm text-gray-400">Member since</dt>
              <dd className="text-sm font-medium text-gray-100">
                {user?.created_at ? new Date(user.created_at).toLocaleDateString() : "—"}
              </dd>
            </div>
          </dl>
        </div>

        {/* Change password */}
        <div className="card">
          <h2 className="mb-4 text-lg font-semibold text-white">Change Password</h2>
          <form onSubmit={handlePasswordChange} className="space-y-4">
            <div>
              <label className="mb-1.5 block text-sm font-medium text-gray-300">Current Password</label>
              <input
                type="password"
                className="input"
                value={passwords.current_password}
                onChange={(e) => setPasswords({ ...passwords, current_password: e.target.value })}
                required
                autoComplete="current-password"
              />
            </div>
            <div>
              <label className="mb-1.5 block text-sm font-medium text-gray-300">New Password</label>
              <input
                type="password"
                className="input"
                value={passwords.new_password}
                onChange={(e) => setPasswords({ ...passwords, new_password: e.target.value })}
                required
                autoComplete="new-password"
              />
            </div>
            <div>
              <label className="mb-1.5 block text-sm font-medium text-gray-300">Confirm New Password</label>
              <input
                type="password"
                className="input"
                value={passwords.confirm}
                onChange={(e) => setPasswords({ ...passwords, confirm: e.target.value })}
                required
                autoComplete="new-password"
              />
            </div>
            <button type="submit" disabled={isChanging} className="btn-primary">
              {isChanging ? "Updating…" : "Update Password"}
            </button>
          </form>
        </div>
      </div>
    </div>
  );
}
