import { Outlet } from "react-router-dom";
import { AppLayout } from "@/components/layout/AppLayout";

// Auth temporarily bypassed — all routes are accessible without login.
// Re-enable by wrapping with isAuthenticated check when auth is added back.
export function ProtectedRoute() {
  return (
    <AppLayout>
      <Outlet />
    </AppLayout>
  );
}
