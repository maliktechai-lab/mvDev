import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "react-hot-toast";

import { ProtectedRoute } from "@/components/common/ProtectedRoute";
import { LoginPage }     from "@/pages/LoginPage";
import { RegisterPage }  from "@/pages/RegisterPage";
import { DashboardPage } from "@/pages/DashboardPage";
import { DetectPage }    from "@/pages/DetectPage";
import { HistoryPage }   from "@/pages/HistoryPage";
import { SettingsPage }  from "@/pages/SettingsPage";

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 30_000,
      retry: 1,
    },
  },
});

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <Routes>
          {/* Public routes */}
          <Route path="/login"    element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />

          {/* Protected routes — wrapped in AppLayout */}
          <Route element={<ProtectedRoute />}>
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/detect"    element={<DetectPage />} />
            <Route path="/history"   element={<HistoryPage />} />
            <Route path="/settings"  element={<SettingsPage />} />
          </Route>

          {/* Default redirect */}
          <Route path="/"  element={<Navigate to="/dashboard" replace />} />
          <Route path="*"  element={<Navigate to="/dashboard" replace />} />
        </Routes>
      </BrowserRouter>

      {/* Toast notifications */}
      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: "#1f2937",
            color: "#f9fafb",
            border: "1px solid #374151",
          },
          success: { iconTheme: { primary: "#22c55e", secondary: "#fff" } },
          error:   { iconTheme: { primary: "#ef4444", secondary: "#fff" } },
        }}
      />
    </QueryClientProvider>
  );
}
