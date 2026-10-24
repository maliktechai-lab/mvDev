import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { Toaster } from "react-hot-toast";

import { ProtectedRoute }  from "@/components/common/ProtectedRoute";
import { LoginPage }       from "@/pages/LoginPage";
import { RegisterPage }    from "@/pages/RegisterPage";
import { DashboardPage }   from "@/pages/DashboardPage";
import { DetectPage }      from "@/pages/DetectPage";
import { HistoryPage }     from "@/pages/HistoryPage";
import { SettingsPage }    from "@/pages/SettingsPage";
import { NotFoundPage }    from "@/pages/NotFoundPage";

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
          {/* Public */}
          <Route path="/login"    element={<LoginPage />} />
          <Route path="/register" element={<RegisterPage />} />

          {/* Protected — wrapped in AppLayout */}
          <Route element={<ProtectedRoute />}>
            <Route path="/dashboard" element={<DashboardPage />} />
            <Route path="/detect"    element={<DetectPage />} />
            <Route path="/history"   element={<HistoryPage />} />
            <Route path="/settings"  element={<SettingsPage />} />
          </Route>

          {/* Default redirect */}
          <Route path="/"  element={<Navigate to="/dashboard" replace />} />
          {/* 404 */}
          <Route path="*"  element={<NotFoundPage />} />
        </Routes>
      </BrowserRouter>

      <Toaster
        position="top-right"
        toastOptions={{
          style: {
            background: "rgba(13, 20, 36, 0.95)",
            color: "#e2e8f0",
            border: "1px solid rgba(255,255,255,0.08)",
            backdropFilter: "blur(20px)",
            borderRadius: "12px",
          },
          success: { iconTheme: { primary: "#00ff88", secondary: "#030712" } },
          error:   { iconTheme: { primary: "#ff003c", secondary: "#030712" } },
        }}
      />
    </QueryClientProvider>
  );
}
