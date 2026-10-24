/**
 * ModelStatusBanner
 * =================
 * Fetches GET /api/v1/health/models and displays a banner when models are not
 * ready.  Shown on pages that invoke detection (DetectPage, DashboardPage).
 *
 * States:
 *   all_ready    → no banner (models available, everything fine)
 *   unavailable  → orange banner — "No model checkpoints loaded"
 *   partial      → yellow info banner — "Some models unavailable"
 *   degraded     → red error banner — "Corrupt model checkpoint detected"
 *   loading      → no banner (don't distract during load)
 *   error        → no banner (health endpoint itself is unreachable — main health check will catch this)
 */

import { useQuery } from "@tanstack/react-query";
import { AlertTriangle, Info, XCircle } from "lucide-react";
import { apiClient } from "@/lib/api";

interface ModelEntry {
  status: "ready" | "unavailable" | "corrupt";
  path: string;
  message: string;
}

interface ModelHealthResponse {
  overall: "all_ready" | "unavailable" | "partial" | "degraded";
  env: string;
  inference_mode: string;
  models: {
    image: ModelEntry;
    video: ModelEntry;
    audio: ModelEntry;
  };
}

function fetchModelHealth(): Promise<ModelHealthResponse> {
  return apiClient.get<ModelHealthResponse>("/health/models").then((r) => r.data);
}

// How long between auto-refreshes of model status (ms).
const REFRESH_INTERVAL = 60_000;

export function ModelStatusBanner() {
  const { data, isLoading, isError } = useQuery<ModelHealthResponse>({
    queryKey: ["model-health"],
    queryFn: fetchModelHealth,
    refetchInterval: REFRESH_INTERVAL,
    retry: 1,
  });

  // Don't render anything while loading or when an error prevents reaching the endpoint
  if (isLoading || isError || !data) return null;
  // All models ready — nothing to tell the user
  if (data.overall === "all_ready") return null;

  const isDevMode = data.inference_mode === "development_random_weights";

  const configs = {
    unavailable: {
      icon: AlertTriangle,
      title: "Model checkpoints not loaded",
      description: isDevMode
        ? "Running in development mode with random weights. Predictions are meaningless — train the model first."
        : "No trained model checkpoints are available. Detection endpoints are disabled.",
      bg: "rgba(251, 146, 60, 0.08)",
      border: "rgba(251, 146, 60, 0.25)",
      color: "#fb923c",
    },
    partial: {
      icon: Info,
      title: "Some models unavailable",
      description: isDevMode
        ? "One or more models are running on random weights (dev mode). See model status below."
        : "One or more model checkpoints are missing. Some detection features are disabled.",
      bg: "rgba(99, 102, 241, 0.08)",
      border: "rgba(99, 102, 241, 0.25)",
      color: "#818cf8",
    },
    degraded: {
      icon: XCircle,
      title: "Corrupt model checkpoint detected",
      description:
        "One or more model checkpoint files cannot be loaded. Detection may be partially unavailable. Check the server logs.",
      bg: "rgba(239, 68, 68, 0.08)",
      border: "rgba(239, 68, 68, 0.25)",
      color: "#f87171",
    },
  } as const;

  const cfg = configs[data.overall as keyof typeof configs];
  if (!cfg) return null;

  const { icon: Icon, title, description, bg, border, color } = cfg;

  // Build a short per-model status line
  const modelLines = Object.entries(data.models)
    .filter(([, m]) => m.status !== "ready")
    .map(([name, m]) => `${name}: ${m.status}`)
    .join("  ·  ");

  return (
    <div
      className="mx-8 mt-4 flex gap-3 rounded-xl px-4 py-3 text-sm"
      style={{ background: bg, border: `1px solid ${border}` }}
      role="alert"
    >
      <Icon className="mt-0.5 h-4 w-4 shrink-0" style={{ color }} />
      <div className="min-w-0 flex-1">
        <p className="font-semibold" style={{ color }}>
          {title}
        </p>
        <p className="mt-0.5 text-xs" style={{ color: "#94a3b8" }}>
          {description}
        </p>
        {modelLines && (
          <p className="mt-1 font-mono text-xs" style={{ color: "#64748b" }}>
            {modelLines}
          </p>
        )}
        <p className="mt-1 text-xs" style={{ color: "#475569" }}>
          Train models with{" "}
          <code className="rounded bg-white/5 px-1">
            python backend/training/train_image.py
          </code>
          {" · "}
          <a href="/api/docs#/Health/model_readiness_api_v1_health_models_get"
             className="underline underline-offset-2"
             target="_blank" rel="noreferrer"
             style={{ color }}>
            View full status
          </a>
        </p>
      </div>
    </div>
  );
}
