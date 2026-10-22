import { useState, useEffect, useRef } from "react";
import { jobsApi } from "@/lib/api";
import type { DetectionJob } from "@/types";

interface UseJobPollerOptions {
  jobId: string | null;
  intervalMs?: number;
  onComplete?: (job: DetectionJob) => void;
  onError?: (job: DetectionJob) => void;
}

interface JobPollerState {
  job: DetectionJob | null;
  isPolling: boolean;
  error: string | null;
}

/**
 * Polls GET /api/v1/jobs/{id} at `intervalMs` until status is
 * "completed" or "failed". Calls callbacks accordingly.
 */
export function useJobPoller({
  jobId,
  intervalMs = 3000,
  onComplete,
  onError,
}: UseJobPollerOptions): JobPollerState {
  const [job, setJob]           = useState<DetectionJob | null>(null);
  const [isPolling, setPolling] = useState(false);
  const [error, setError]       = useState<string | null>(null);
  const intervalRef             = useRef<ReturnType<typeof setInterval> | null>(null);

  const stopPolling = () => {
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    setPolling(false);
  };

  useEffect(() => {
    if (!jobId) return;

    setPolling(true);
    setError(null);

    const poll = async () => {
      try {
        const res  = await jobsApi.get(jobId);
        const data = res.data as DetectionJob;
        setJob(data);

        if (data.status === "completed") {
          stopPolling();
          onComplete?.(data);
        } else if (data.status === "failed") {
          stopPolling();
          setError(data.error_msg || "Detection failed");
          onError?.(data);
        }
      } catch (err) {
        stopPolling();
        setError("Failed to fetch job status");
      }
    };

    // Poll immediately then on interval
    poll();
    intervalRef.current = setInterval(poll, intervalMs);

    return () => stopPolling();
  }, [jobId]);  // eslint-disable-line react-hooks/exhaustive-deps

  return { job, isPolling, error };
}
