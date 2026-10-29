/**
 * realApi.ts
 * ==========
 * Replaces mockApi.ts. Calls the DeepGuard FastAPI backend for real detection.
 *
 * Endpoints used:
 *  POST /api/v1/auth/login         → get JWT
 *  POST /api/v1/auth/register      → create account
 *  POST /api/v1/detect/image       → sync image detection (200)
 *  POST /api/v1/detect/video       → async video detection (202 + job_id)
 *  GET  /api/v1/jobs/{id}          → poll job status / result
 *
 * Auth: JWT Bearer.  Tokens are stored in localStorage.
 * Base URL: VITE_API_URL env var (default: /api/v1, proxied by vite.config.ts).
 */

import axios, { type AxiosProgressEvent } from 'axios';
import type {
  AnalysisResult,
  MediaType,
  BackendImageResponse,
  BackendJobResponse,
  BackendVideoSubmitResponse,
} from '../types';

// ── Config ─────────────────────────────────────────────────────────────────

const BASE_URL = (import.meta.env.VITE_API_URL as string | undefined) ?? '/api/v1';

// Video polling: poll every 3 seconds, give up after 5 minutes.
const POLL_INTERVAL_MS = 3_000;
const POLL_TIMEOUT_MS  = 5 * 60 * 1_000;

// ── Axios instance ─────────────────────────────────────────────────────────

export const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 600_000, // 10 min for large uploads
});

// Attach stored JWT on every request
apiClient.interceptors.request.use((config) => {
  const token = localStorage.getItem('dg_access_token');
  if (token) {
    config.headers = config.headers ?? {};
    config.headers['Authorization'] = `Bearer ${token}`;
  }
  return config;
});

// ── Auth helpers ───────────────────────────────────────────────────────────

export interface AuthTokens {
  access_token: string;
  refresh_token: string;
  token_type: string;
}

export async function login(email: string, password: string): Promise<AuthTokens> {
  const res = await apiClient.post<AuthTokens>('/auth/login', { email, password });
  localStorage.setItem('dg_access_token', res.data.access_token);
  localStorage.setItem('dg_refresh_token', res.data.refresh_token);
  return res.data;
}

export async function register(
  email: string,
  username: string,
  password: string,
  full_name?: string,
): Promise<void> {
  await apiClient.post('/auth/register', { email, username, password, full_name });
}

export function logout(): void {
  localStorage.removeItem('dg_access_token');
  localStorage.removeItem('dg_refresh_token');
}

export function isLoggedIn(): boolean {
  return !!localStorage.getItem('dg_access_token');
}

export function getStoredToken(): string | null {
  return localStorage.getItem('dg_access_token');
}

// ── Response → AnalysisResult mapping ─────────────────────────────────────

/**
 * Maps the backend's branch_scores object to the frontend's BranchScores shape.
 *
 * Backend  → Frontend
 * spatial  → spatial
 * frequency → frequency
 * attention → fused  (closest semantic equivalent; no "fused" branch in backend)
 *
 * All values are clamped to [0,1].
 */
export function mapBranchScores(raw: Record<string, number> | null | undefined) {
  const r = raw ?? {};
  const clamp = (v: number | undefined) =>
    typeof v === 'number' ? Math.max(0, Math.min(1, v)) : 0;
  return {
    spatial:   clamp(r['spatial']),
    frequency: clamp(r['frequency']),
    // Use attention if present, fall back to mean(spatial,frequency)
    fused:
      typeof r['attention'] === 'number'
        ? clamp(r['attention'])
        : clamp((clamp(r['spatial']) + clamp(r['frequency'])) / 2),
  };
}

export function mapVerdict(verdict: string): 'REAL' | 'FAKE' | 'UNKNOWN' {
  if (verdict === 'fake') return 'FAKE';
  if (verdict === 'real') return 'REAL';
  return 'UNKNOWN';
}

/**
 * Map backend image response → AnalysisResult.
 * Never invents scores for unavailable fields.
 */
export function mapImageResponse(
  response: BackendImageResponse,
  file: File,
): AnalysisResult {
  const branchScores = mapBranchScores(
    response.branch_scores as Record<string, number> | null | undefined,
  );
  const warnings: string[] = [
    'Results are model-based estimates, not forensic proof.',
  ];
  if (response.random_weights_warning) {
    warnings.unshift(
      'Development mode: model weights are random — results are meaningless.',
    );
  }
  if (response.verdict === 'unknown') {
    warnings.unshift('Inconclusive: no face was detected in this image.');
  }

  return {
    id:           response.job_id,
    mediaType:    'image',
    label:        mapVerdict(response.verdict),
    confidence:   response.confidence,
    branchScores,
    heatmapUrl:   response.heatmap_url ?? undefined,
    processingTime: response.processing_ms,
    warnings,
    metadata: {
      filename:  file.name,
      fileSize:  file.size,
    },
    // Extra backend fields surfaced for display
    fakeProbability: response.fake_probability,
    realProbability: response.real_probability,
    facesDetected:   response.faces_detected,
    randomWeightsWarning: response.random_weights_warning ?? false,
  };
}

/**
 * Map a completed video job → AnalysisResult.
 * frame_scores may contain null entries (frames where no face was detected).
 */
export function mapVideoJobResponse(
  job: BackendJobResponse,
  file: File,
): AnalysisResult {
  const r = job.result!;
  const raw = r.raw_output as Record<string, unknown> | null | undefined;

  const branchScores = mapBranchScores(
    r.branch_scores as Record<string, number> | null | undefined,
  );

  // Build frameResults from the backend's frame_scores array.
  // Each entry may have score=null (unscored frame).
  const frameResults = (r.frame_scores ?? []).map((fs) => ({
    frameIndex:     fs.frame,
    timestamp:      fs.timestamp,
    // For an unscored frame score is null — represent as null, not 0
    spatialScore:   fs.score,   // null | number
    frequencyScore: fs.score,   // null | number (same signal — backend doesn't split per-frame)
    fusedScore:     fs.score,   // null | number
    label: (fs.score !== null
      ? (fs.score >= 0.5 ? 'FAKE' : 'REAL')
      : 'UNKNOWN') as 'REAL' | 'FAKE' | 'UNKNOWN',
    faceStatus: fs.face_status ?? 'detected',
    windowCount: fs.window_count ?? 0,
  }));

  const warnings: string[] = [
    'Results are model-based estimates, not forensic proof.',
    'Frame sampling may miss brief manipulations.',
  ];
  const randomWeightsWarning = !!(raw?.['random_weights_warning']);
  if (randomWeightsWarning) {
    warnings.unshift('Development mode: model weights are random — results are meaningless.');
  }
  if (r.verdict === 'unknown') {
    warnings.unshift('Inconclusive: no frames could be scored (no faces detected).');
  }

  const fakeProbability = (raw?.['fake_probability'] as number | null | undefined) ?? null;
  const realProbability = (raw?.['real_probability'] as number | null | undefined) ?? null;
  const suspiciousTimestamps = (raw?.['suspicious_timestamps'] as number[] | null | undefined) ?? [];

  return {
    id:            job.id,
    mediaType:     'video',
    label:         mapVerdict(r.verdict),
    confidence:    r.confidence,
    branchScores,
    frameResults,
    heatmapUrl:    r.heatmap_url ?? undefined,
    processingTime: job.ended_at && job.started_at
      ? new Date(job.ended_at).getTime() - new Date(job.started_at).getTime()
      : 0,
    warnings,
    metadata: {
      filename:        file.name,
      fileSize:        file.size,
      duration:        r.duration_sec ?? undefined,
      framesAnalyzed:  r.frames_analyzed ?? undefined,
    },
    // Extra backend fields
    fakeProbability,
    realProbability,
    facesDetected:   r.faces_detected ?? 0,
    facesNotFound:   (raw?.['faces_not_found'] as number | undefined) ?? 0,
    facesFailed:     (raw?.['faces_failed'] as number | undefined) ?? 0,
    scoredFramesCount: (raw?.['scored_frames_count'] as number | undefined) ?? 0,
    suspiciousTimestamps,
    randomWeightsWarning,
  };
}

// ── Core analyzeMedia function (drop-in replacement for mockApi) ────────────

export type ProgressCallback = (stage: string, progress: number) => void;

/**
 * Analyze an image or video file using the DeepGuard backend.
 *
 * Stages reported to the progress callback:
 *   Image: Uploading… → Analyzing… → Complete
 *   Video: Uploading… → Processing in background… → Checking status… → Complete
 *
 * Throws a typed ApiError on backend errors.
 * Throws AnalysisTimeoutError if video polling exceeds POLL_TIMEOUT_MS.
 */
export async function analyzeMedia(
  file: File,
  onProgress?: ProgressCallback,
): Promise<AnalysisResult> {
  const mediaType: MediaType = file.type.startsWith('video/') ? 'video' : 'image';

  if (mediaType === 'image') {
    return analyzeImage(file, onProgress);
  } else {
    return analyzeVideo(file, onProgress);
  }
}

// ── Image ──────────────────────────────────────────────────────────────────

async function analyzeImage(
  file: File,
  onProgress?: ProgressCallback,
): Promise<AnalysisResult> {
  onProgress?.('Uploading media...', 5);

  const formData = new FormData();
  formData.append('file', file);

  let uploadDone = false;

  const res = await apiClient.post<BackendImageResponse>(
    '/detect/image?generate_heatmap=true',
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e: AxiosProgressEvent) => {
        if (e.total && !uploadDone) {
          const pct = Math.round((e.loaded / e.total) * 40); // 0-40%
          onProgress?.('Uploading media...', pct);
        }
      },
    },
  );
  uploadDone = true;

  // Backend runs sync inference; once the POST returns we have the result
  onProgress?.('Running spatial branch (EfficientNet-B4)...', 55);
  onProgress?.('Running frequency branch (FFT)...', 70);
  onProgress?.('Fusing branch predictions...', 85);
  onProgress?.('Generating Grad-CAM++ heatmap...', 95);
  onProgress?.('Complete', 100);

  return mapImageResponse(res.data, file);
}

// ── Video ──────────────────────────────────────────────────────────────────

export class AnalysisTimeoutError extends Error {
  constructor(jobId: string) {
    super(`Video analysis timed out after ${POLL_TIMEOUT_MS / 1000}s (job: ${jobId})`);
    this.name = 'AnalysisTimeoutError';
  }
}

async function analyzeVideo(
  file: File,
  onProgress?: ProgressCallback,
): Promise<AnalysisResult> {
  onProgress?.('Uploading media...', 5);

  const formData = new FormData();
  formData.append('file', file);

  const submitRes = await apiClient.post<BackendVideoSubmitResponse>(
    '/detect/video',
    formData,
    {
      headers: { 'Content-Type': 'multipart/form-data' },
      onUploadProgress: (e: AxiosProgressEvent) => {
        if (e.total) {
          const pct = Math.round((e.loaded / e.total) * 40);
          onProgress?.('Uploading media...', pct);
        }
      },
    },
  );

  const { job_id } = submitRes.data;
  onProgress?.('Processing in background...', 45);

  // ── Poll until complete ────────────────────────────────────────────────
  const deadline = Date.now() + POLL_TIMEOUT_MS;
  let pollCount  = 0;

  while (Date.now() < deadline) {
    await sleep(POLL_INTERVAL_MS);
    pollCount++;

    // Simulate progress creep during polling (45% → 90%)
    const pct = Math.min(90, 45 + pollCount * 3);
    onProgress?.('Checking status...', pct);

    const jobRes = await apiClient.get<BackendJobResponse>(`/jobs/${job_id}`);
    const job    = jobRes.data;

    if (job.status === 'completed' && job.result) {
      onProgress?.('Complete', 100);
      return mapVideoJobResponse(job, file);
    }

    if (job.status === 'failed') {
      throw new Error(job.error_msg ?? 'Video detection failed on the server.');
    }

    // pending or running — keep polling
    if (job.status === 'running') {
      onProgress?.('Running inference on frames...', pct);
    }
  }

  throw new AnalysisTimeoutError(job_id);
}

// ── Utilities ──────────────────────────────────────────────────────────────

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Extract a user-readable error message from an axios error or plain Error.
 */
export function getErrorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    const detail = (err.response?.data as { detail?: string } | undefined)?.detail;
    if (detail) return detail;
    if (err.response?.status === 503) {
      return 'Model not available — no trained checkpoint loaded. Run in development mode or train the model first.';
    }
    if (err.response?.status === 401) {
      return 'Authentication required. Please log in.';
    }
    if (err.response?.status === 413) {
      return 'File is too large for upload.';
    }
    if (err.response?.status === 415) {
      return 'Unsupported file type.';
    }
    if (err.message) return err.message;
  }
  if (err instanceof Error) return err.message;
  return 'An unexpected error occurred.';
}
