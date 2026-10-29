/**
 * Frontend types for DeepGuard BLINK integration.
 *
 * Design principles:
 * - score fields use `number | null` where the backend may not provide a value.
 * - The `label` field uses 'UNKNOWN' for inconclusive verdicts (no faces found, etc).
 * - Older result records may be missing optional fields — all extensions are optional.
 */

// ── Media & Status ─────────────────────────────────────────────────────────

export type MediaType = 'image' | 'video';
export type AnalysisStatus =
  | 'idle'
  | 'uploading'
  | 'analyzing'
  | 'polling'    // async video: waiting for Celery job to finish
  | 'complete'
  | 'error';

// ── Per-frame result (video only) ──────────────────────────────────────────

/**
 * A single frame result entry from the video pipeline.
 * `score` fields are null when the frame could not be scored (no face detected
 * or face detector failed). They must never be treated as 0 or 0.5.
 */
export interface FrameResult {
  frameIndex:     number;
  timestamp:      number;
  /** null = frame was not scored (no face / detector error) */
  spatialScore:   number | null;
  /** null = frame was not scored */
  frequencyScore: number | null;
  /** null = frame was not scored */
  fusedScore:     number | null;
  label:          'REAL' | 'FAKE' | 'UNKNOWN';
  heatmapUrl?:    string;
  /** 'detected' | 'not_found' | 'failed' */
  faceStatus?:    string;
  /** How many sliding windows contributed to this frame's score */
  windowCount?:   number;
}

// ── Branch scores ──────────────────────────────────────────────────────────

export interface BranchScores {
  /** Spatial branch (texture/pixel artifacts) */
  spatial:   number;
  /** Frequency branch (FFT/DCT spectrum) */
  frequency: number;
  /**
   * Fused branch.
   * Backend produces attention scores; we map attention → fused here because
   * the backend doesn't expose a combined "fused" score directly.
   * For videos this is the average across scored frames.
   */
  fused:     number;
}

// ── Main analysis result ───────────────────────────────────────────────────

export interface AnalysisResult {
  id:            string;
  mediaType:     MediaType;
  /**
   * Overall verdict.
   * 'UNKNOWN' = inconclusive (no faces detected, model unavailable, etc.)
   * This is deliberately different from 'REAL' — absence of evidence is not
   * evidence of authenticity.
   */
  label:         'REAL' | 'FAKE' | 'UNKNOWN';
  confidence:    number;
  branchScores:  BranchScores;
  frameResults?: FrameResult[];
  heatmapUrl?:   string;
  processingTime: number;
  warnings:      string[];
  metadata: {
    filename:    string;
    fileSize:    number;
    dimensions?: { width: number; height: number };
    duration?:   number;
    framesAnalyzed?: number;
  };

  // ── Extended backend fields (not in original mock) ────────────────────
  /** null when verdict is unknown (no evidence gathered) */
  fakeProbability?:   number | null;
  /** null when verdict is unknown */
  realProbability?:   number | null;
  facesDetected?:     number;
  facesNotFound?:     number;
  facesFailed?:       number;
  scoredFramesCount?: number;
  suspiciousTimestamps?: number[];
  /** True when the backend is running with untrained random weights (dev mode) */
  randomWeightsWarning?: boolean;
}

// ── Zustand store ──────────────────────────────────────────────────────────

export interface AppState {
  status:       AnalysisStatus;
  uploadedFile: File | null;
  previewUrl:   string | null;
  result:       AnalysisResult | null;
  error:        string | null;
  /** Current progress stage label */
  progressStage: string;
  /** Progress percentage (0–100) */
  progressPct:  number;
  /** Celery job ID for async video jobs */
  pollingJobId: string | null;

  setFile:     (file: File) => void;
  setStatus:   (status: AnalysisStatus) => void;
  setResult:   (result: AnalysisResult) => void;
  setError:    (error: string) => void;
  setProgress: (stage: string, pct: number) => void;
  reset:       () => void;
}

// ── Backend API response shapes ────────────────────────────────────────────
// These match the actual FastAPI response schemas — used for type-safe deserialization.

/** Response from POST /api/v1/detect/image (synchronous, HTTP 200) */
export interface BackendImageResponse {
  job_id:                 string;
  verdict:                'real' | 'fake' | 'unknown';
  confidence:             number;
  fake_probability:       number;
  real_probability:       number;
  faces_detected:         number;
  branch_scores:          Record<string, number>;
  heatmap_url:            string | null;
  processing_ms:          number;
  random_weights_warning: boolean;
}

/** Response from POST /api/v1/detect/video (async, HTTP 202) */
export interface BackendVideoSubmitResponse {
  job_id:    string;
  status:    string;
  message:   string;
  poll_url:  string;
}

/** Per-frame score entry in a job result (score is nullable) */
export interface BackendFrameScore {
  frame:        number;
  score:        number | null;
  timestamp:    number;
  face_status?: 'detected' | 'not_found' | 'failed';
  window_count?: number;
}

/** Result sub-object on a completed DetectionJob */
export interface BackendDetectionResult {
  id:              string;
  verdict:         'real' | 'fake' | 'unknown';
  confidence:      number;
  branch_scores:   Record<string, number> | null;
  frame_scores:    BackendFrameScore[] | null;
  chunk_scores:    unknown[] | null;
  heatmap_url:     string | null;
  duration_sec:    number | null;
  frames_analyzed: number | null;
  faces_detected:  number | null;
  raw_output:      Record<string, unknown> | null;
  created_at:      string;
}

/** Response from GET /api/v1/jobs/{id} */
export interface BackendJobResponse {
  id:                string;
  original_filename: string;
  media_type:        'image' | 'video' | 'audio';
  file_size_bytes:   number;
  status:            'pending' | 'running' | 'completed' | 'failed';
  error_msg:         string | null;
  started_at:        string | null;
  ended_at:          string | null;
  created_at:        string;
  result:            BackendDetectionResult | null;
}
