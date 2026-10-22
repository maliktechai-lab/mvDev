// ── Auth Types ─────────────────────────────────────────────────────────────

export interface User {
  id: string;
  email: string;
  username: string;
  full_name: string | null;
  is_active: boolean;
  is_admin: boolean;
  created_at: string;
}

export interface TokenResponse {
  access_token: string;
  refresh_token: string;
  token_type: string;
  expires_in: number;
}

export interface LoginRequest {
  email: string;
  password: string;
}

export interface RegisterRequest {
  email: string;
  username: string;
  full_name?: string;
  password: string;
}

// ── Media Types ────────────────────────────────────────────────────────────

export type MediaType = "image" | "video" | "audio";
export type JobStatus = "pending" | "running" | "completed" | "failed";
export type Verdict = "real" | "fake" | "unknown";

// ── Detection Types ────────────────────────────────────────────────────────

export interface BranchScores {
  spatial?: number;
  frequency?: number;
  attention?: number;
  temporal?: number;
  biological?: number;
  raw_waveform?: number;
  spectral?: number;
  prosody?: number;
  [key: string]: number | undefined;
}

export interface FrameScore {
  frame: number;
  score: number;
  timestamp: number;
}

export interface ChunkScore {
  chunk: number;
  score: number;
  start: number;
  end: number;
}

export interface DetectionResult {
  id: string;
  verdict: Verdict;
  confidence: number;
  branch_scores: BranchScores | null;
  frame_scores: FrameScore[] | null;
  chunk_scores: ChunkScore[] | null;
  heatmap_url: string | null;
  duration_sec: number | null;
  frames_analyzed: number | null;
  faces_detected: number | null;
  created_at: string;
}

export interface DetectionJob {
  id: string;
  original_filename: string;
  media_type: MediaType;
  file_size_bytes: number;
  status: JobStatus;
  error_msg: string | null;
  started_at: string | null;
  ended_at: string | null;
  created_at: string;
  result: DetectionResult | null;
}

// ── API Response Types ─────────────────────────────────────────────────────

export interface APIError {
  detail: string;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}
