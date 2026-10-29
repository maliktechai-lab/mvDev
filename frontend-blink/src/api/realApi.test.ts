/**
 * Tests for the API response mapping logic in realApi.ts.
 *
 * Tests the pure mapping functions directly (no HTTP mocking) to verify
 * that backend responses are correctly mapped to the frontend's
 * AnalysisResult type, with special attention to:
 * - Null frame scores (unscored frames)
 * - UNKNOWN verdicts
 * - Random weights warning propagation
 * - Older results with missing optional fields
 * - Error message extraction
 */

import { describe, it, expect } from 'vitest';
import {
  getErrorMessage,
  AnalysisTimeoutError,
  mapBranchScores,
  mapVerdict,
  mapImageResponse,
  mapVideoJobResponse,
} from '../api/realApi';
import type {
  BackendImageResponse,
  BackendJobResponse,
} from '../types';

// ── Helper: mock file ──────────────────────────────────────────────────────

function mockFile(name = 'test.jpg', size = 1024, type = 'image/jpeg'): File {
  const blob = new Blob([new Uint8Array(size)], { type });
  return new File([blob], name, { type });
}

// ── getErrorMessage ────────────────────────────────────────────────────────

describe('getErrorMessage', () => {
  it('extracts detail from axios error', () => {
    const err = {
      isAxiosError: true,
      response: { status: 400, data: { detail: 'File too small' } },
      message: 'Request failed',
    };
    expect(getErrorMessage(err)).toBe('File too small');
  });

  it('returns 503 message for unavailable model', () => {
    const err = {
      isAxiosError: true,
      response: { status: 503, data: {} },
      message: 'Service Unavailable',
    };
    const msg = getErrorMessage(err);
    expect(msg).toContain('Model not available');
  });

  it('returns 401 auth message', () => {
    const err = {
      isAxiosError: true,
      response: { status: 401, data: {} },
      message: 'Unauthorized',
    };
    expect(getErrorMessage(err)).toContain('Authentication required');
  });

  it('returns 413 file too large message', () => {
    const err = {
      isAxiosError: true,
      response: { status: 413, data: {} },
      message: 'Payload Too Large',
    };
    expect(getErrorMessage(err)).toContain('too large');
  });

  it('returns 415 unsupported type message', () => {
    const err = {
      isAxiosError: true,
      response: { status: 415, data: {} },
      message: 'Unsupported Media Type',
    };
    expect(getErrorMessage(err)).toContain('Unsupported file type');
  });

  it('falls back to error.message', () => {
    const err = new Error('Something went wrong');
    expect(getErrorMessage(err)).toBe('Something went wrong');
  });

  it('handles unknown error type', () => {
    expect(getErrorMessage({})).toBe('An unexpected error occurred.');
  });
});

// ── AnalysisTimeoutError ────────────────────────────────────────────────────

describe('AnalysisTimeoutError', () => {
  it('has correct name and message', () => {
    const err = new AnalysisTimeoutError('job-123');
    expect(err.name).toBe('AnalysisTimeoutError');
    expect(err.message).toContain('job-123');
    expect(err.message).toContain('timed out');
  });
});

// ── mapVerdict ──────────────────────────────────────────────────────────────

describe('mapVerdict', () => {
  it('maps "fake" to FAKE', () => {
    expect(mapVerdict('fake')).toBe('FAKE');
  });

  it('maps "real" to REAL', () => {
    expect(mapVerdict('real')).toBe('REAL');
  });

  it('maps "unknown" to UNKNOWN', () => {
    expect(mapVerdict('unknown')).toBe('UNKNOWN');
  });

  it('maps any other string to UNKNOWN', () => {
    expect(mapVerdict('error')).toBe('UNKNOWN');
    expect(mapVerdict('')).toBe('UNKNOWN');
  });
});

// ── mapBranchScores ─────────────────────────────────────────────────────────

describe('mapBranchScores', () => {
  it('maps spatial, frequency, attention → fused', () => {
    const scores = mapBranchScores({ spatial: 0.8, frequency: 0.7, attention: 0.75 });
    expect(scores.spatial).toBe(0.8);
    expect(scores.frequency).toBe(0.7);
    expect(scores.fused).toBe(0.75);
  });

  it('falls back to mean(spatial, frequency) when attention absent', () => {
    const scores = mapBranchScores({ spatial: 0.6, frequency: 0.4 });
    expect(scores.fused).toBeCloseTo(0.5);
  });

  it('clamps values to [0, 1]', () => {
    const scores = mapBranchScores({ spatial: 1.5, frequency: -0.3 });
    expect(scores.spatial).toBe(1);
    expect(scores.frequency).toBe(0);
  });

  it('handles null input', () => {
    const scores = mapBranchScores(null);
    expect(scores.spatial).toBe(0);
    expect(scores.frequency).toBe(0);
    expect(scores.fused).toBe(0);
  });

  it('handles undefined input', () => {
    const scores = mapBranchScores(undefined);
    expect(scores.spatial).toBe(0);
    expect(scores.frequency).toBe(0);
    expect(scores.fused).toBe(0);
  });
});

// ── mapImageResponse ───────────────────────────────────────────────────────

describe('mapImageResponse', () => {
  it('maps a successful image detection response', () => {
    const response: BackendImageResponse = {
      job_id: 'abc-123',
      verdict: 'fake',
      confidence: 0.87,
      fake_probability: 0.87,
      real_probability: 0.13,
      faces_detected: 1,
      branch_scores: { spatial: 0.85, frequency: 0.80, attention: 0.82 },
      heatmap_url: '/heatmaps/abc.png',
      processing_ms: 1200,
      random_weights_warning: false,
    };

    const result = mapImageResponse(response, mockFile('test.jpg', 2048, 'image/jpeg'));

    expect(result.id).toBe('abc-123');
    expect(result.label).toBe('FAKE');
    expect(result.confidence).toBe(0.87);
    expect(result.branchScores.spatial).toBe(0.85);
    expect(result.branchScores.fused).toBe(0.82);
    expect(result.heatmapUrl).toBe('/heatmaps/abc.png');
    expect(result.facesDetected).toBe(1);
    expect(result.fakeProbability).toBe(0.87);
    expect(result.realProbability).toBe(0.13);
    expect(result.randomWeightsWarning).toBe(false);
  });

  it('maps an UNKNOWN verdict when no faces detected', () => {
    const response: BackendImageResponse = {
      job_id: 'abc-456',
      verdict: 'unknown',
      confidence: 0.0,
      fake_probability: 0.0,
      real_probability: 0.0,
      faces_detected: 0,
      branch_scores: {},
      heatmap_url: null,
      processing_ms: 500,
      random_weights_warning: false,
    };

    const result = mapImageResponse(response, mockFile('no-face.jpg', 2048, 'image/jpeg'));

    expect(result.label).toBe('UNKNOWN');
    expect(result.confidence).toBe(0.0);
    expect(result.facesDetected).toBe(0);
    expect(result.heatmapUrl).toBeUndefined();
    expect(result.warnings).toContain(
      'Inconclusive: no face was detected in this image.',
    );
  });

  it('propagates random_weights_warning from backend', () => {
    const response: BackendImageResponse = {
      job_id: 'abc-789',
      verdict: 'real',
      confidence: 0.92,
      fake_probability: 0.08,
      real_probability: 0.92,
      faces_detected: 1,
      branch_scores: { spatial: 0.1, frequency: 0.15, attention: 0.12 },
      heatmap_url: null,
      processing_ms: 800,
      random_weights_warning: true,
    };

    const result = mapImageResponse(response, mockFile('test.png', 2048, 'image/png'));

    expect(result.randomWeightsWarning).toBe(true);
    expect(result.warnings[0]).toContain('random');
    expect(result.warnings[0]).toContain('meaningless');
  });

  it('handles missing branch_scores gracefully', () => {
    const response: BackendImageResponse = {
      job_id: 'abc-000',
      verdict: 'real',
      confidence: 0.9,
      fake_probability: 0.1,
      real_probability: 0.9,
      faces_detected: 1,
      branch_scores: {},
      heatmap_url: null,
      processing_ms: 300,
      random_weights_warning: false,
    };

    const result = mapImageResponse(response, mockFile('test.jpg', 1024, 'image/jpeg'));
    expect(result.branchScores.spatial).toBe(0);
    expect(result.branchScores.frequency).toBe(0);
    expect(result.branchScores.fused).toBe(0);
  });
});

// ── mapVideoJobResponse ────────────────────────────────────────────────────

describe('mapVideoJobResponse with null frame scores', () => {
  it('preserves null scores for unscored frames', () => {
    const job: BackendJobResponse = {
      id: 'vid-123',
      original_filename: 'test.mp4',
      media_type: 'video',
      file_size_bytes: 50000,
      status: 'completed',
      error_msg: null,
      started_at: '2025-01-01T00:00:00Z',
      ended_at: '2025-01-01T00:00:05Z',
      created_at: '2025-01-01T00:00:00Z',
      result: {
        id: 'res-123',
        verdict: 'unknown',
        confidence: 0.0,
        branch_scores: {},
        frame_scores: [
          { frame: 0, score: 0.82, timestamp: 0.0, face_status: 'detected', window_count: 2 },
          { frame: 1, score: null, timestamp: 1.0, face_status: 'not_found', window_count: 0 },
          { frame: 2, score: null, timestamp: 2.0, face_status: 'failed', window_count: 0 },
          { frame: 3, score: 0.15, timestamp: 3.0, face_status: 'detected', window_count: 1 },
        ],
        chunk_scores: null,
        heatmap_url: null,
        duration_sec: 4.0,
        frames_analyzed: 4,
        faces_detected: 2,
        raw_output: {
          fake_probability: 0.0,
          real_probability: 0.0,
          suspicious_timestamps: [],
          faces_not_found: 1,
          faces_failed: 1,
          scored_frames_count: 2,
        },
        created_at: '2025-01-01T00:00:05Z',
      },
    };

    const result = mapVideoJobResponse(job, mockFile('test.mp4', 50000, 'video/mp4'));

    expect(result.label).toBe('UNKNOWN');
    expect(result.frameResults).toHaveLength(4);

    // Frame 0: scored as FAKE
    expect(result.frameResults![0].fusedScore).toBe(0.82);
    expect(result.frameResults![0].label).toBe('FAKE');

    // Frame 1: null score → UNKNOWN, not 0 or 0.5
    expect(result.frameResults![1].fusedScore).toBeNull();
    expect(result.frameResults![1].label).toBe('UNKNOWN');

    // Frame 2: null score → UNKNOWN
    expect(result.frameResults![2].fusedScore).toBeNull();
    expect(result.frameResults![2].label).toBe('UNKNOWN');

    // Frame 3: scored as REAL
    expect(result.frameResults![3].fusedScore).toBe(0.15);
    expect(result.frameResults![3].label).toBe('REAL');

    // Extended metadata
    expect(result.facesNotFound).toBe(1);
    expect(result.facesFailed).toBe(1);
    expect(result.scoredFramesCount).toBe(2);
    expect(result.metadata.framesAnalyzed).toBe(4);
    expect(result.metadata.duration).toBe(4.0);
  });

  it('handles all-null frame scores (no faces detected anywhere)', () => {
    const job: BackendJobResponse = {
      id: 'vid-no-faces',
      original_filename: 'test.mp4',
      media_type: 'video',
      file_size_bytes: 50000,
      status: 'completed',
      error_msg: null,
      started_at: '2025-01-01T00:00:00Z',
      ended_at: '2025-01-01T00:00:10Z',
      created_at: '2025-01-01T00:00:00Z',
      result: {
        id: 'res-456',
        verdict: 'unknown',
        confidence: 0.0,
        branch_scores: {},
        frame_scores: [
          { frame: 0, score: null, timestamp: 0.0, face_status: 'not_found', window_count: 0 },
          { frame: 1, score: null, timestamp: 1.0, face_status: 'not_found', window_count: 0 },
          { frame: 2, score: null, timestamp: 2.0, face_status: 'not_found', window_count: 0 },
        ],
        chunk_scores: null,
        heatmap_url: null,
        duration_sec: 3.0,
        frames_analyzed: 3,
        faces_detected: 0,
        raw_output: {
          fake_probability: 0.0,
          real_probability: 0.0,
          suspicious_timestamps: [],
          faces_not_found: 3,
          faces_failed: 0,
          scored_frames_count: 0,
        },
        created_at: '2025-01-01T00:00:10Z',
      },
    };

    const result = mapVideoJobResponse(job, mockFile('test.mp4', 50000, 'video/mp4'));

    expect(result.label).toBe('UNKNOWN');
    expect(result.frameResults!.every((f) => f.fusedScore === null)).toBe(true);
    expect(result.frameResults!.every((f) => f.label === 'UNKNOWN')).toBe(true);
    expect(result.scoredFramesCount).toBe(0);
    expect(result.warnings).toContain(
      'Inconclusive: no frames could be scored (no faces detected).',
    );
  });

  it('handles older result with missing optional fields', () => {
    // Simulate an older record that predates some fields (raw_output partially populated)
    const job: BackendJobResponse = {
      id: 'vid-old',
      original_filename: 'old.mp4',
      media_type: 'video',
      file_size_bytes: 50000,
      status: 'completed',
      error_msg: null,
      started_at: '2025-01-01T00:00:00Z',
      ended_at: '2025-01-01T00:00:08Z',
      created_at: '2025-01-01T00:00:00Z',
      result: {
        id: 'res-old',
        verdict: 'fake',
        confidence: 0.88,
        branch_scores: { spatial: 0.9, frequency: 0.85, attention: 0.87 },
        frame_scores: [
          { frame: 0, score: 0.88, timestamp: 0.0 },
        ],
        chunk_scores: null,
        heatmap_url: '/heatmaps/old.png',
        duration_sec: 1.0,
        frames_analyzed: 1,
        faces_detected: 1,
        // raw_output is null — older record
        raw_output: null,
        created_at: '2025-01-01T00:00:08Z',
      },
    };

    const result = mapVideoJobResponse(job, mockFile('old.mp4', 50000, 'video/mp4'));

    expect(result.label).toBe('FAKE');
    expect(result.confidence).toBe(0.88);
    expect(result.heatmapUrl).toBe('/heatmaps/old.png');
    // Missing raw_output fields default to null/0/empty — not fabricated
    expect(result.fakeProbability).toBeNull();
    expect(result.realProbability).toBeNull();
    expect(result.facesNotFound).toBe(0);
    expect(result.facesFailed).toBe(0);
    expect(result.scoredFramesCount).toBe(0);
    expect(result.suspiciousTimestamps).toEqual([]);
  });
});
