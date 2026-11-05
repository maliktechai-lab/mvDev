/**
 * JobHistory
 * ==========
 * Slide-in panel showing the current user's past detection jobs.
 * Fetches GET /api/v1/jobs (paginated, 20 per page).
 *
 * Design: matches BLINK monochrome aesthetic — monospaced, raw borders.
 */

import { useEffect, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { apiClient, isLoggedIn } from '../api/realApi';

// ── Types ──────────────────────────────────────────────────────────────────

interface JobSummary {
  id: string;
  original_filename: string;
  media_type: 'image' | 'video' | 'audio';
  file_size_bytes: number;
  status: 'pending' | 'running' | 'completed' | 'failed';
  error_msg: string | null;
  started_at: string | null;
  ended_at: string | null;
  created_at: string;
  result: {
    verdict: 'real' | 'fake' | 'unknown';
    confidence: number;
  } | null;
}

interface Props {
  open: boolean;
  onClose: () => void;
}

// ── Helpers ────────────────────────────────────────────────────────────────

function verdictColor(verdict: string) {
  if (verdict === 'fake') return '#ff2020';
  if (verdict === 'real') return '#00ff88';
  return '#f59e0b';
}

function statusColor(status: string) {
  if (status === 'completed') return '#00ff88';
  if (status === 'failed')    return '#ff2020';
  if (status === 'running')   return '#f59e0b';
  return 'var(--grey-2)';
}

function mediaIcon(type: string) {
  if (type === 'video') return '▶';
  if (type === 'audio') return '♫';
  return '◼';
}

function relativeTime(iso: string) {
  const diff = Date.now() - new Date(iso).getTime();
  const mins = Math.floor(diff / 60_000);
  if (mins < 1)  return 'just now';
  if (mins < 60) return `${mins}m ago`;
  const hrs = Math.floor(mins / 60);
  if (hrs < 24)  return `${hrs}h ago`;
  return `${Math.floor(hrs / 24)}d ago`;
}

function fmtSize(bytes: number) {
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(0)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

// ── Component ──────────────────────────────────────────────────────────────

export default function JobHistory({ open, onClose }: Props) {
  const [jobs, setJobs] = useState<JobSummary[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [page, setPage] = useState(1);
  const [hasMore, setHasMore] = useState(true);

  const PAGE_SIZE = 20;

  const fetchJobs = useCallback(async (pageNum: number, append = false) => {
    if (!isLoggedIn()) {
      setError('Sign in to view your analysis history.');
      return;
    }
    setLoading(true);
    setError(null);
    try {
      const res = await apiClient.get<JobSummary[]>('/jobs', {
        params: { page: pageNum, page_size: PAGE_SIZE },
      });
      const data = res.data;
      setJobs((prev) => append ? [...prev, ...data] : data);
      setHasMore(data.length === PAGE_SIZE);
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : 'Failed to load history.';
      setError(msg);
    } finally {
      setLoading(false);
    }
  }, []);

  // Reload when panel opens
  useEffect(() => {
    if (open) {
      setPage(1);
      setJobs([]);
      fetchJobs(1, false);
    }
  }, [open, fetchJobs]);

  const loadMore = () => {
    const next = page + 1;
    setPage(next);
    fetchJobs(next, true);
  };

  return (
    <AnimatePresence>
      {open && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: 0.2 }}
            onClick={onClose}
            style={{
              position: 'fixed', inset: 0, zIndex: 8000,
              background: 'rgba(0,0,0,0.6)', backdropFilter: 'blur(4px)',
            }}
          />

          {/* Panel */}
          <motion.aside
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', stiffness: 340, damping: 36 }}
            style={{
              position: 'fixed', top: 0, right: 0, bottom: 0, zIndex: 8001,
              width: '100%', maxWidth: 480,
              background: '#000',
              borderLeft: '1px solid var(--border)',
              display: 'flex', flexDirection: 'column',
              overflow: 'hidden',
            }}
          >
            {/* Header */}
            <div style={{
              padding: '22px 28px',
              borderBottom: '1px solid var(--border)',
              display: 'flex', alignItems: 'center', justifyContent: 'space-between',
              flexShrink: 0,
            }}>
              <div>
                <div style={{
                  fontFamily: 'JetBrains Mono', fontSize: 9,
                  letterSpacing: '0.2em', color: 'var(--grey-2)',
                  textTransform: 'uppercase', marginBottom: 6,
                }}>
                  — Analysis History
                </div>
                <div style={{
                  fontFamily: 'Syne', fontWeight: 800,
                  fontSize: 22, letterSpacing: '-0.03em', color: 'var(--white)',
                }}>
                  PAST JOBS
                </div>
              </div>
              <button
                onClick={onClose}
                style={{
                  background: 'none', border: '1px solid var(--border)',
                  color: 'var(--grey-2)', cursor: 'pointer',
                  padding: '6px 12px',
                  fontFamily: 'JetBrains Mono', fontSize: 10,
                  letterSpacing: '0.1em', textTransform: 'uppercase',
                  transition: 'color 0.2s, border-color 0.2s',
                }}
                onMouseEnter={(e) => {
                  (e.currentTarget as HTMLButtonElement).style.color = 'var(--white)';
                  (e.currentTarget as HTMLButtonElement).style.borderColor = 'var(--grey-2)';
                }}
                onMouseLeave={(e) => {
                  (e.currentTarget as HTMLButtonElement).style.color = 'var(--grey-2)';
                  (e.currentTarget as HTMLButtonElement).style.borderColor = 'var(--border)';
                }}
              >
                ✕ Close
              </button>
            </div>

            {/* Body */}
            <div style={{ flex: 1, overflowY: 'auto', padding: '0 0 40px' }}>

              {/* Error state */}
              {error && (
                <div style={{
                  margin: '24px 28px',
                  padding: '14px 16px',
                  border: '1px solid rgba(255,32,32,0.3)',
                  fontFamily: 'JetBrains Mono', fontSize: 11,
                  color: '#ff2020', letterSpacing: '0.04em',
                }}>
                  ✕ {error}
                </div>
              )}

              {/* Empty state */}
              {!loading && !error && jobs.length === 0 && (
                <div style={{
                  display: 'flex', flexDirection: 'column',
                  alignItems: 'center', justifyContent: 'center',
                  padding: '80px 28px', gap: 16, textAlign: 'center',
                }}>
                  <div style={{
                    fontFamily: 'Syne', fontWeight: 800,
                    fontSize: 18, color: 'var(--grey-3)', letterSpacing: '-0.02em',
                  }}>
                    No jobs yet
                  </div>
                  <div style={{
                    fontFamily: 'JetBrains Mono', fontSize: 10,
                    color: 'var(--grey-3)', letterSpacing: '0.1em', textTransform: 'uppercase',
                  }}>
                    Run a detection to see results here
                  </div>
                </div>
              )}

              {/* Job list */}
              {jobs.map((job, i) => (
                <JobRow key={job.id} job={job} isFirst={i === 0} />
              ))}

              {/* Loading indicator */}
              {loading && (
                <div style={{ padding: '28px', textAlign: 'center' }}>
                  <motion.div
                    animate={{ opacity: [1, 0.3, 1] }}
                    transition={{ repeat: Infinity, duration: 0.8 }}
                    style={{
                      fontFamily: 'JetBrains Mono', fontSize: 10,
                      color: 'var(--grey-2)', letterSpacing: '0.14em',
                      textTransform: 'uppercase',
                    }}
                  >
                    Loading…
                  </motion.div>
                </div>
              )}

              {/* Load more */}
              {!loading && hasMore && jobs.length > 0 && (
                <div style={{ padding: '16px 28px' }}>
                  <button
                    onClick={loadMore}
                    style={{
                      width: '100%', padding: '12px',
                      background: 'none',
                      border: '1px solid var(--border)',
                      color: 'var(--grey-2)',
                      fontFamily: 'JetBrains Mono', fontSize: 10,
                      letterSpacing: '0.12em', textTransform: 'uppercase',
                      cursor: 'pointer',
                      transition: 'color 0.2s, border-color 0.2s',
                    }}
                    onMouseEnter={(e) => {
                      (e.currentTarget as HTMLButtonElement).style.color = 'var(--white)';
                      (e.currentTarget as HTMLButtonElement).style.borderColor = 'var(--grey-2)';
                    }}
                    onMouseLeave={(e) => {
                      (e.currentTarget as HTMLButtonElement).style.color = 'var(--grey-2)';
                      (e.currentTarget as HTMLButtonElement).style.borderColor = 'var(--border)';
                    }}
                  >
                    Load more ↓
                  </button>
                </div>
              )}
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}

// ── Job row ────────────────────────────────────────────────────────────────

function JobRow({ job, isFirst }: { job: JobSummary; isFirst: boolean }) {
  const verdict = job.result?.verdict;
  const confidence = job.result?.confidence;

  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.25 }}
      style={{
        padding: '18px 28px',
        borderBottom: '1px solid var(--border)',
        borderTop: isFirst ? '1px solid var(--border)' : 'none',
        display: 'grid',
        gridTemplateColumns: '1fr auto',
        gap: 12,
        alignItems: 'start',
      }}
    >
      {/* Left */}
      <div style={{ minWidth: 0 }}>
        {/* Filename + media icon */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6,
        }}>
          <span style={{
            fontFamily: 'JetBrains Mono', fontSize: 10,
            color: 'var(--grey-2)', flexShrink: 0,
          }}>
            {mediaIcon(job.media_type)}
          </span>
          <span style={{
            fontFamily: 'Inter', fontSize: 13, fontWeight: 500,
            color: 'var(--white)',
            overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap',
          }}>
            {job.original_filename}
          </span>
        </div>

        {/* Meta row */}
        <div style={{
          display: 'flex', alignItems: 'center', gap: 12, flexWrap: 'wrap',
        }}>
          {/* Status badge */}
          <span style={{
            fontFamily: 'JetBrains Mono', fontSize: 9,
            letterSpacing: '0.12em', textTransform: 'uppercase',
            color: statusColor(job.status),
          }}>
            {job.status}
          </span>

          <span style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: 'var(--grey-3)' }}>
            {job.media_type.toUpperCase()}
          </span>

          <span style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: 'var(--grey-3)' }}>
            {fmtSize(job.file_size_bytes)}
          </span>

          <span style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: 'var(--grey-3)' }}>
            {relativeTime(job.created_at)}
          </span>
        </div>

        {/* Error message */}
        {job.error_msg && (
          <div style={{
            marginTop: 6,
            fontFamily: 'JetBrains Mono', fontSize: 10,
            color: '#ff2020', letterSpacing: '0.03em',
          }}>
            {job.error_msg}
          </div>
        )}
      </div>

      {/* Right: verdict */}
      <div style={{ textAlign: 'right', flexShrink: 0 }}>
        {verdict ? (
          <>
            <div style={{
              fontFamily: 'Syne', fontWeight: 800,
              fontSize: 13, letterSpacing: '0.04em',
              color: verdictColor(verdict),
              textTransform: 'uppercase',
            }}>
              {verdict}
            </div>
            {confidence != null && (
              <div style={{
                fontFamily: 'JetBrains Mono', fontSize: 9,
                color: 'var(--grey-2)', marginTop: 3, letterSpacing: '0.06em',
              }}>
                {(confidence * 100).toFixed(1)}%
              </div>
            )}
          </>
        ) : (
          <span style={{
            fontFamily: 'JetBrains Mono', fontSize: 9,
            color: 'var(--grey-3)', letterSpacing: '0.1em',
          }}>
            —
          </span>
        )}
      </div>
    </motion.div>
  );
}
