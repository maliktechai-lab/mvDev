import { motion } from 'framer-motion';
import type { AnalysisResult } from '../types';

function fmtBytes(b: number) {
  if (b < 1048576) return `${(b / 1024).toFixed(0)} KB`;
  return `${(b / 1048576).toFixed(1)} MB`;
}

export default function AnalysisResultCard({ result }: { result: AnalysisResult }) {
  const isFake = result.label === 'FAKE';
  const pct = Math.round(result.confidence * 100);
  const accent = isFake ? 'var(--red)' : 'var(--green)';
  const accentDim = isFake ? 'var(--red-dim)' : 'var(--green-dim)';
  const circ = 2 * Math.PI * 42;

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.96, y: 16 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.25, 0.46, 0.45, 0.94] }}
      className="card-elevated"
      style={{ overflow: 'hidden' }}
    >
      {/* Verdict stripe */}
      <div style={{
        height: 3,
        background: isFake
          ? 'linear-gradient(90deg, var(--red), transparent)'
          : 'linear-gradient(90deg, var(--green), transparent)',
      }} />

      <div style={{ padding: 24 }}>
        {/* Top row */}
        <div style={{ display: 'flex', alignItems: 'flex-start', gap: 20, marginBottom: 24 }}>
          {/* Confidence ring */}
          <div style={{ position: 'relative', width: 96, height: 96, flexShrink: 0 }}>
            <svg width="96" height="96" viewBox="0 0 100 100" style={{ transform: 'rotate(-90deg)' }}>
              <circle cx="50" cy="50" r="42" fill="none" stroke="var(--surface-2)" strokeWidth="5" />
              <motion.circle
                cx="50" cy="50" r="42" fill="none"
                stroke={accent} strokeWidth="5" strokeLinecap="round"
                strokeDasharray={circ}
                initial={{ strokeDashoffset: circ }}
                animate={{ strokeDashoffset: circ * (1 - result.confidence) }}
                transition={{ delay: 0.3, duration: 1.4, ease: [0.16, 1, 0.3, 1] }}
              />
            </svg>
            <div style={{
              position: 'absolute', inset: 0,
              display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center',
            }}>
              <motion.span
                initial={{ opacity: 0 }}
                animate={{ opacity: 1 }}
                transition={{ delay: 0.5 }}
                style={{ fontFamily: 'Syne', fontSize: 22, fontWeight: 800, color: accent, lineHeight: 1 }}
              >
                {pct}%
              </motion.span>
              <span style={{ fontSize: 10, color: 'var(--ink-3)', fontWeight: 500, marginTop: 2 }}>confidence</span>
            </div>
          </div>

          {/* Verdict text */}
          <div style={{ flex: 1, paddingTop: 4 }}>
            <motion.div
              initial={{ opacity: 0, x: -12 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.2 }}
            >
              <div style={{ fontSize: 11, fontWeight: 600, letterSpacing: '0.08em', textTransform: 'uppercase', color: 'var(--ink-3)', marginBottom: 6 }}>
                Verdict
              </div>
              <div style={{ fontFamily: 'Syne', fontSize: 32, fontWeight: 800, color: accent, letterSpacing: '-0.03em', lineHeight: 1 }}>
                {result.label}
              </div>
              <div style={{ marginTop: 8, fontSize: 13, color: 'var(--ink-2)', lineHeight: 1.5 }}>
                {isFake
                  ? 'Synthetic or manipulated content detected by dual-branch analysis.'
                  : 'No manipulation signatures detected. Content appears authentic.'}
              </div>
              <div style={{ marginTop: 12, display: 'flex', gap: 8 }}>
                <span className="pill" style={{ background: accentDim, border: `1px solid ${accent}30`, color: accent, textTransform: 'uppercase', fontSize: 10 }}>
                  {isFake ? 'High Risk' : 'Authentic'}
                </span>
                <span className="pill" style={{ background: 'var(--surface)', border: '1px solid var(--border)', color: 'var(--ink-3)', textTransform: 'uppercase', fontSize: 10 }}>
                  Verifai v2.4
                </span>
              </div>
            </motion.div>
          </div>
        </div>

        {/* Divider */}
        <div className="divider" style={{ marginBottom: 20 }} />

        {/* Meta grid */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
          {[
            { label: 'File', value: result.metadata.filename },
            { label: 'Size', value: fmtBytes(result.metadata.fileSize) },
            { label: 'Processed in', value: `${(result.processingTime / 1000).toFixed(2)}s` },
            {
              label: 'Media',
              value: result.mediaType === 'video'
                ? `Video · ${result.metadata.framesAnalyzed} frames`
                : `${result.metadata.dimensions?.width}×${result.metadata.dimensions?.height}`
            },
          ].map(({ label, value }) => (
            <div key={label} style={{ background: 'var(--surface)', border: '1px solid var(--border)', borderRadius: 10, padding: '10px 14px' }}>
              <div style={{ fontSize: 11, color: 'var(--ink-3)', fontWeight: 500, marginBottom: 3 }}>{label}</div>
              <div style={{ fontSize: 13, fontWeight: 600, color: 'var(--ink)', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{value}</div>
            </div>
          ))}
        </div>

        {/* Disclaimer */}
        {result.warnings.length > 0 && (
          <div style={{ marginTop: 16, padding: '10px 14px', borderRadius: 10, background: 'rgba(245,158,11,0.06)', border: '1px solid rgba(245,158,11,0.15)', fontSize: 12, color: 'rgba(245,158,11,0.8)', lineHeight: 1.5 }}>
            {result.warnings[0]}
          </div>
        )}
      </div>
    </motion.div>
  );
}
