import { useState } from 'react';
import { motion } from 'framer-motion';
import type { AnalysisResult } from '../types';

function fmt(b: number) {
  if (b < 1048576) return `${(b / 1024).toFixed(0)} KB`;
  return `${(b / 1048576).toFixed(1)} MB`;
}

function pad(s: string, n: number) { return s.padEnd(n); }

export default function ForensicReport({ result }: { result: AnalysisResult }) {
  const [copied, setCopied] = useState(false);
  const isFake = result.label === 'FAKE';
  const accent = isFake ? 'var(--red)' : 'var(--green)';
  const id = `VFY-${result.id.slice(-8).toUpperCase()}`;
  const ts = new Date().toISOString().replace('T', ' ').slice(0, 19) + ' UTC';

  const reportJson = {
    report_id: id,
    timestamp: ts,
    engine: 'Verifai v2.4.0',
    verdict: result.label,
    confidence: `${Math.round(result.confidence * 100)}%`,
    media: {
      filename: result.metadata.filename,
      type: result.mediaType,
      size: fmt(result.metadata.fileSize),
      dimensions: result.metadata.dimensions ?? null,
      duration_sec: result.metadata.duration ?? null,
      frames_analyzed: result.metadata.framesAnalyzed ?? null,
    },
    scores: {
      spatial: `${Math.round(result.branchScores.spatial * 100)}%`,
      frequency: `${Math.round(result.branchScores.frequency * 100)}%`,
      fused: `${Math.round(result.branchScores.fused * 100)}%`,
    },
    frame_results: result.frameResults?.map(f => ({
      frame: f.frameIndex,
      time: `${f.timestamp.toFixed(2)}s`,
      verdict: f.label,
      score: f.fusedScore !== null ? `${Math.round(f.fusedScore * 100)}%` : 'N/A',
    })) ?? null,
    warnings: result.warnings,
    disclaimer: 'Probabilistic AI output. Not forensic or legal evidence.',
  };

  const jsonStr = JSON.stringify(reportJson, null, 2);

  const download = () => {
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([jsonStr], { type: 'application/json' }));
    a.download = `${id}.json`;
    a.click();
  };

  const copy = () => {
    navigator.clipboard.writeText(jsonStr);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const lines: { text: string; color: string }[] = [
    { text: `# VERIFAI FORENSIC ANALYSIS REPORT`, color: 'var(--cyan)' },
    { text: `# ─────────────────────────────────────────`, color: 'var(--border-2)' },
    { text: ``, color: '' },
    { text: `  report_id       ${pad(id, 20)}`, color: 'var(--ink-2)' },
    { text: `  timestamp       ${ts}`, color: 'var(--ink-2)' },
    { text: `  engine          Verifai v2.4.0`, color: 'var(--ink-2)' },
    { text: `  filename        ${result.metadata.filename}`, color: 'var(--ink-2)' },
    { text: `  media_type      ${result.mediaType}`, color: 'var(--ink-2)' },
    { text: `  file_size       ${fmt(result.metadata.fileSize)}`, color: 'var(--ink-2)' },
    ...(result.metadata.dimensions ? [{ text: `  dimensions      ${result.metadata.dimensions.width}×${result.metadata.dimensions.height}`, color: 'var(--ink-2)' }] : []),
    ...(result.metadata.framesAnalyzed ? [{ text: `  frames          ${result.metadata.framesAnalyzed} sampled`, color: 'var(--ink-2)' }] : []),
    { text: `  processing_time ${(result.processingTime / 1000).toFixed(2)}s`, color: 'var(--ink-2)' },
    { text: ``, color: '' },
    { text: `  spatial_score   ${Math.round(result.branchScores.spatial * 100)}%`, color: 'var(--cyan)' },
    { text: `  frequency_score ${Math.round(result.branchScores.frequency * 100)}%`, color: 'var(--violet)' },
    { text: `  fused_score     ${Math.round(result.branchScores.fused * 100)}%`, color: accent },
    { text: ``, color: '' },
    { text: `  ┌──────────────────────────────────────┐`, color: accent },
    { text: `  │  verdict       ${pad(result.label, 22)}│`, color: accent },
    { text: `  │  confidence    ${pad(Math.round(result.confidence * 100) + '%', 22)}│`, color: accent },
    { text: `  │  status        ${pad(isFake ? 'SYNTHETIC DETECTED' : 'AUTHENTIC', 22)}│`, color: accent },
    { text: `  └──────────────────────────────────────┘`, color: accent },
    { text: ``, color: '' },
    { text: `  ! Not forensic or legal evidence.`, color: 'rgba(245,158,11,0.7)' },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.5 }}
      style={{ background: 'var(--bg-1)', border: '1px solid var(--border)', borderRadius: 16, overflow: 'hidden' }}
    >
      {/* Window bar */}
      <div style={{
        padding: '10px 16px', borderBottom: '1px solid var(--border)',
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        background: 'var(--surface)',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ display: 'flex', gap: 6 }}>
            {['#ff5f57', '#febc2e', '#28c840'].map(c => (
              <div key={c} style={{ width: 9, height: 9, borderRadius: '50%', background: c, opacity: 0.75 }} />
            ))}
          </div>
          <span style={{ fontFamily: 'JetBrains Mono', fontSize: 11, color: 'var(--ink-3)' }}>
            forensic-report — {id}
          </span>
        </div>
        <div style={{ display: 'flex', gap: 8 }}>
          <button onClick={copy}
            style={{ background: 'none', border: '1px solid var(--border)', borderRadius: 7, padding: '4px 12px', fontSize: 11, fontWeight: 600, color: 'var(--ink-2)', cursor: 'pointer', fontFamily: 'Inter' }}>
            {copied ? '✓ Copied' : 'Copy'}
          </button>
          <motion.button onClick={download} whileHover={{ scale: 1.04 }} whileTap={{ scale: 0.97 }}
            style={{ background: 'var(--cyan-dim)', border: '1px solid rgba(46,232,255,0.25)', borderRadius: 7, padding: '4px 12px', fontSize: 11, fontWeight: 600, color: 'var(--cyan)', cursor: 'pointer', fontFamily: 'Inter' }}>
            ↓ Download JSON
          </motion.button>
        </div>
      </div>

      {/* Terminal body */}
      <div style={{ padding: '16px 20px', maxHeight: 320, overflowY: 'auto', fontFamily: 'JetBrains Mono', fontSize: 12, lineHeight: 1.9, background: 'rgba(0,0,0,0.3)' }}>
        {lines.map((line, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ delay: 0.6 + i * 0.035, duration: 0.1 }}
            style={{ color: line.color, whiteSpace: 'pre' }}
          >
            {line.text}
          </motion.div>
        ))}
      </div>
    </motion.div>
  );
}
