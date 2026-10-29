import { motion } from 'framer-motion';
import { RadarChart, PolarGrid, PolarAngleAxis, Radar, ResponsiveContainer, Tooltip } from 'recharts';
import type { BranchScores as BSType } from '../types';

interface Props { scores: BSType; label: 'REAL' | 'FAKE' | 'UNKNOWN'; }

function Bar({ value, color, delay }: { value: number; color: string; delay: number }) {
  return (
    <div style={{ height: 4, background: 'var(--surface-2)', borderRadius: 2, overflow: 'hidden' }}>
      <motion.div
        initial={{ width: 0 }}
        animate={{ width: `${value * 100}%` }}
        transition={{ delay, duration: 1.2, ease: [0.16, 1, 0.3, 1] }}
        style={{ height: '100%', background: color, borderRadius: 2 }}
      />
    </div>
  );
}

export default function BranchScoresPanel({ scores, label }: Props) {
  const isFake = label === 'FAKE';
  const accent = isFake ? 'var(--red)' : 'var(--green)';

  const radar = [
    { m: 'Spatial',   v: Math.round(scores.spatial * 100) },
    { m: 'Frequency', v: Math.round(scores.frequency * 100) },
    { m: 'Fused',     v: Math.round(scores.fused * 100) },
    { m: 'Margin',    v: Math.round(Math.abs(scores.fused - 0.5) * 200) },
    { m: 'Agreement', v: Math.round((1 - Math.abs(scores.spatial - scores.frequency)) * 100) },
  ];

  const branches = [
    { name: 'Spatial CNN', sub: 'EfficientNet-B4', value: scores.spatial, color: 'var(--cyan)' },
    { name: 'Frequency',   sub: 'FFT + Conv',       value: scores.frequency, color: 'var(--violet)' },
    { name: 'Fused',       sub: 'Branch fusion',    value: scores.fused, color: accent },
  ];

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.15 }}
      className="card"
      style={{ padding: 24 }}
    >
      {/* Header */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 24 }}>
        <div>
          <div style={{ fontFamily: 'Syne', fontSize: 16, fontWeight: 700, color: 'var(--ink)' }}>Branch Scores</div>
          <div style={{ fontSize: 12, color: 'var(--ink-3)', marginTop: 2 }}>Spatial + frequency analysis breakdown</div>
        </div>
        <span className="pill" style={{ background: isFake ? 'var(--red-dim)' : 'var(--green-dim)', border: `1px solid ${isFake ? 'rgba(255,69,96,0.25)' : 'rgba(13,206,138,0.25)'}`, color: accent }}>
          {label}
        </span>
      </div>

      {/* Branch bars */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 20, marginBottom: 24 }}>
        {branches.map(({ name, sub, value, color }, i) => (
          <div key={name}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: 8 }}>
              <div>
                <span style={{ fontSize: 13, fontWeight: 600, color: 'var(--ink)' }}>{name}</span>
                <span style={{ fontSize: 11, color: 'var(--ink-3)', marginLeft: 8 }}>{sub}</span>
              </div>
              <span style={{ fontFamily: 'JetBrains Mono', fontSize: 13, fontWeight: 600, color }}>
                {Math.round(value * 100)}%
              </span>
            </div>
            <Bar value={value} color={color} delay={0.3 + i * 0.15} />
          </div>
        ))}
      </div>

      <div className="divider" style={{ marginBottom: 20 }} />

      {/* Radar */}
      <div style={{ height: 200 }}>
        <ResponsiveContainer width="100%" height="100%">
          <RadarChart data={radar}>
            <PolarGrid stroke="var(--border)" />
            <PolarAngleAxis dataKey="m" tick={{ fill: 'var(--ink-3)', fontSize: 11, fontFamily: 'Inter' }} />
            <Radar dataKey="v" stroke={accent} fill={accent} fillOpacity={0.08} strokeWidth={1.5} />
            <Tooltip
              contentStyle={{
                background: 'var(--bg-2)', border: '1px solid var(--border-2)',
                borderRadius: 10, color: 'var(--ink)', fontSize: 12,
                fontFamily: 'JetBrains Mono',
              }}
              formatter={(v) => [`${v}%`, ''] as [string, string]}
            />
          </RadarChart>
        </ResponsiveContainer>
      </div>

      {/* Branch descriptions */}
      <div className="divider" style={{ margin: '20px 0' }} />
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
        {[
          { title: 'Spatial Branch', body: 'Scans pixel-level textures, blending seams, lighting artifacts via EfficientNet-B4.', color: 'var(--cyan)' },
          { title: 'Frequency Branch', body: 'Reveals spectral anomalies via 2D FFT invisible to the human eye.', color: 'var(--violet)' },
        ].map(({ title, body, color }) => (
          <div key={title} style={{ padding: '12px 14px', borderRadius: 10, background: 'var(--surface)', border: '1px solid var(--border)' }}>
            <div style={{ fontSize: 12, fontWeight: 600, color, marginBottom: 4 }}>{title}</div>
            <div style={{ fontSize: 11, color: 'var(--ink-3)', lineHeight: 1.5 }}>{body}</div>
          </div>
        ))}
      </div>
    </motion.div>
  );
}
