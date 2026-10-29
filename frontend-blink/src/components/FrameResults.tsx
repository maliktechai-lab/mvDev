import { motion } from 'framer-motion';
import { AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer, ReferenceLine } from 'recharts';
import type { FrameResult } from '../types';

interface Props {
  frames: FrameResult[];
}

export default function FrameResults({ frames }: Props) {
  const chartData = frames.map((f) => ({
    frame: `F${f.frameIndex + 1}`,
    // Null scores are rendered as gaps in the chart (undefined → no point drawn)
    spatial: f.spatialScore !== null ? Math.round(f.spatialScore * 100) : null,
    frequency: f.frequencyScore !== null ? Math.round(f.frequencyScore * 100) : null,
    fused: f.fusedScore !== null ? Math.round(f.fusedScore * 100) : null,
  }));

  // Count only scored frames for fake/real stats — unscored (UNKNOWN) are separate
  const scoredFrames = frames.filter(f => f.label !== 'UNKNOWN');
  const fakeCount = scoredFrames.filter(f => f.label === 'FAKE').length;
  const realCount = scoredFrames.filter(f => f.label === 'REAL').length;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4 }}
      className="glass-card p-6 space-y-5"
    >
      <div className="flex items-center justify-between flex-wrap gap-2">
        <div className="flex items-center gap-2">
          <span className="text-xl">🎬</span>
          <h3 className="font-bold text-lg" style={{ color: '#f0f4ff' }}>Frame-Level Analysis</h3>
        </div>
        <div className="flex gap-3 text-xs">
          <span className="px-2 py-1 rounded-full" style={{ background: 'rgba(239,68,68,0.12)', color: '#ef4444', border: '1px solid rgba(239,68,68,0.3)' }}>
            {fakeCount} Fake
          </span>
          <span className="px-2 py-1 rounded-full" style={{ background: 'rgba(16,185,129,0.12)', color: '#10b981', border: '1px solid rgba(16,185,129,0.3)' }}>
            {realCount} Real
          </span>
        </div>
      </div>

      {/* Chart */}
      <div className="h-48">
        <ResponsiveContainer width="100%" height="100%">
          <AreaChart data={chartData} margin={{ top: 5, right: 5, bottom: 0, left: -20 }}>
            <defs>
              <linearGradient id="colorSpatial" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#00d4ff" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#00d4ff" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="colorFreq" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#7c3aed" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#7c3aed" stopOpacity={0} />
              </linearGradient>
              <linearGradient id="colorFused" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%" stopColor="#ec4899" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#ec4899" stopOpacity={0} />
              </linearGradient>
            </defs>
            <XAxis dataKey="frame" tick={{ fill: '#8b9fc5', fontSize: 11 }} axisLine={false} tickLine={false} />
            <YAxis tick={{ fill: '#8b9fc5', fontSize: 10 }} axisLine={false} tickLine={false} domain={[0, 100]} />
            <Tooltip
              contentStyle={{
                background: '#0a1628', border: '1px solid rgba(0,212,255,0.3)',
                borderRadius: '8px', color: '#f0f4ff', fontSize: 12,
              }}
              formatter={(value) => [`${value}%`, ''] as [string, string]}
            />
            <ReferenceLine y={50} stroke="rgba(255,255,255,0.15)" strokeDasharray="4 4" />
            <Area type="monotone" dataKey="spatial" stroke="#00d4ff" strokeWidth={2} fill="url(#colorSpatial)" name="Spatial" />
            <Area type="monotone" dataKey="frequency" stroke="#7c3aed" strokeWidth={2} fill="url(#colorFreq)" name="Frequency" />
            <Area type="monotone" dataKey="fused" stroke="#ec4899" strokeWidth={2} fill="url(#colorFused)" name="Fused" />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Legend */}
      <div className="flex flex-wrap gap-4 text-xs">
        {[{ color: '#00d4ff', label: 'Spatial' }, { color: '#7c3aed', label: 'Frequency' }, { color: '#ec4899', label: 'Fused' }].map(({ color, label }) => (
          <div key={label} className="flex items-center gap-1.5">
            <div className="w-3 h-1.5 rounded-full" style={{ background: color }} />
            <span style={{ color: '#8b9fc5' }}>{label}</span>
          </div>
        ))}
        <div className="flex items-center gap-1.5 ml-auto">
          <div className="w-8 border-t border-dashed" style={{ borderColor: 'rgba(255,255,255,0.3)' }} />
          <span style={{ color: '#8b9fc5' }}>50% threshold</span>
        </div>
      </div>

      {/* Frame grid */}
      <div className="grid grid-cols-4 gap-2">
        {frames.map((frame, i) => {
          const isFakeFrame = frame.label === 'FAKE';
          const isUnknown = frame.label === 'UNKNOWN';
          const colors = isUnknown
            ? { bg: 'rgba(245,158,11,0.06)', border: 'rgba(245,158,11,0.2)', text: '#f59e0b' }
            : isFakeFrame
            ? { bg: 'rgba(239,68,68,0.08)', border: 'rgba(239,68,68,0.3)', text: '#ef4444' }
            : { bg: 'rgba(16,185,129,0.08)', border: 'rgba(16,185,129,0.3)', text: '#10b981' };
          return (
            <motion.div
              key={frame.frameIndex}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ delay: 0.1 * i }}
              className="rounded-lg p-2 text-center space-y-1"
              style={{
                background: colors.bg,
                border: `1px solid ${colors.border}`,
              }}
            >
              <div className="text-xs font-mono" style={{ color: '#8b9fc5' }}>F{frame.frameIndex + 1}</div>
              <div className="text-xs font-bold" style={{ color: colors.text }}>
                {frame.label}
              </div>
              <div className="text-xs font-mono" style={{ color: '#8b9fc5' }}>
                {frame.fusedScore !== null ? `${Math.round(frame.fusedScore * 100)}%` : '—'}
              </div>
            </motion.div>
          );
        })}
      </div>
    </motion.div>
  );
}
