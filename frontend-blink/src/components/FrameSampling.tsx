import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import type { FrameResult } from '../types';

interface Props {
  frames: FrameResult[];
}

const COLOR_MAP = {
  FAKE: { bg: 'rgba(239,68,68,0.1)', border: 'rgba(239,68,68,0.35)', text: '#ef4444', badge: 'rgba(239,68,68,0.15)' },
  REAL: { bg: 'rgba(16,185,129,0.1)', border: 'rgba(16,185,129,0.35)', text: '#10b981', badge: 'rgba(16,185,129,0.15)' },
  UNKNOWN: { bg: 'rgba(245,158,11,0.06)', border: 'rgba(245,158,11,0.25)', text: '#f59e0b', badge: 'rgba(245,158,11,0.1)' },
};

export default function FrameSampling({ frames }: Props) {
  const [selected, setSelected] = useState<number | null>(null);
  const [filter, setFilter] = useState<'all' | 'FAKE' | 'REAL'>('all');

  const fakeCount = frames.filter(f => f.label === 'FAKE').length;
  const realCount = frames.filter(f => f.label === 'REAL').length;
  // Average only over scored frames — null scores must not be treated as 0
  const scoredFrames = frames.filter(f => f.fusedScore !== null);
  const avgFused = scoredFrames.length > 0
    ? scoredFrames.reduce((s, f) => s + (f.fusedScore ?? 0), 0) / scoredFrames.length
    : 0;
  const filtered = filter === 'all' ? frames : frames.filter(f => f.label === filter);
  const selectedFrame = selected !== null ? frames.find(f => f.frameIndex === selected) ?? null : null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.4 }}
      className="glass-card p-6 space-y-5"
    >
      <div className="flex items-center justify-between flex-wrap gap-3">
        <div className="flex items-center gap-2">
          <span className="text-xl">🎞️</span>
          <div>
            <h3 className="font-bold text-base" style={{ color: '#f0f4ff' }}>Frame Sampling</h3>
            <p className="text-xs" style={{ color: '#8b9fc5' }}>{frames.length} frames sampled · uniform interval</p>
          </div>
        </div>
        <div className="flex gap-1 p-1 rounded-lg" style={{ background: 'rgba(255,255,255,0.04)' }}>
          {(['all', 'FAKE', 'REAL'] as const).map(tab => (
            <button
              key={tab}
              onClick={() => setFilter(tab)}
              className="px-3 py-1.5 rounded-md text-xs font-semibold transition-all duration-150"
              style={{
                background: filter === tab
                  ? tab === 'FAKE' ? 'rgba(239,68,68,0.2)' : tab === 'REAL' ? 'rgba(16,185,129,0.2)' : 'rgba(255,255,255,0.08)'
                  : 'transparent',
                color: filter === tab
                  ? tab === 'FAKE' ? '#ef4444' : tab === 'REAL' ? '#10b981' : '#f0f4ff'
                  : '#8b9fc5',
              }}
            >
              {tab === 'all' ? `All (${frames.length})` : tab === 'FAKE' ? `Fake (${fakeCount})` : `Real (${realCount})`}
            </button>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3">
        {[
          { label: 'Fake Frames', value: `${fakeCount}/${frames.length}`, color: '#ef4444' },
          { label: 'Real Frames', value: `${realCount}/${frames.length}`, color: '#10b981' },
          { label: 'Avg. Score', value: scoredFrames.length > 0 ? `${Math.round(avgFused * 100)}%` : 'N/A', color: '#00d4ff' },
        ].map(({ label, value, color }) => (
          <div key={label} className="rounded-xl p-3 text-center space-y-1" style={{ background: 'rgba(255,255,255,0.03)', border: '1px solid rgba(255,255,255,0.06)' }}>
            <div className="text-lg font-black" style={{ color }}>{value}</div>
            <div className="text-xs" style={{ color: '#8b9fc5' }}>{label}</div>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-4 gap-2">
        <AnimatePresence mode="popLayout">
          {filtered.map((frame, i) => {
            const isUnknown = frame.label === 'UNKNOWN';
            const colors = isUnknown
              ? { bg: 'rgba(245,158,11,0.06)', border: 'rgba(245,158,11,0.2)', text: '#f59e0b', badge: 'rgba(245,158,11,0.1)' }
              : frame.label === 'FAKE'
              ? { bg: 'rgba(239,68,68,0.1)', border: 'rgba(239,68,68,0.35)', text: '#ef4444', badge: 'rgba(239,68,68,0.15)' }
              : { bg: 'rgba(16,185,129,0.1)', border: 'rgba(16,185,129,0.35)', text: '#10b981', badge: 'rgba(16,185,129,0.15)' };
            const isSelected = selected === frame.frameIndex;
            return (
              <motion.button
                key={frame.frameIndex}
                layout
                initial={{ opacity: 0, scale: 0.85 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.85 }}
                transition={{ delay: i * 0.04 }}
                onClick={() => setSelected(isSelected ? null : frame.frameIndex)}
                className="rounded-xl p-3 text-center space-y-2"
                style={{
                  background: isSelected ? colors.bg : 'rgba(255,255,255,0.03)',
                  border: `1px solid ${isSelected ? colors.border : 'rgba(255,255,255,0.08)'}`,
                  cursor: 'pointer',
                }}
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.97 }}
              >
                <div className="w-full aspect-video rounded-lg flex items-center justify-center text-xs font-mono" style={{ background: `${colors.text}10` }}>
                  <span style={{ color: colors.text }}>F{frame.frameIndex + 1}</span>
                </div>
                <div className="text-xs font-mono" style={{ color: '#8b9fc5' }}>{frame.timestamp.toFixed(1)}s</div>
                <div className="text-xs font-bold py-0.5 px-1.5 rounded-md" style={{ background: colors.badge, color: colors.text }}>{frame.label}</div>
                <div className="text-xs font-mono font-bold" style={{ color: colors.text }}>
                  {frame.fusedScore !== null ? `${Math.round(frame.fusedScore * 100)}%` : '—'}
                </div>
              </motion.button>
            );
          })}
        </AnimatePresence>
      </div>

      <AnimatePresence>
        {selectedFrame && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="overflow-hidden"
          >
            <div
              className="rounded-xl p-4 space-y-3"
              style={{ background: COLOR_MAP[selectedFrame.label].bg, border: `1px solid ${COLOR_MAP[selectedFrame.label].border}` }}
            >
              <div className="flex items-center justify-between">
                <span className="font-semibold text-sm" style={{ color: '#f0f4ff' }}>Frame {selectedFrame.frameIndex + 1} · {selectedFrame.timestamp.toFixed(2)}s</span>
                <span className="text-xs font-bold px-2.5 py-1 rounded-full" style={{ background: COLOR_MAP[selectedFrame.label].badge, color: COLOR_MAP[selectedFrame.label].text, border: `1px solid ${COLOR_MAP[selectedFrame.label].border}` }}>
                  {selectedFrame.label}
                </span>
              </div>
              <div className="grid grid-cols-3 gap-3 text-xs font-mono">
                {[
                  { label: 'Spatial', value: selectedFrame.spatialScore, color: '#00d4ff' },
                  { label: 'Frequency', value: selectedFrame.frequencyScore, color: '#7c3aed' },
                  { label: 'Fused', value: selectedFrame.fusedScore, color: COLOR_MAP[selectedFrame.label as 'FAKE' | 'REAL' | 'UNKNOWN'].text },
                ].map(({ label, value, color }) => (
                  <div key={label} className="space-y-1.5">
                    <div className="flex justify-between">
                      <span style={{ color: '#8b9fc5' }}>{label}</span>
                      <span style={{ color }}>{value !== null ? `${Math.round(value * 100)}%` : '—'}</span>
                    </div>
                    <div className="h-1.5 rounded-full" style={{ background: 'rgba(255,255,255,0.06)' }}>
                      <div className="h-full rounded-full" style={{ width: `${value !== null ? value * 100 : 0}%`, background: `linear-gradient(90deg, ${color}55, ${color})`, transition: 'width 0.5s ease' }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      <p className="text-xs" style={{ color: '#4a5568' }}>⚠️ Frame sampling uses uniform intervals. Brief manipulations between sampled frames may not be detected.</p>
    </motion.div>
  );
}
