import { useState } from "react";
import { motion } from "framer-motion";
import { AlertTriangle, Music2 } from "lucide-react";

interface ChunkScore { chunk: number; score: number; start: number; end: number; }

interface AudioWaveformProps {
  chunkScores: ChunkScore[];
  duration: number;
  suspiciousTimestamps: number[];
}

function fmt(s: number) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

export function AudioWaveform({ chunkScores, duration, suspiciousTimestamps }: AudioWaveformProps) {
  const [hovered, setHovered] = useState<number | null>(null);
  if (!chunkScores.length) return null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      className="card-glow"
    >
      <div className="mb-4 flex items-center gap-2">
        <Music2 className="h-4 w-4" style={{ color: "#00f5ff" }} />
        <h3 className="text-sm font-bold text-gray-200">Segment-by-Segment Waveform</h3>
        <span className="ml-auto text-xs text-gray-600">{chunkScores.length} segments · {fmt(duration)}</span>
      </div>

      {/* Waveform visualization — symmetric bars */}
      <div
        className="relative mb-2 flex items-center justify-center gap-px overflow-hidden rounded-xl"
        style={{ height: 80, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.05)" }}
      >
        {chunkScores.map((cs, i) => {
          const h      = Math.max(8, cs.score * 90);
          const color  = cs.score >= 0.7 ? "#ff003c" : cs.score >= 0.5 ? "#f97316" : "#00f5ff";
          const isHov  = hovered === i;
          return (
            <motion.div
              key={cs.chunk}
              className="relative flex cursor-crosshair flex-col items-center justify-center gap-px"
              style={{ flex: 1, height: "100%" }}
              onMouseEnter={() => setHovered(i)}
              onMouseLeave={() => setHovered(null)}
            >
              {/* Top half bar */}
              <motion.div
                style={{
                  width: "70%",
                  height: `${h / 2}%`,
                  background: isHov ? "#ffffff" : color,
                  borderRadius: "2px 2px 0 0",
                  opacity: isHov ? 1 : 0.8,
                  boxShadow: isHov ? `0 0 8px ${color}` : "none",
                  minWidth: 1,
                }}
                initial={{ scaleY: 0 }}
                animate={{ scaleY: 1 }}
                transition={{ delay: i * 0.005, duration: 0.4 }}
              />
              {/* Center line */}
              <div style={{ height: 1, width: "100%", background: "rgba(255,255,255,0.05)" }} />
              {/* Bottom half bar (mirrored) */}
              <motion.div
                style={{
                  width: "70%",
                  height: `${h / 2}%`,
                  background: isHov ? "#ffffff" : color,
                  borderRadius: "0 0 2px 2px",
                  opacity: isHov ? 1 : 0.5,
                  minWidth: 1,
                }}
                initial={{ scaleY: 0 }}
                animate={{ scaleY: 1 }}
                transition={{ delay: i * 0.005, duration: 0.4 }}
              />

              {/* Hover tooltip */}
              {isHov && (
                <div
                  className="pointer-events-none absolute top-1 rounded px-2 py-0.5 text-xs font-mono"
                  style={{
                    background: "rgba(10,15,30,0.95)",
                    border: "1px solid rgba(255,255,255,0.1)",
                    color,
                    zIndex: 10,
                    whiteSpace: "nowrap",
                  }}
                >
                  {fmt(cs.start)}–{fmt(cs.end)} · {Math.round(cs.score * 100)}%
                </div>
              )}
            </motion.div>
          );
        })}
      </div>

      {/* Time axis */}
      <div className="mb-4 flex justify-between text-xs text-gray-700">
        <span>0:00</span>
        <span>{fmt(duration / 2)}</span>
        <span>{fmt(duration)}</span>
      </div>

      {/* Legend */}
      <div className="flex gap-5 text-xs">
        {[["#00f5ff", "Natural (<50%)"], ["#f97316", "Uncertain (50–70%)"], ["#ff003c", "Synthetic (>70%)"]].map(([c, l]) => (
          <div key={l} className="flex items-center gap-1.5">
            <div className="h-2.5 w-2.5 rounded-full" style={{ background: c, boxShadow: `0 0 4px ${c}` }} />
            <span style={{ color: "#475569" }}>{l}</span>
          </div>
        ))}
      </div>

      {/* Suspicious segments */}
      {suspiciousTimestamps.length > 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="mt-4 rounded-xl p-3"
          style={{ background: "rgba(255,0,60,0.06)", border: "1px solid rgba(255,0,60,0.2)" }}
        >
          <div className="mb-2 flex items-center gap-2">
            <AlertTriangle className="h-3.5 w-3.5 text-red-400" />
            <p className="text-xs font-semibold text-red-400">Synthetic segments detected</p>
          </div>
          <div className="flex flex-wrap gap-2">
            {suspiciousTimestamps.slice(0, 10).map((ts, i) => (
              <motion.span
                key={i}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                transition={{ delay: i * 0.04, type: "spring" }}
                className="rounded-lg px-2.5 py-1 font-mono text-xs text-red-300"
                style={{ background: "rgba(255,0,60,0.12)", border: "1px solid rgba(255,0,60,0.2)" }}
              >
                {fmt(ts)}
              </motion.span>
            ))}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
