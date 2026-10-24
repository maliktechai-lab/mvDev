import { useState } from "react";
import { motion } from "framer-motion";
import { AlertTriangle, Film } from "lucide-react";

interface FrameScore { frame: number; score: number; timestamp: number; }

interface VideoTimelineProps {
  frameScores: FrameScore[];
  duration: number;
  suspiciousTimestamps: number[];
}

function fmt(s: number) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

export function VideoTimeline({ frameScores, duration, suspiciousTimestamps }: VideoTimelineProps) {
  const [hovered, setHovered] = useState<number | null>(null);
  if (!frameScores.length) return null;

  const hoveredFrame = hovered !== null ? frameScores[hovered] : null;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.2 }}
      className="card-glow"
    >
      <div className="mb-4 flex items-center gap-2">
        <Film className="h-4 w-4" style={{ color: "#bf00ff" }} />
        <h3 className="text-sm font-bold text-gray-200">Frame-by-Frame Timeline</h3>
        <span className="ml-auto text-xs text-gray-600">{frameScores.length} frames · {fmt(duration)}</span>
      </div>

      {/* Timeline bar */}
      <div
        className="group relative mb-2 overflow-hidden rounded-xl"
        style={{ height: 64, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.05)" }}
      >
        <div className="flex h-full items-end gap-px px-1">
          {frameScores.map((fs, i) => {
            const h    = Math.max(6, fs.score * 100);
            const color = fs.score >= 0.7 ? "#ff003c" : fs.score >= 0.5 ? "#f97316" : "#00ff88";
            const isHov = hovered === i;
            return (
              <motion.div
                key={fs.frame}
                className="flex-1 cursor-crosshair rounded-t transition-all"
                style={{
                  height:   `${h}%`,
                  background: isHov ? "#ffffff" : color,
                  opacity: isHov ? 1 : 0.7,
                  boxShadow: isHov ? `0 0 8px ${color}` : "none",
                  minWidth: 1,
                }}
                initial={{ scaleY: 0, originY: 1 }}
                animate={{ scaleY: 1 }}
                transition={{ delay: i * 0.002, duration: 0.3 }}
                onMouseEnter={() => setHovered(i)}
                onMouseLeave={() => setHovered(null)}
                title={`Frame ${fs.frame} @ ${fmt(fs.timestamp)} — ${Math.round(fs.score * 100)}%`}
              />
            );
          })}
        </div>

        {/* Hover tooltip */}
        {hoveredFrame && (
          <div
            className="pointer-events-none absolute top-1 rounded-lg px-2 py-1 text-xs font-mono"
            style={{
              background: "rgba(10,15,30,0.95)",
              border: "1px solid rgba(255,255,255,0.1)",
              color: hoveredFrame.score >= 0.7 ? "#ff003c" : hoveredFrame.score >= 0.5 ? "#f97316" : "#00ff88",
              left: `${(hovered! / frameScores.length) * 85}%`,
              transform: "translateX(-50%)",
              zIndex: 10,
            }}
          >
            {fmt(hoveredFrame.timestamp)} — {Math.round(hoveredFrame.score * 100)}%
          </div>
        )}
      </div>

      {/* Time axis */}
      <div className="mb-4 flex justify-between text-xs text-gray-700">
        <span>0:00</span>
        <span>{fmt(duration / 4)}</span>
        <span>{fmt(duration / 2)}</span>
        <span>{fmt(duration * 0.75)}</span>
        <span>{fmt(duration)}</span>
      </div>

      {/* Legend */}
      <div className="flex gap-5 text-xs">
        {[["#00ff88", "Authentic (<50%)"], ["#f97316", "Uncertain (50–70%)"], ["#ff003c", "Suspicious (>70%)"]].map(([c, l]) => (
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
            <p className="text-xs font-semibold text-red-400">Suspicious segments detected</p>
          </div>
          <div className="flex flex-wrap gap-2">
            {suspiciousTimestamps.slice(0, 12).map((ts, i) => (
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
            {suspiciousTimestamps.length > 12 && (
              <span className="text-xs text-gray-600">+{suspiciousTimestamps.length - 12} more</span>
            )}
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
