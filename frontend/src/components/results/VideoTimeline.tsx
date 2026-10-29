import { useState } from "react";
import { motion } from "framer-motion";
import { AlertTriangle, Film, EyeOff } from "lucide-react";

interface FrameScore {
  frame: number;
  /** null when no face was detected or face detector failed */
  score: number | null;
  timestamp: number;
  face_status?: "detected" | "not_found" | "failed";
  window_count?: number;
}

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

function faceStatusLabel(status: FrameScore["face_status"]): string {
  if (status === "not_found") return "No face detected — frame not scored";
  if (status === "failed")    return "Detector error — frame not scored";
  return "Face detected";
}

/**
 * Colour and height for a scored bar.
 * Unscored frames use a separate fixed appearance.
 */
function scoredBar(score: number) {
  const h     = Math.max(6, score * 100);
  const color = score >= 0.7 ? "#ff003c" : score >= 0.5 ? "#f97316" : "#00ff88";
  return { h, color };
}

export function VideoTimeline({ frameScores, duration, suspiciousTimestamps }: VideoTimelineProps) {
  const [hovered, setHovered] = useState<number | null>(null);
  if (!frameScores.length) return null;

  const scoredCount   = frameScores.filter(fs => fs.score !== null).length;
  const unscoredCount = frameScores.length - scoredCount;
  const notFoundCount = frameScores.filter(fs => fs.face_status === "not_found").length;
  const failedCount   = frameScores.filter(fs => fs.face_status === "failed").length;

  const hoveredFs = hovered !== null ? frameScores[hovered] : null;

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
        <span className="ml-auto text-xs text-gray-600">
          {scoredCount} scored · {frameScores.length} total · {fmt(duration)}
        </span>
      </div>

      {/* Unscored-frames notice */}
      {unscoredCount > 0 && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          className="mb-3 flex items-start gap-2 rounded-xl px-3 py-2 text-xs"
          style={{
            background: "rgba(148,163,184,0.06)",
            border: "1px solid rgba(148,163,184,0.15)",
          }}
        >
          <EyeOff className="mt-0.5 h-3.5 w-3.5 shrink-0 text-gray-500" />
          <span className="text-gray-500">
            {unscoredCount} frame{unscoredCount !== 1 ? "s" : ""} could not be scored
            {notFoundCount > 0 && ` (${notFoundCount} no face)`}
            {failedCount > 0   && ` (${failedCount} detector error)`}
            {". "}Shown as grey marks — they carry no inference about authentic or
            manipulated content.
          </span>
        </motion.div>
      )}

      {scoredCount === 0 ? (
        /* All frames unscored — no inference possible */
        <div
          className="flex items-center justify-center rounded-xl py-8 text-sm text-gray-600"
          style={{ background: "rgba(255,255,255,0.02)", border: "1px solid rgba(255,255,255,0.05)" }}
        >
          No frames could be scored — no faces were detected in this video.
        </div>
      ) : (
        <>
          {/*
            Timeline bar — ALL frames rendered in original order at their correct
            proportional positions.

            Scored frames  → coloured bar whose height encodes fake probability.
            Unscored frames → thin neutral grey stub (height fixed at 20%),
                              visually distinct from both authentic and suspicious scores.

            No probability is assigned to unscored frames.
          */}
          <div
            className="group relative mb-2 overflow-hidden rounded-xl"
            style={{ height: 64, background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.05)" }}
          >
            <div className="flex h-full items-end gap-px px-1">
              {frameScores.map((fs, i) => {
                const isHov    = hovered === i;
                const isScored = fs.score !== null;

                if (isScored) {
                  const { h, color } = scoredBar(fs.score as number);
                  return (
                    <motion.div
                      key={fs.frame}
                      className="flex-1 cursor-crosshair rounded-t transition-all"
                      style={{
                        height:     `${h}%`,
                        background: isHov ? "#ffffff" : color,
                        opacity:    isHov ? 1 : 0.7,
                        boxShadow:  isHov ? `0 0 8px ${color}` : "none",
                        minWidth:   1,
                      }}
                      initial={{ scaleY: 0, originY: 1 }}
                      animate={{ scaleY: 1 }}
                      transition={{ delay: i * 0.002, duration: 0.3 }}
                      onMouseEnter={() => setHovered(i)}
                      onMouseLeave={() => setHovered(null)}
                      title={`Frame ${fs.frame} @ ${fmt(fs.timestamp)} — ${Math.round((fs.score as number) * 100)}% (${faceStatusLabel(fs.face_status)})`}
                    />
                  );
                }

                // Unscored frame — neutral grey stub, distinct from any probability value
                return (
                  <motion.div
                    key={fs.frame}
                    className="flex-1 cursor-default rounded-t"
                    style={{
                      height:     "20%",
                      background: isHov ? "rgba(148,163,184,0.6)" : "rgba(148,163,184,0.2)",
                      minWidth:   1,
                      borderTop:  "1px dashed rgba(148,163,184,0.35)",
                    }}
                    initial={{ scaleY: 0, originY: 1 }}
                    animate={{ scaleY: 1 }}
                    transition={{ delay: i * 0.002, duration: 0.3 }}
                    onMouseEnter={() => setHovered(i)}
                    onMouseLeave={() => setHovered(null)}
                    title={`Frame ${fs.frame} @ ${fmt(fs.timestamp)} — ${faceStatusLabel(fs.face_status)}`}
                  />
                );
              })}
            </div>

            {/* Hover tooltip */}
            {hoveredFs && (
              <div
                className="pointer-events-none absolute top-1 rounded-lg px-2 py-1 text-xs font-mono"
                style={{
                  background: "rgba(10,15,30,0.95)",
                  border: "1px solid rgba(255,255,255,0.1)",
                  color: hoveredFs.score !== null
                    ? (hoveredFs.score >= 0.7 ? "#ff003c" : hoveredFs.score >= 0.5 ? "#f97316" : "#00ff88")
                    : "#94a3b8",
                  left: `${(hovered! / frameScores.length) * 85}%`,
                  transform: "translateX(-50%)",
                  zIndex: 10,
                }}
              >
                {fmt(hoveredFs.timestamp)}
                {hoveredFs.score !== null
                  ? ` — ${Math.round(hoveredFs.score * 100)}%`
                  : ` — ${faceStatusLabel(hoveredFs.face_status)}`}
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
          <div className="flex flex-wrap gap-4 text-xs">
            {[
              ["#00ff88", "Authentic (<50%)"],
              ["#f97316", "Uncertain (50–70%)"],
              ["#ff003c", "Suspicious (>70%)"],
            ].map(([c, l]) => (
              <div key={l} className="flex items-center gap-1.5">
                <div className="h-2.5 w-2.5 rounded-full" style={{ background: c, boxShadow: `0 0 4px ${c}` }} />
                <span style={{ color: "#475569" }}>{l}</span>
              </div>
            ))}
            {unscoredCount > 0 && (
              <div className="flex items-center gap-1.5">
                <div
                  className="h-2.5 w-2.5 rounded"
                  style={{ background: "rgba(148,163,184,0.3)", border: "1px dashed rgba(148,163,184,0.5)" }}
                />
                <span style={{ color: "#475569" }}>Not scored (no face / error)</span>
              </div>
            )}
          </div>
        </>
      )}

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
