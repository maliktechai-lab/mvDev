import { clsx } from "clsx";

interface FrameScore {
  frame: number;
  score: number;
  timestamp: number;
}

interface VideoTimelineProps {
  frameScores: FrameScore[];
  duration: number;
  suspiciousTimestamps: number[];
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function VideoTimeline({ frameScores, duration, suspiciousTimestamps }: VideoTimelineProps) {
  if (!frameScores.length) return null;

  return (
    <div className="card">
      <h3 className="mb-4 text-sm font-semibold text-gray-300">Frame-by-Frame Analysis</h3>

      {/* Timeline bar */}
      <div className="relative mb-3 flex h-12 items-end gap-px overflow-hidden rounded-lg bg-gray-800 px-1">
        {frameScores.map((fs) => {
          const heightPct = Math.max(4, fs.score * 100);
          const color =
            fs.score >= 0.7
              ? "bg-red-500"
              : fs.score >= 0.5
              ? "bg-yellow-500"
              : "bg-green-500";

          return (
            <div
              key={fs.frame}
              className={clsx("flex-1 rounded-t transition-all", color)}
              style={{ height: `${heightPct}%` }}
              title={`Frame ${fs.frame} @ ${formatTime(fs.timestamp)} — ${Math.round(fs.score * 100)}% fake`}
              role="presentation"
            />
          );
        })}
      </div>

      {/* Time labels */}
      <div className="flex justify-between text-xs text-gray-600">
        <span>0:00</span>
        <span>{formatTime(duration / 2)}</span>
        <span>{formatTime(duration)}</span>
      </div>

      {/* Legend */}
      <div className="mt-3 flex gap-4 text-xs">
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-green-500" />
          <span className="text-gray-400">Authentic (&lt;50%)</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-yellow-500" />
          <span className="text-gray-400">Uncertain (50-70%)</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-red-500" />
          <span className="text-gray-400">Suspicious (&gt;70%)</span>
        </span>
      </div>

      {/* Suspicious timestamps */}
      {suspiciousTimestamps.length > 0 && (
        <div className="mt-4 rounded-lg border border-red-700/30 bg-red-900/10 p-3">
          <p className="mb-2 text-xs font-semibold text-red-400">
            ⚠ Suspicious segments detected at:
          </p>
          <div className="flex flex-wrap gap-2">
            {suspiciousTimestamps.slice(0, 12).map((ts, i) => (
              <span
                key={i}
                className="rounded bg-red-900/30 px-2 py-0.5 font-mono text-xs text-red-300"
              >
                {formatTime(ts)}
              </span>
            ))}
            {suspiciousTimestamps.length > 12 && (
              <span className="text-xs text-gray-500">
                +{suspiciousTimestamps.length - 12} more
              </span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
