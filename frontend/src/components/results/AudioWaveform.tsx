import { clsx } from "clsx";

interface ChunkScore {
  chunk: number;
  score: number;
  start: number;
  end: number;
}

interface AudioWaveformProps {
  chunkScores: ChunkScore[];
  duration: number;
  suspiciousTimestamps: number[];
}

function formatTime(seconds: number): string {
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, "0")}`;
}

export function AudioWaveform({ chunkScores, duration, suspiciousTimestamps }: AudioWaveformProps) {
  if (!chunkScores.length) return null;

  return (
    <div className="card">
      <h3 className="mb-4 text-sm font-semibold text-gray-300">Segment-by-Segment Analysis</h3>

      {/* Segment bars */}
      <div className="flex items-end gap-1 overflow-hidden rounded-lg bg-gray-800 px-2 py-2" style={{ height: 64 }}>
        {chunkScores.map((cs) => {
          const heightPct = Math.max(8, cs.score * 100);
          const color =
            cs.score >= 0.7 ? "bg-red-500" :
            cs.score >= 0.5 ? "bg-yellow-500" : "bg-green-500";
          return (
            <div
              key={cs.chunk}
              className={clsx("flex-1 rounded-t transition-all", color)}
              style={{ height: `${heightPct}%` }}
              title={`${formatTime(cs.start)}–${formatTime(cs.end)} — ${Math.round(cs.score * 100)}% synthetic`}
              role="presentation"
            />
          );
        })}
      </div>

      {/* Labels */}
      <div className="mt-2 flex justify-between text-xs text-gray-600">
        <span>0:00</span>
        <span>{formatTime(duration / 2)}</span>
        <span>{formatTime(duration)}</span>
      </div>

      {/* Legend */}
      <div className="mt-3 flex gap-4 text-xs">
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-green-500" /><span className="text-gray-400">Natural</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-yellow-500" /><span className="text-gray-400">Uncertain</span>
        </span>
        <span className="flex items-center gap-1.5">
          <span className="h-2.5 w-2.5 rounded-full bg-red-500" /><span className="text-gray-400">Synthetic</span>
        </span>
      </div>

      {/* Suspicious segments */}
      {suspiciousTimestamps.length > 0 && (
        <div className="mt-4 rounded-lg border border-red-700/30 bg-red-900/10 p-3">
          <p className="mb-2 text-xs font-semibold text-red-400">⚠ Synthetic segments at:</p>
          <div className="flex flex-wrap gap-2">
            {suspiciousTimestamps.slice(0, 10).map((ts, i) => (
              <span key={i} className="rounded bg-red-900/30 px-2 py-0.5 font-mono text-xs text-red-300">
                {formatTime(ts)}
              </span>
            ))}
            {suspiciousTimestamps.length > 10 && (
              <span className="text-xs text-gray-500">+{suspiciousTimestamps.length - 10} more</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}
