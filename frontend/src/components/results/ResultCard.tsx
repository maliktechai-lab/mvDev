import { CheckCircle2, XCircle, HelpCircle, Eye } from "lucide-react";
import { clsx } from "clsx";

interface BranchScores {
  spatial?: number;
  frequency?: number;
  attention?: number;
  [key: string]: number | undefined;
}

interface DetectionResult {
  job_id: string;
  verdict: "real" | "fake" | "unknown";
  confidence: number;
  fake_probability: number;
  real_probability: number;
  faces_detected: number;
  branch_scores: BranchScores;
  heatmap_url: string | null;
  processing_ms: number;
}

interface ResultCardProps {
  result: DetectionResult;
  filename: string;
}

function ConfidenceMeter({ value, verdict }: { value: number; verdict: string }) {
  const pct = Math.round(value * 100);
  const color =
    verdict === "fake"
      ? "bg-red-500"
      : verdict === "real"
      ? "bg-green-500"
      : "bg-gray-500";

  return (
    <div>
      <div className="mb-1.5 flex justify-between text-sm">
        <span className="text-gray-400">Confidence</span>
        <span className="font-mono font-semibold text-white">{pct}%</span>
      </div>
      <div className="h-3 overflow-hidden rounded-full bg-gray-800">
        <div
          className={clsx("h-full rounded-full transition-all duration-700", color)}
          style={{ width: `${pct}%` }}
          role="progressbar"
          aria-valuenow={pct}
          aria-valuemin={0}
          aria-valuemax={100}
        />
      </div>
    </div>
  );
}

function BranchBar({
  label,
  value,
}: {
  label: string;
  value: number;
}) {
  const pct = Math.round(value * 100);
  const color =
    pct >= 70 ? "bg-red-500" : pct >= 50 ? "bg-yellow-500" : "bg-green-500";

  return (
    <div>
      <div className="mb-1 flex justify-between text-xs">
        <span className="capitalize text-gray-400">{label}</span>
        <span className="font-mono text-gray-300">{pct}%</span>
      </div>
      <div className="h-1.5 overflow-hidden rounded-full bg-gray-800">
        <div
          className={clsx("h-full rounded-full transition-all duration-500", color)}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

export function ResultCard({ result, filename }: ResultCardProps) {
  const verdictConfig = {
    fake:    { icon: XCircle,       color: "text-red-400",   bg: "bg-red-900/20 border-red-700/30",   label: "DEEPFAKE DETECTED" },
    real:    { icon: CheckCircle2,  color: "text-green-400", bg: "bg-green-900/20 border-green-700/30", label: "AUTHENTIC MEDIA" },
    unknown: { icon: HelpCircle,    color: "text-gray-400",  bg: "bg-gray-800 border-gray-700",        label: "INCONCLUSIVE" },
  }[result.verdict] ?? { icon: HelpCircle, color: "text-gray-400", bg: "bg-gray-800 border-gray-700", label: "INCONCLUSIVE" };

  const VerdictIcon = verdictConfig.icon;

  return (
    <div className="space-y-4">
      {/* Verdict banner */}
      <div className={clsx("rounded-xl border p-6", verdictConfig.bg)}>
        <div className="flex items-center gap-4">
          <VerdictIcon className={clsx("h-12 w-12", verdictConfig.color)} />
          <div>
            <p className={clsx("text-2xl font-bold", verdictConfig.color)}>
              {verdictConfig.label}
            </p>
            <p className="text-sm text-gray-400">
              {filename} · {result.faces_detected} face{result.faces_detected !== 1 ? "s" : ""} analyzed · {result.processing_ms}ms
            </p>
          </div>
        </div>

        {/* Probability bars */}
        <div className="mt-5 grid grid-cols-2 gap-4">
          <div className="rounded-lg bg-black/20 p-3 text-center">
            <p className="text-xs text-gray-400">Fake Probability</p>
            <p className="mt-1 font-mono text-2xl font-bold text-red-400">
              {Math.round(result.fake_probability * 100)}%
            </p>
          </div>
          <div className="rounded-lg bg-black/20 p-3 text-center">
            <p className="text-xs text-gray-400">Real Probability</p>
            <p className="mt-1 font-mono text-2xl font-bold text-green-400">
              {Math.round(result.real_probability * 100)}%
            </p>
          </div>
        </div>

        <div className="mt-4">
          <ConfidenceMeter value={result.confidence} verdict={result.verdict} />
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {/* Branch scores */}
        {Object.keys(result.branch_scores).length > 0 && (
          <div className="card">
            <h3 className="mb-4 text-sm font-semibold text-gray-300">Branch Analysis</h3>
            <div className="space-y-3">
              {Object.entries(result.branch_scores).map(([key, val]) =>
                val !== undefined ? (
                  <BranchBar key={key} label={key} value={val} />
                ) : null
              )}
            </div>
            <p className="mt-3 text-xs text-gray-600">
              Each branch independently analyzes a different signal domain.
              Higher = more likely fake.
            </p>
          </div>
        )}

        {/* Heatmap */}
        {result.heatmap_url && (
          <div className="card">
            <div className="mb-3 flex items-center gap-2">
              <Eye className="h-4 w-4 text-brand-400" />
              <h3 className="text-sm font-semibold text-gray-300">GradCAM Explanation</h3>
            </div>
            <img
              src={result.heatmap_url}
              alt="GradCAM heatmap showing suspicious regions"
              className="w-full rounded-lg object-contain"
            />
            <p className="mt-2 text-xs text-gray-600">
              Red regions indicate areas that most influenced the fake verdict.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
