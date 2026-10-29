import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { CircularProgressbar, buildStyles } from "react-circular-progressbar";
import "react-circular-progressbar/dist/styles.css";
import {
  XCircle, CheckCircle2, HelpCircle, Eye,
  Zap, Clock, Users, AlertTriangle, Shield,
} from "lucide-react";

interface BranchScores {
  spatial?: number;
  frequency?: number;
  attention?: number;
  temporal?: number;
  biological?: number;
  raw_waveform?: number;
  spectral?: number;
  prosody?: number;
  [key: string]: number | undefined;
}

export interface ResultCardData {
  job_id: string;
  verdict: "real" | "fake" | "unknown";
  confidence: number;
  fake_probability: number;
  real_probability: number;
  faces_detected: number;
  branch_scores: BranchScores;
  heatmap_url: string | null;
  processing_ms: number;
  /** True when backend is running in dev mode with random/untrained weights */
  random_weights_warning?: boolean;
}

interface ResultCardProps {
  result: ResultCardData;
  filename: string;
}

// ── Animated counter ─────────────────────────────────────────────────────
function AnimatedNumber({ value, suffix = "%" }: { value: number; suffix?: string }) {
  const [display, setDisplay] = useState(0);
  useEffect(() => {
    let start = 0;
    const end  = value;
    const dur  = 1200;
    const step = 16;
    const inc  = (end / (dur / step));
    const timer = setInterval(() => {
      start += inc;
      if (start >= end) { setDisplay(end); clearInterval(timer); }
      else               { setDisplay(Math.round(start)); }
    }, step);
    return () => clearInterval(timer);
  }, [value]);
  return <>{display}{suffix}</>;
}

// ── Branch score bar ─────────────────────────────────────────────────────
const BRANCH_META: Record<string, { label: string; icon: string; desc: string }> = {
  spatial:      { label: "Spatial",      icon: "🔬", desc: "Texture + pixel-level artifacts" },
  frequency:    { label: "Frequency",    icon: "📡", desc: "DCT / FFT spectrum analysis" },
  attention:    { label: "Attention",    icon: "👁️",  desc: "GradCAM saliency regions" },
  temporal:     { label: "Temporal",     icon: "⏱️",  desc: "LSTM frame sequence" },
  biological:   { label: "Biological",   icon: "💓", desc: "rPPG heartbeat signal" },
  raw_waveform: { label: "Raw Waveform", icon: "〰️", desc: "RawNet2 end-to-end" },
  spectral:     { label: "Spectral",     icon: "🎵", desc: "Mel-spectrogram artifacts" },
  prosody:      { label: "Prosody",      icon: "🗣️",  desc: "Pitch & rhythm patterns" },
};

function BranchBar({ name, value }: { name: string; value: number }) {
  const pct  = Math.round(value * 100);
  const meta = BRANCH_META[name] ?? { label: name, icon: "⚡", desc: "" };
  const color = pct >= 70 ? "#ff003c" : pct >= 50 ? "#f97316" : "#00ff88";
  const bgColor = pct >= 70 ? "rgba(255,0,60,0.08)" : pct >= 50 ? "rgba(249,115,22,0.08)" : "rgba(0,255,136,0.06)";
  const borderColor = pct >= 70 ? "rgba(255,0,60,0.2)" : pct >= 50 ? "rgba(249,115,22,0.2)" : "rgba(0,255,136,0.15)";

  return (
    <motion.div
      initial={{ opacity: 0, x: -15 }}
      animate={{ opacity: 1, x: 0 }}
      className="rounded-xl px-4 py-3"
      style={{ background: bgColor, border: `1px solid ${borderColor}` }}
    >
      <div className="mb-2 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span className="text-base">{meta.icon}</span>
          <div>
            <p className="text-sm font-semibold text-gray-200">{meta.label}</p>
            <p className="text-xs text-gray-600">{meta.desc}</p>
          </div>
        </div>
        <span className="font-mono text-lg font-bold" style={{ color }}>
          <AnimatedNumber value={pct} />
        </span>
      </div>
      <div className="progress-bar">
        <motion.div
          className="progress-fill"
          initial={{ width: 0 }}
          animate={{ width: `${pct}%` }}
          transition={{ duration: 1, delay: 0.3, ease: [0.16, 1, 0.3, 1] }}
          style={{ background: `linear-gradient(90deg, ${color}99, ${color})` }}
        />
      </div>
    </motion.div>
  );
}

// ── Main component ───────────────────────────────────────────────────────
export function ResultCard({ result, filename }: ResultCardProps) {
  const isFake    = result.verdict === "fake";
  const isReal    = result.verdict === "real";
  const pct       = Math.round(result.confidence * 100);
  const fakePct   = Math.round(result.fake_probability * 100);
  const realPct   = Math.round(result.real_probability * 100);

  const verdictConfig = {
    fake: {
      label:      "DEEPFAKE DETECTED",
      sublabel:   "This media has been identified as synthetically generated",
      Icon:       XCircle,
      color:      "#ff003c",
      bgClass:    "card-fake",
      ringColor:  "#ff003c",
      ringTrail:  "rgba(255,0,60,0.1)",
      glowStyle:  { boxShadow: "0 0 60px rgba(255,0,60,0.3), 0 0 120px rgba(255,0,60,0.1)" },
      textClass:  "gradient-text-fake",
    },
    real: {
      label:      "AUTHENTIC MEDIA",
      sublabel:   "This media appears to be genuine and unmanipulated",
      Icon:       CheckCircle2,
      color:      "#00ff88",
      bgClass:    "card-real",
      ringColor:  "#00ff88",
      ringTrail:  "rgba(0,255,136,0.1)",
      glowStyle:  { boxShadow: "0 0 60px rgba(0,255,136,0.2), 0 0 120px rgba(0,255,136,0.08)" },
      textClass:  "gradient-text-real",
    },
    unknown: {
      label:      "INCONCLUSIVE",
      sublabel:   "Insufficient confidence to make a determination",
      Icon:       HelpCircle,
      color:      "#94a3b8",
      bgClass:    "card",
      ringColor:  "#94a3b8",
      ringTrail:  "rgba(148,163,184,0.1)",
      glowStyle:  {},
      textClass:  "text-gray-400",
    },
  }[result.verdict] ?? {
    label: "INCONCLUSIVE", sublabel: "", Icon: HelpCircle,
    color: "#94a3b8", bgClass: "card", ringColor: "#94a3b8",
    ringTrail: "rgba(148,163,184,0.1)", glowStyle: {}, textClass: "text-gray-400",
  };

  const { Icon, color, bgClass, ringTrail, glowStyle, textClass, label, sublabel } = verdictConfig;
  const branchEntries = Object.entries(result.branch_scores).filter(([, v]) => v !== undefined) as [string, number][];

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      className="space-y-5"
    >
      {/* ── Dev-mode random weights warning ────────────────────────────── */}
      {result.random_weights_warning && (
        <div
          className="flex gap-3 rounded-xl px-4 py-3 text-sm"
          style={{
            background: "rgba(234, 179, 8, 0.08)",
            border: "1px solid rgba(234, 179, 8, 0.3)",
          }}
          role="alert"
        >
          <AlertTriangle className="mt-0.5 h-4 w-4 shrink-0 text-yellow-400" />
          <div>
            <p className="font-semibold text-yellow-400">Development mode — random weights</p>
            <p className="mt-0.5 text-xs text-slate-400">
              This result was generated by an untrained model. The scores below are meaningless and
              must not be interpreted as a real deepfake detection. Train the model first.
            </p>
          </div>
        </div>
      )}

      {/* ── Verdict banner ─────────────────────────────────────────────── */}
      <div className={`${bgClass} relative overflow-hidden`} style={glowStyle}>
        {/* Animated background glow */}
        <motion.div
          className="absolute inset-0 pointer-events-none"
          animate={{ opacity: [0.3, 0.6, 0.3] }}
          transition={{ duration: 3, repeat: Infinity }}
          style={{
            background: `radial-gradient(ellipse at 30% 50%, ${color}18 0%, transparent 60%)`,
          }}
        />

        {/* Ripple effect on reveal */}
        <AnimatePresence>
          {[0, 1].map((i) => (
            <motion.div
              key={i}
              className="absolute rounded-full pointer-events-none"
              style={{
                top: "50%", left: "10%",
                width: 100, height: 100,
                marginTop: -50, marginLeft: -50,
                border: `2px solid ${color}`,
              }}
              initial={{ scale: 0, opacity: 0.8 }}
              animate={{ scale: 4, opacity: 0 }}
              transition={{ duration: 1.5, delay: i * 0.5 }}
            />
          ))}
        </AnimatePresence>

        <div className="relative flex flex-col gap-6 sm:flex-row sm:items-center">
          {/* Circular confidence meter */}
          <motion.div
            initial={{ scale: 0, rotate: -180 }}
            animate={{ scale: 1, rotate: 0 }}
            transition={{ duration: 0.8, type: "spring", stiffness: 200 }}
            className="mx-auto shrink-0 sm:mx-0"
            style={{ width: 110, height: 110 }}
          >
            <CircularProgressbar
              value={pct}
              text={`${pct}%`}
              styles={buildStyles({
                textSize:            "20px",
                textColor:           color,
                pathColor:           color,
                trailColor:          ringTrail,
                pathTransitionDuration: 1.5,
              })}
            />
          </motion.div>

          {/* Verdict text */}
          <div className="flex-1 text-center sm:text-left">
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 0.3, duration: 0.5 }}
              className="flex items-center justify-center gap-3 sm:justify-start"
            >
              <motion.div
                animate={{
                  filter: isFake
                    ? ["drop-shadow(0 0 6px #ff003c)", "drop-shadow(0 0 18px #ff003c)", "drop-shadow(0 0 6px #ff003c)"]
                    : isReal
                    ? ["drop-shadow(0 0 6px #00ff88)", "drop-shadow(0 0 18px #00ff88)", "drop-shadow(0 0 6px #00ff88)"]
                    : [],
                }}
                transition={{ duration: 2, repeat: Infinity }}
              >
                <Icon className="h-8 w-8" style={{ color }} />
              </motion.div>
              <h2 className={`text-2xl font-black tracking-wider ${textClass}`}>{label}</h2>
            </motion.div>

            <motion.p
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={{ delay: 0.5 }}
              className="mt-1 text-sm text-gray-500"
            >
              {sublabel}
            </motion.p>

            {/* Meta info chips */}
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
              className="mt-4 flex flex-wrap items-center justify-center gap-3 sm:justify-start"
            >
              {[
                { icon: Shield,    text: filename.length > 24 ? filename.slice(0, 24) + "…" : filename },
                { icon: Users,     text: `${result.faces_detected} face${result.faces_detected !== 1 ? "s" : ""} detected` },
                { icon: Clock,     text: `${result.processing_ms}ms inference` },
              ].map(({ icon: I, text }) => (
                <div
                  key={text}
                  className="flex items-center gap-1.5 rounded-full px-3 py-1 text-xs text-gray-400"
                  style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.07)" }}
                >
                  <I className="h-3 w-3" />
                  {text}
                </div>
              ))}
            </motion.div>
          </div>
        </div>

        {/* Probability dials — only shown for real/fake; unknown has no evidential basis */}
        {result.verdict !== "unknown" && (
        <motion.div
          initial={{ opacity: 0, y: 15 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4, duration: 0.5 }}
          className="mt-6 grid grid-cols-2 gap-3"
        >
          <div
            className="rounded-2xl p-4 text-center"
            style={{ background: "rgba(255,0,60,0.08)", border: "1px solid rgba(255,0,60,0.15)" }}
          >
            <p className="text-xs font-medium uppercase tracking-widest" style={{ color: "#ff003c99" }}>
              Fake Probability
            </p>
            <p className="mt-2 font-mono text-3xl font-black" style={{ color: "#ff003c" }}>
              <AnimatedNumber value={fakePct} />
            </p>
            <div className="progress-bar mt-2">
              <motion.div
                className="progress-fill"
                initial={{ width: 0 }}
                animate={{ width: `${fakePct}%` }}
                transition={{ duration: 1.2, delay: 0.5, ease: [0.16, 1, 0.3, 1] }}
                style={{ background: "linear-gradient(90deg, #9f1239, #ff003c)" }}
              />
            </div>
          </div>

          <div
            className="rounded-2xl p-4 text-center"
            style={{ background: "rgba(0,255,136,0.06)", border: "1px solid rgba(0,255,136,0.12)" }}
          >
            <p className="text-xs font-medium uppercase tracking-widest" style={{ color: "#00ff8899" }}>
              Real Probability
            </p>
            <p className="mt-2 font-mono text-3xl font-black" style={{ color: "#00ff88" }}>
              <AnimatedNumber value={realPct} />
            </p>
            <div className="progress-bar mt-2">
              <motion.div
                className="progress-fill"
                initial={{ width: 0 }}
                animate={{ width: `${realPct}%` }}
                transition={{ duration: 1.2, delay: 0.5, ease: [0.16, 1, 0.3, 1] }}
                style={{ background: "linear-gradient(90deg, #059669, #00ff88)" }}
              />
            </div>
          </div>
        </motion.div>
        )}
      </div>

      {/* ── Branch scores ──────────────────────────────────────────────── */}
      {branchEntries.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5, duration: 0.5 }}
          className="card-glow"
        >
          <div className="mb-4 flex items-center gap-2">
            <Zap className="h-4 w-4" style={{ color: "#6366f1" }} />
            <h3 className="text-sm font-bold text-gray-200">Multi-Branch Analysis</h3>
            <span className="ml-auto text-xs text-gray-600">
              Each branch independently evaluates a signal domain
            </span>
          </div>
          <div className="space-y-2">
            {branchEntries.map(([k, v], i) => (
              <motion.div
                key={k}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.6 + i * 0.08 }}
              >
                <BranchBar name={k} value={v} />
              </motion.div>
            ))}
          </div>
        </motion.div>
      )}

      {/* ── GradCAM heatmap ─────────────────────────────────────────────── */}
      {result.heatmap_url && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7, duration: 0.5 }}
          className="card-glow"
        >
          <div className="mb-3 flex items-center gap-2">
            <Eye className="h-4 w-4" style={{ color: "#00f5ff" }} />
            <h3 className="text-sm font-bold text-gray-200">GradCAM Explanation</h3>
          </div>
          <div
            className="overflow-hidden rounded-xl"
            style={{ border: "1px solid rgba(0,245,255,0.15)" }}
          >
            <img
              src={result.heatmap_url}
              alt="GradCAM heatmap — red regions drove fake verdict"
              className="w-full object-contain"
            />
          </div>
          <div className="mt-2 flex items-center gap-1.5">
            <AlertTriangle className="h-3.5 w-3.5 text-yellow-500" />
            <p className="text-xs text-gray-600">
              Red regions indicate areas with the strongest influence on the fake verdict.
            </p>
          </div>
        </motion.div>
      )}
    </motion.div>
  );
}
