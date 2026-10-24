import { useEffect, useState } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { NeuralNetworkOrb } from "@/components/3d/NeuralNetworkOrb";
import {
  Cpu, Eye, Waves, Activity, GitBranch,
  Layers, Zap, CheckCircle2, Clock,
} from "lucide-react";

export type MediaTypeAnalysis = "image" | "video" | "audio";

interface PipelineStep {
  id: string;
  label: string;
  sublabel: string;
  icon: React.ElementType;
  color: string;
  durationMs: number;
}

const IMAGE_STEPS: PipelineStep[] = [
  { id: "preprocess",  label: "Preprocessing",      sublabel: "Face detection + alignment",         icon: Eye,       color: "#00f5ff",  durationMs: 800  },
  { id: "spatial",     label: "Spatial Branch",      sublabel: "EfficientNet-B4 texture analysis",   icon: Layers,    color: "#6366f1",  durationMs: 1200 },
  { id: "frequency",   label: "Frequency Branch",    sublabel: "DCT spectrum + FFT artifacts",       icon: Waves,     color: "#bf00ff",  durationMs: 1000 },
  { id: "attention",   label: "Attention Branch",    sublabel: "GradCAM saliency mapping",           icon: Activity,  color: "#f97316",  durationMs: 900  },
  { id: "fusion",      label: "Branch Fusion",       sublabel: "Multi-branch score aggregation",     icon: GitBranch, color: "#00ff88",  durationMs: 600  },
  { id: "verdict",     label: "Verdict Generation",  sublabel: "Final classification + confidence",  icon: Zap,       color: "#fbbf24",  durationMs: 400  },
];

const VIDEO_STEPS: PipelineStep[] = [
  { id: "extract",     label: "Frame Extraction",    sublabel: "Keyframe sampling + decoding",       icon: Cpu,       color: "#00f5ff",  durationMs: 1000 },
  { id: "spatial",     label: "Spatial Branch",      sublabel: "Per-frame EfficientNet analysis",    icon: Layers,    color: "#6366f1",  durationMs: 1400 },
  { id: "temporal",    label: "Temporal Branch",     sublabel: "LSTM sequence inconsistency",        icon: Activity,  color: "#bf00ff",  durationMs: 1200 },
  { id: "biological",  label: "Biological Branch",   sublabel: "rPPG heartbeat signal analysis",     icon: Waves,     color: "#f97316",  durationMs: 1300 },
  { id: "fusion",      label: "Branch Fusion",       sublabel: "Multi-branch score aggregation",     icon: GitBranch, color: "#00ff88",  durationMs: 700  },
  { id: "verdict",     label: "Verdict Generation",  sublabel: "Final classification + confidence",  icon: Zap,       color: "#fbbf24",  durationMs: 400  },
];

const AUDIO_STEPS: PipelineStep[] = [
  { id: "preprocess",  label: "Audio Processing",    sublabel: "Waveform normalization + segmentation", icon: Waves,  color: "#00f5ff",  durationMs: 800  },
  { id: "raw",         label: "Raw Waveform Branch", sublabel: "RawNet2 end-to-end features",           icon: Activity, color: "#6366f1", durationMs: 1200 },
  { id: "spectral",    label: "Spectral Branch",     sublabel: "MFCC + mel-spectrogram artifacts",      icon: Layers, color: "#bf00ff",  durationMs: 1000 },
  { id: "prosody",     label: "Prosody Branch",      sublabel: "Pitch / rhythm inconsistency",          icon: Cpu,    color: "#f97316",  durationMs: 900  },
  { id: "fusion",      label: "Branch Fusion",       sublabel: "Multi-branch score aggregation",        icon: GitBranch, color: "#00ff88", durationMs: 600 },
  { id: "verdict",     label: "Verdict Generation",  sublabel: "Final classification + confidence",     icon: Zap,    color: "#fbbf24",  durationMs: 400  },
];

function getSteps(type: MediaTypeAnalysis) {
  if (type === "video") return VIDEO_STEPS;
  if (type === "audio") return AUDIO_STEPS;
  return IMAGE_STEPS;
}

interface AnalysisPipelineProps {
  mediaType: MediaTypeAnalysis;
  /** If true the pipeline runs animated steps; if false shows a static "waiting" state */
  running: boolean;
  /** Percentage 0–100 for the overall progress ring */
  uploadPct?: number;
}

export function AnalysisPipeline({ mediaType, running, uploadPct = 0 }: AnalysisPipelineProps) {
  const steps = getSteps(mediaType);
  const [activeStep, setActiveStep] = useState(-1);
  const [completedSteps, setCompletedSteps] = useState<Set<number>>(new Set());

  // Drive step animation
  useEffect(() => {
    if (!running) {
      setActiveStep(-1);
      setCompletedSteps(new Set());
      return;
    }

    let step = 0;
    setActiveStep(0);

    const advance = () => {
      setCompletedSteps((prev) => new Set([...prev, step]));
      step++;
      if (step < steps.length) {
        setActiveStep(step);
        timerId = setTimeout(advance, steps[step].durationMs);
      }
    };

    let timerId = setTimeout(advance, steps[0].durationMs);
    return () => clearTimeout(timerId);
  }, [running, mediaType]); // eslint-disable-line react-hooks/exhaustive-deps

  const currentStep = steps[activeStep];
  const overallPct  = activeStep >= 0
    ? Math.round(((completedSteps.size / steps.length) * 100))
    : 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, y: -20 }}
      className="card-glow relative overflow-hidden"
    >
      {/* Top gradient strip */}
      <div
        className="absolute inset-x-0 top-0 h-px"
        style={{ background: "linear-gradient(90deg, transparent, rgba(99,102,241,0.6), transparent)" }}
      />

      {/* Header row */}
      <div className="mb-6 flex items-center gap-4">
        <div className="relative">
          <NeuralNetworkOrb size={80} />
        </div>
        <div className="flex-1">
          <div className="flex items-center gap-2">
            <motion.div
              className="h-2 w-2 rounded-full"
              style={{ background: "#00f5ff" }}
              animate={{ opacity: running ? [1, 0.3, 1] : 0.3 }}
              transition={{ duration: 0.8, repeat: Infinity }}
            />
            <p className="text-sm font-semibold" style={{ color: "#00f5ff" }}>
              {running ? "Analysis in progress" : "Preparing…"}
            </p>
          </div>
          <AnimatePresence mode="wait">
            <motion.p
              key={currentStep?.id ?? "idle"}
              initial={{ opacity: 0, y: 5 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -5 }}
              className="mt-1 text-xs"
              style={{ color: "#475569" }}
            >
              {currentStep ? `${currentStep.label} — ${currentStep.sublabel}` : "Starting pipeline…"}
            </motion.p>
          </AnimatePresence>
        </div>

        {/* Overall progress */}
        <div className="text-right">
          <motion.p
            className="font-mono text-2xl font-bold"
            style={{ color: "#6366f1" }}
            animate={{ opacity: [0.7, 1, 0.7] }}
            transition={{ duration: 1.2, repeat: Infinity }}
          >
            {running ? `${overallPct}%` : `${uploadPct}%`}
          </motion.p>
          <p className="text-xs" style={{ color: "#334155" }}>
            {running ? "analyzed" : "uploaded"}
          </p>
        </div>
      </div>

      {/* Progress bar */}
      <div className="mb-6">
        <div className="progress-bar mb-1">
          <motion.div
            className="progress-fill"
            style={{ background: "linear-gradient(90deg, #4f46e5, #00f5ff, #bf00ff)" }}
            animate={{ width: running ? `${overallPct}%` : `${uploadPct}%` }}
            transition={{ duration: 0.5, ease: "easeOut" }}
          />
          {/* Shimmer */}
          <motion.div
            className="absolute inset-y-0"
            style={{
              width: "30%",
              background: "linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent)",
            }}
            animate={{ left: ["-30%", "130%"] }}
            transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
          />
        </div>
      </div>

      {/* Pipeline steps */}
      <div className="space-y-2">
        {steps.map((step, i) => {
          const isDone   = completedSteps.has(i);
          const isActive = activeStep === i && running;
          const isPending = i > activeStep || !running;

          return (
            <motion.div
              key={step.id}
              initial={{ opacity: 0, x: -10 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: i * 0.05, duration: 0.4 }}
              className="flex items-center gap-3 rounded-xl px-4 py-3 transition-all duration-300"
              style={{
                background: isActive
                  ? `${step.color}14`
                  : isDone
                  ? "rgba(0,255,136,0.05)"
                  : "rgba(255,255,255,0.02)",
                border: `1px solid ${
                  isActive ? `${step.color}40`
                  : isDone  ? "rgba(0,255,136,0.15)"
                  : "rgba(255,255,255,0.04)"}`,
              }}
            >
              {/* Icon */}
              <div
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg transition-all"
                style={{
                  background: isActive ? `${step.color}20`
                    : isDone  ? "rgba(0,255,136,0.15)"
                    : "rgba(255,255,255,0.04)",
                }}
              >
                {isDone ? (
                  <motion.div
                    initial={{ scale: 0 }}
                    animate={{ scale: 1 }}
                    transition={{ type: "spring", stiffness: 400 }}
                  >
                    <CheckCircle2 className="h-4 w-4" style={{ color: "#00ff88" }} />
                  </motion.div>
                ) : isActive ? (
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ duration: 1.5, repeat: Infinity, ease: "linear" }}
                  >
                    <step.icon className="h-4 w-4" style={{ color: step.color }} />
                  </motion.div>
                ) : (
                  <step.icon
                    className="h-4 w-4"
                    style={{ color: isPending ? "#1e293b" : step.color }}
                  />
                )}
              </div>

              {/* Label */}
              <div className="flex-1 min-w-0">
                <p
                  className="text-sm font-medium transition-colors"
                  style={{
                    color: isActive ? step.color : isDone ? "#94a3b8" : "#1e293b",
                  }}
                >
                  {step.label}
                </p>
                <p className="text-xs truncate" style={{ color: isActive ? "#475569" : "#0f172a" }}>
                  {step.sublabel}
                </p>
              </div>

              {/* Status */}
              <div className="shrink-0">
                {isDone ? (
                  <span className="text-xs font-mono" style={{ color: "#00ff88" }}>done</span>
                ) : isActive ? (
                  <motion.div className="flex gap-1">
                    {[0, 1, 2].map((j) => (
                      <motion.div
                        key={j}
                        className="h-1.5 w-1.5 rounded-full"
                        style={{ background: step.color }}
                        animate={{ opacity: [0.3, 1, 0.3], y: [0, -3, 0] }}
                        transition={{ duration: 0.6, delay: j * 0.15, repeat: Infinity }}
                      />
                    ))}
                  </motion.div>
                ) : (
                  <Clock className="h-3.5 w-3.5" style={{ color: "#0f172a" }} />
                )}
              </div>
            </motion.div>
          );
        })}
      </div>
    </motion.div>
  );
}
