import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import { motion, AnimatePresence } from "framer-motion";
import {
  UploadCloud, FileImage, FileVideo, FileAudio,
  X, RotateCcw, Cpu, ChevronRight,
} from "lucide-react";
import { clsx } from "clsx";
import toast from "react-hot-toast";
import { detectApi } from "@/lib/api";
import { ResultCard, type ResultCardData } from "@/components/results/ResultCard";
import { VideoTimeline } from "@/components/results/VideoTimeline";
import { AudioWaveform } from "@/components/results/AudioWaveform";
import { AnalysisPipeline } from "@/components/analysis/AnalysisPipeline";
import { ScannerBeam } from "@/components/3d/ScannerBeam";
import { useJobPoller } from "@/hooks/useJobPoller";
import type { DetectionJob } from "@/types";

type MediaType = "image" | "video" | "audio";
type Stage = "idle" | "dropped" | "uploading" | "analyzing" | "done" | "polling";

const ACCEPTED = {
  image: ["image/jpeg", "image/png", "image/webp", "image/bmp"],
  video: ["video/mp4", "video/webm", "video/avi", "video/quicktime", "video/x-matroska"],
  audio: ["audio/mpeg", "audio/wav", "audio/ogg", "audio/flac", "audio/mp4"],
};

function detectType(file: File): MediaType | null {
  for (const [t, mimes] of Object.entries(ACCEPTED)) {
    if (mimes.includes(file.type)) return t as MediaType;
  }
  return null;
}

function fmtBytes(b: number) {
  if (b < 1024)          return `${b} B`;
  if (b < 1024 * 1024)   return `${(b / 1024).toFixed(1)} KB`;
  return `${(b / (1024 * 1024)).toFixed(1)} MB`;
}

const ICONS: Record<MediaType, React.ElementType> = {
  image: FileImage, video: FileVideo, audio: FileAudio,
};

const TYPE_COLORS: Record<MediaType, string> = {
  image: "#00f5ff", video: "#bf00ff", audio: "#00ff88",
};

// ── Stage label ──────────────────────────────────────────────────────────
function stageLabel(stage: Stage, mediaType: MediaType | null): string {
  if (stage === "uploading")  return "Uploading…";
  if (stage === "analyzing")  return `Analyzing ${mediaType ?? "file"}…`;
  if (stage === "polling")    return "Processing in background…";
  if (stage === "done")       return "Analysis complete";
  return "";
}

// ── Drop zone ────────────────────────────────────────────────────────────
function DropZone({ onDrop, isDragActive }: {
  onDrop: (files: File[]) => void;
  isDragActive: boolean;
}) {
  const { getRootProps, getInputProps } = useDropzone({
    onDrop,
    multiple: false,
    accept: {
      "image/*": [".jpg", ".jpeg", ".png", ".webp", ".bmp"],
      "video/*": [".mp4", ".webm", ".avi", ".mov", ".mkv"],
      "audio/*": [".mp3", ".wav", ".ogg", ".flac", ".m4a"],
    },
  });

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0, scale: 0.95 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
    >
      <div
        {...getRootProps()}
        className={clsx(
          "group relative cursor-pointer rounded-2xl p-16 text-center transition-all duration-300",
        )}
        style={{
          background: isDragActive
            ? "rgba(0,245,255,0.08)"
            : "rgba(255,255,255,0.02)",
          border: `2px dashed ${isDragActive ? "rgba(0,245,255,0.6)" : "rgba(255,255,255,0.08)"}`,
          boxShadow: isDragActive ? "0 0 40px rgba(0,245,255,0.15), inset 0 0 40px rgba(0,245,255,0.05)" : "none",
        }}
      >
        <input {...getInputProps()} aria-label="Upload media file" />

        {/* Corner brackets */}
        {["top-3 left-3 border-t border-l", "top-3 right-3 border-t border-r",
          "bottom-3 left-3 border-b border-l", "bottom-3 right-3 border-b border-r"].map((cls, i) => (
          <div
            key={i}
            className={`absolute h-6 w-6 ${cls} transition-all duration-300`}
            style={{
              borderColor: isDragActive ? "rgba(0,245,255,0.7)" : "rgba(255,255,255,0.12)",
              boxShadow: isDragActive ? `0 0 8px rgba(0,245,255,0.5)` : "none",
            }}
          />
        ))}

        {/* Icon */}
        <motion.div
          animate={isDragActive
            ? { scale: [1, 1.2, 1], y: [0, -8, 0] }
            : { y: [0, -6, 0] }}
          transition={isDragActive
            ? { duration: 0.5, repeat: 0 }
            : { duration: 3, repeat: Infinity, ease: "easeInOut" }}
          className="mx-auto mb-5 flex h-20 w-20 items-center justify-center rounded-2xl"
          style={{
            background: isDragActive
              ? "rgba(0,245,255,0.15)"
              : "rgba(255,255,255,0.04)",
            border: `1px solid ${isDragActive ? "rgba(0,245,255,0.4)" : "rgba(255,255,255,0.07)"}`,
            boxShadow: isDragActive ? "0 0 30px rgba(0,245,255,0.3)" : "none",
          }}
        >
          <UploadCloud
            className="h-10 w-10"
            style={{ color: isDragActive ? "#00f5ff" : "#334155" }}
          />
        </motion.div>

        <p className="mb-2 text-xl font-semibold text-gray-300">
          {isDragActive ? "Release to analyze" : "Drop media here"}
        </p>
        <p className="text-sm text-gray-600">
          or click to browse your files
        </p>
        <p className="mt-3 text-xs text-gray-700">
          Images (JPG, PNG, WebP) · Videos (MP4, WebM, MKV) · Audio (MP3, WAV, FLAC)
        </p>

        {/* Supported types row */}
        <div className="mt-6 flex justify-center gap-3">
          {[
            { icon: FileImage, label: "Image",  color: "#00f5ff", limit: "10 MB" },
            { icon: FileVideo, label: "Video",  color: "#bf00ff", limit: "500 MB" },
            { icon: FileAudio, label: "Audio",  color: "#00ff88", limit: "50 MB" },
          ].map(({ icon: Icon, label, color, limit }) => (
            <div
              key={label}
              className="flex items-center gap-2 rounded-xl px-3 py-2 text-xs"
              style={{
                background: `${color}0a`,
                border: `1px solid ${color}20`,
                color: "#475569",
              }}
            >
              <Icon className="h-3.5 w-3.5" style={{ color }} />
              <span>{label}</span>
              <span style={{ color: "#1e293b" }}>· {limit}</span>
            </div>
          ))}
        </div>
      </div>
    </motion.div>
  );
}

// ── File card ────────────────────────────────────────────────────────────
function FileCard({ file, mediaType, onClear, stage, uploadPct }: {
  file: File;
  mediaType: MediaType;
  onClear: () => void;
  stage: Stage;
  uploadPct: number;
}) {
  const Icon  = ICONS[mediaType];
  const color = TYPE_COLORS[mediaType];
  const busy  = stage === "uploading" || stage === "analyzing" || stage === "polling";

  return (
    <motion.div
      initial={{ opacity: 0, y: 15 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
      className="relative overflow-hidden rounded-2xl p-5"
      style={{
        background: "rgba(255,255,255,0.03)",
        border: `1px solid ${color}25`,
        boxShadow: `0 0 20px ${color}0a`,
      }}
    >
      {/* Scanner overlay when uploading */}
      <ScannerBeam active={stage === "uploading"} color="cyan" />

      <div className="flex items-center gap-4">
        <motion.div
          animate={busy ? { boxShadow: [`0 0 10px ${color}40`, `0 0 25px ${color}60`, `0 0 10px ${color}40`] } : {}}
          transition={{ duration: 1.5, repeat: Infinity }}
          className="flex h-14 w-14 shrink-0 items-center justify-center rounded-xl"
          style={{ background: `${color}18`, border: `1px solid ${color}30` }}
        >
          <Icon className="h-7 w-7" style={{ color }} />
        </motion.div>

        <div className="min-w-0 flex-1">
          <p className="truncate font-semibold text-gray-100">{file.name}</p>
          <div className="mt-1 flex items-center gap-2 text-xs text-gray-500">
            <span
              className="rounded-full px-2 py-0.5 font-semibold uppercase"
              style={{ background: `${color}15`, color }}
            >
              {mediaType}
            </span>
            <span>{fmtBytes(file.size)}</span>
            {stage !== "idle" && stage !== "dropped" && (
              <motion.span
                animate={{ opacity: [0.5, 1, 0.5] }}
                transition={{ duration: 1, repeat: Infinity }}
                style={{ color }}
              >
                · {stageLabel(stage, mediaType)}
              </motion.span>
            )}
          </div>
        </div>

        {!busy && stage !== "done" && (
          <button
            onClick={onClear}
            className="rounded-xl p-2 text-gray-600 transition-all hover:bg-white/5 hover:text-gray-300"
            aria-label="Remove file"
          >
            <X className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* Upload progress bar */}
      <AnimatePresence>
        {stage === "uploading" && uploadPct > 0 && uploadPct < 100 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: "auto" }}
            exit={{ opacity: 0, height: 0 }}
            className="mt-4"
          >
            <div className="mb-1 flex justify-between text-xs">
              <span style={{ color: "#475569" }}>Uploading…</span>
              <span className="font-mono" style={{ color }}>{uploadPct}%</span>
            </div>
            <div className="progress-bar">
              <motion.div
                className="progress-fill"
                animate={{ width: `${uploadPct}%` }}
                style={{ background: `linear-gradient(90deg, ${color}88, ${color})` }}
              />
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}

// ── Polling status card ──────────────────────────────────────────────────
function PollingCard({ job }: { job: DetectionJob | null }) {
  const steps = ["pending", "running", "completed"];
  const cur   = job ? steps.indexOf(job.status) : 0;

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      className="card-glow"
    >
      <div className="mb-3 flex items-center gap-2">
        <motion.div
          className="h-2 w-2 rounded-full"
          style={{ background: "#bf00ff" }}
          animate={{ opacity: [1, 0.3, 1] }}
          transition={{ duration: 0.7, repeat: Infinity }}
        />
        <p className="text-sm font-semibold" style={{ color: "#bf00ff" }}>
          {job?.status === "running" ? "Running inference…" : "Queued for processing…"}
        </p>
      </div>
      <div className="flex items-center gap-3">
        {steps.map((step, i) => (
          <div key={step} className="flex items-center gap-2">
            <div
              className="h-2 w-2 rounded-full"
              style={{
                background: i < cur ? "#00ff88" : i === cur ? "#bf00ff" : "rgba(255,255,255,0.1)",
                boxShadow: i === cur ? "0 0 8px #bf00ff" : "none",
              }}
            />
            <span className="text-xs capitalize" style={{ color: i === cur ? "#bf00ff" : i < cur ? "#00ff88" : "#1e293b" }}>
              {step}
            </span>
            {i < steps.length - 1 && (
              <div className="h-px w-6" style={{ background: i < cur ? "#00ff88" : "rgba(255,255,255,0.05)" }} />
            )}
          </div>
        ))}
      </div>
    </motion.div>
  );
}

// ── Main page ────────────────────────────────────────────────────────────
export function DetectPage() {
  const [file, setFile]               = useState<File | null>(null);
  const [mediaType, setMediaType]     = useState<MediaType | null>(null);
  const [stage, setStage]             = useState<Stage>("idle");
  const [uploadPct, setUploadPct]     = useState(0);
  const [imageResult, setImageResult] = useState<ResultCardData | null>(null);
  const [pollingJobId, setPollingJobId] = useState<string | null>(null);
  const [completedJob, setCompletedJob] = useState<DetectionJob | null>(null);
  const [isDragActive, setIsDragActive] = useState(false);

  const { job: polledJob } = useJobPoller({
    jobId: pollingJobId,
    intervalMs: 3000,
    onComplete: (job) => {
      setCompletedJob(job);
      setPollingJobId(null);
      setStage("done");
      toast.success("Analysis complete!");
    },
    onError: (job) => {
      setPollingJobId(null);
      setStage("dropped");
      toast.error(job.error_msg || "Analysis failed");
    },
  });

  const onDrop = useCallback((accepted: File[]) => {
    const f = accepted[0];
    if (!f) return;
    const t = detectType(f);
    if (!t) { toast.error("Unsupported file type"); return; }
    setFile(f);
    setMediaType(t);
    setStage("dropped");
    setImageResult(null);
    setCompletedJob(null);
    setPollingJobId(null);
    setUploadPct(0);
    setIsDragActive(false);
  }, []);

  const { getRootProps: dz, getInputProps: dzInput } = useDropzone({
    onDrop, multiple: false,
    onDragEnter: () => setIsDragActive(true),
    onDragLeave: () => setIsDragActive(false),
    accept: {
      "image/*": [".jpg", ".jpeg", ".png", ".webp", ".bmp"],
      "video/*": [".mp4", ".webm", ".avi", ".mov", ".mkv"],
      "audio/*": [".mp3", ".wav", ".ogg", ".flac", ".m4a"],
    },
  });

  const clearFile = () => {
    setFile(null); setMediaType(null); setStage("idle");
    setImageResult(null); setCompletedJob(null);
    setPollingJobId(null); setUploadPct(0);
  };

  const handleAnalyze = async () => {
    if (!file || !mediaType) return;
    setStage("uploading");
    setUploadPct(0);
    try {
      if (mediaType === "image") {
        const res = await detectApi.image(file, true, (p) => {
          setUploadPct(p);
          if (p === 100) setStage("analyzing");
        });
        setImageResult(res.data as ResultCardData);
        setStage("done");
        toast.success("Analysis complete!");

      } else if (mediaType === "video") {
        const res = await detectApi.video(file, setUploadPct);
        setPollingJobId(res.data.job_id);
        setStage("polling");
        toast("Video uploaded — analyzing in background…", { icon: "🎬" });

      } else if (mediaType === "audio") {
        const res = await detectApi.audio(file, setUploadPct);
        setPollingJobId(res.data.job_id);
        setStage("polling");
        toast("Audio uploaded — analyzing in background…", { icon: "🎵" });
      }
    } catch {
      setStage("dropped");
    }
  };

  const hasResult = imageResult !== null || completedJob !== null;
  const videoResult = completedJob?.result;
  const frameScores = videoResult?.frame_scores ?? [];
  const suspiciousTs = (completedJob?.result as { raw_output?: { suspicious_timestamps?: number[] } } | undefined)
    ?.raw_output?.suspicious_timestamps ?? [];

  return (
    <div
      className="min-h-full"
      style={{
        background: "linear-gradient(180deg, #0a0f1e 0%, #030712 100%)",
      }}
    >
      {/* Page header */}
      <div className="px-8 pt-8 pb-2">
        <motion.div
          initial={{ opacity: 0, y: -10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.4 }}
        >
          <div className="flex items-center gap-2 text-xs text-gray-600 mb-2">
            <Cpu className="h-3.5 w-3.5" />
            <span>DeepGuard</span>
            <ChevronRight className="h-3 w-3" />
            <span className="text-gray-400">Analyze Media</span>
          </div>
          <h1 className="text-3xl font-bold text-white">
            Detect{" "}
            <span className="gradient-text">Synthetic Media</span>
          </h1>
          <p className="mt-1 text-sm text-gray-500">
            Upload an image, video, or audio file for multi-branch deepfake analysis.
          </p>
        </motion.div>
      </div>

      <div className="px-8 pb-8">
        <div className="mx-auto max-w-3xl">
          {/* Breadcrumb stage indicator */}
          {stage !== "idle" && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="mb-6 flex items-center gap-2"
            >
              {["dropped", "uploading", "analyzing", "polling", "done"].map((s, i) => {
                const stages = ["dropped", "uploading", "analyzing", "polling", "done"];
                const cur = stages.indexOf(stage);
                const isDone = i <= cur;
                const isActive = s === stage;
                const labels: Record<string, string> = {
                  dropped: "File Selected", uploading: "Uploading", analyzing: "Analyzing",
                  polling: "Processing", done: "Complete",
                };
                return (
                  <div key={s} className="flex items-center gap-2">
                    <div
                      className="flex h-6 w-6 items-center justify-center rounded-full text-xs font-bold transition-all"
                      style={{
                        background: isDone ? (isActive ? "#6366f1" : "rgba(99,102,241,0.3)") : "rgba(255,255,255,0.05)",
                        color: isDone ? (isActive ? "#fff" : "#818cf8") : "#1e293b",
                        boxShadow: isActive ? "0 0 12px rgba(99,102,241,0.6)" : "none",
                      }}
                    >
                      {i + 1}
                    </div>
                    <span className="text-xs" style={{ color: isActive ? "#818cf8" : isDone ? "#334155" : "#0f172a" }}>
                      {labels[s]}
                    </span>
                    {i < 4 && <div className="h-px w-4" style={{ background: isDone ? "rgba(99,102,241,0.4)" : "rgba(255,255,255,0.04)" }} />}
                  </div>
                );
              })}
            </motion.div>
          )}

          <div className="space-y-5">
            {/* Drop zone (idle) */}
            <AnimatePresence>
              {stage === "idle" && (
                <div {...dz()}>
                  <input {...dzInput()} />
                  <DropZone onDrop={onDrop} isDragActive={isDragActive} />
                </div>
              )}
            </AnimatePresence>

            {/* File card (any stage) */}
            <AnimatePresence>
              {file && mediaType && stage !== "idle" && (
                <FileCard
                  file={file}
                  mediaType={mediaType}
                  onClear={clearFile}
                  stage={stage}
                  uploadPct={uploadPct}
                />
              )}
            </AnimatePresence>

            {/* Analyze button */}
            <AnimatePresence>
              {stage === "dropped" && (
                <motion.div
                  initial={{ opacity: 0, y: 10 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0, y: -10 }}
                  className="flex gap-3"
                >
                  <motion.button
                    onClick={handleAnalyze}
                    className="btn-primary flex-1 justify-center py-4 text-base"
                    whileHover={{ scale: 1.01 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <Cpu className="h-5 w-5" />
                    Run {mediaType?.charAt(0).toUpperCase()}{mediaType?.slice(1)} Analysis
                  </motion.button>
                  <button onClick={clearFile} className="btn-secondary px-4" aria-label="Cancel">
                    <X className="h-4 w-4" />
                  </button>
                </motion.div>
              )}
            </AnimatePresence>

            {/* Analysis pipeline visualization */}
            <AnimatePresence>
              {(stage === "analyzing" || stage === "uploading") && mediaType && (
                <AnalysisPipeline
                  mediaType={mediaType}
                  running={stage === "analyzing"}
                  uploadPct={uploadPct}
                />
              )}
            </AnimatePresence>

            {/* Polling status */}
            <AnimatePresence>
              {stage === "polling" && <PollingCard job={polledJob} />}
            </AnimatePresence>

            {/* Results */}
            <AnimatePresence>
              {stage === "done" && hasResult && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  className="space-y-5"
                >
                  {/* Image result */}
                  {imageResult && file && (
                    <ResultCard result={imageResult} filename={file.name} />
                  )}

                  {/* Video result */}
                  {completedJob?.result && completedJob.media_type === "video" && file && (
                    <>
                      <ResultCard
                        result={{
                          job_id:           completedJob.id,
                          verdict:          completedJob.result.verdict,
                          confidence:       completedJob.result.confidence,
                          fake_probability: (completedJob.result as { raw_output?: { fake_probability?: number } }).raw_output?.fake_probability ?? completedJob.result.confidence,
                          real_probability: (completedJob.result as { raw_output?: { real_probability?: number } }).raw_output?.real_probability ?? (1 - completedJob.result.confidence),
                          faces_detected:   completedJob.result.faces_detected ?? 0,
                          branch_scores:    completedJob.result.branch_scores ?? {},
                          heatmap_url:      completedJob.result.heatmap_url ?? null,
                          processing_ms:    completedJob.ended_at && completedJob.started_at
                            ? new Date(completedJob.ended_at).getTime() - new Date(completedJob.started_at).getTime()
                            : 0,
                        }}
                        filename={file.name}
                      />
                      {frameScores.length > 0 && (
                        <VideoTimeline
                          frameScores={frameScores}
                          duration={completedJob.result.duration_sec ?? 0}
                          suspiciousTimestamps={suspiciousTs}
                        />
                      )}
                    </>
                  )}

                  {/* Audio result */}
                  {completedJob?.result && completedJob.media_type === "audio" && file && (
                    <>
                      <ResultCard
                        result={{
                          job_id:           completedJob.id,
                          verdict:          completedJob.result.verdict,
                          confidence:       completedJob.result.confidence,
                          fake_probability: (completedJob.result as { raw_output?: { fake_probability?: number } }).raw_output?.fake_probability ?? completedJob.result.confidence,
                          real_probability: (completedJob.result as { raw_output?: { real_probability?: number } }).raw_output?.real_probability ?? (1 - completedJob.result.confidence),
                          faces_detected:   0,
                          branch_scores:    completedJob.result.branch_scores ?? {},
                          heatmap_url:      null,
                          processing_ms:    completedJob.ended_at && completedJob.started_at
                            ? new Date(completedJob.ended_at).getTime() - new Date(completedJob.started_at).getTime()
                            : 0,
                        }}
                        filename={file.name}
                      />
                      {(completedJob.result.chunk_scores?.length ?? 0) > 0 && (
                        <AudioWaveform
                          chunkScores={completedJob.result.chunk_scores!}
                          duration={completedJob.result.duration_sec ?? 0}
                          suspiciousTimestamps={suspiciousTs}
                        />
                      )}
                    </>
                  )}

                  {/* Analyze again */}
                  <motion.button
                    onClick={clearFile}
                    className="btn-secondary w-full justify-center py-3"
                    whileHover={{ scale: 1.01 }}
                    whileTap={{ scale: 0.98 }}
                  >
                    <RotateCcw className="h-4 w-4" />
                    Analyze Another File
                  </motion.button>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>
      </div>
    </div>
  );
}
