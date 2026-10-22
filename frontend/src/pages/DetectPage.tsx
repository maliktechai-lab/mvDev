import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import {
  UploadCloud, FileImage, FileVideo, FileAudio,
  X, AlertCircle, RotateCcw, Clock,
} from "lucide-react";
import { clsx } from "clsx";
import toast from "react-hot-toast";
import { detectApi } from "@/lib/api";
import { ResultCard } from "@/components/results/ResultCard";
import { VideoTimeline } from "@/components/results/VideoTimeline";
import { AudioWaveform } from "@/components/results/AudioWaveform";
import { useJobPoller } from "@/hooks/useJobPoller";
import type { DetectionJob } from "@/types";

type MediaType = "image" | "video" | "audio";

const ACCEPTED_TYPES: Record<MediaType, string[]> = {
  image: ["image/jpeg", "image/png", "image/webp", "image/bmp"],
  video: ["video/mp4", "video/webm", "video/avi", "video/quicktime", "video/x-matroska"],
  audio: ["audio/mpeg", "audio/wav", "audio/ogg", "audio/flac", "audio/mp4"],
};

const FILE_ICONS: Record<MediaType, React.ElementType> = {
  image: FileImage,
  video: FileVideo,
  audio: FileAudio,
};

function detectMediaType(file: File): MediaType | null {
  for (const [type, mimes] of Object.entries(ACCEPTED_TYPES)) {
    if (mimes.includes(file.type)) return type as MediaType;
  }
  return null;
}

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

// Polling status display
function JobStatusBanner({ job }: { job: DetectionJob | null }) {
  if (!job) return null;
  const steps = ["pending", "running", "completed"];
  const current = steps.indexOf(job.status);

  return (
    <div className="card">
      <div className="mb-4 flex items-center gap-3">
        <Clock className="h-5 w-5 text-brand-400 animate-pulse-slow" />
        <p className="font-medium text-gray-200">
          {job.status === "running" ? "Analyzing video…" : "Processing…"}
        </p>
      </div>
      <div className="flex items-center gap-2">
        {steps.map((step, i) => (
          <div key={step} className="flex items-center gap-2">
            <div
              className={clsx(
                "h-2 w-2 rounded-full",
                i < current
                  ? "bg-green-500"
                  : i === current
                  ? "bg-brand-500 animate-pulse"
                  : "bg-gray-700",
              )}
            />
            <span
              className={clsx(
                "text-xs capitalize",
                i === current ? "text-brand-400" : "text-gray-600",
              )}
            >
              {step}
            </span>
            {i < steps.length - 1 && (
              <div className={clsx("h-px w-6", i < current ? "bg-green-700" : "bg-gray-700")} />
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

export function DetectPage() {
  const [file, setFile]               = useState<File | null>(null);
  const [mediaType, setMediaType]     = useState<MediaType | null>(null);
  const [uploadPct, setUploadPct]     = useState(0);
  const [isUploading, setIsUploading] = useState(false);
  const [imageResult, setImageResult] = useState<object | null>(null);
  const [pollingJobId, setPollingJobId] = useState<string | null>(null);
  const [completedJob, setCompletedJob] = useState<DetectionJob | null>(null);

  // Video / async job polling
  const { job: polledJob, isPolling } = useJobPoller({
    jobId: pollingJobId,
    intervalMs: 3000,
    onComplete: (job) => {
      setCompletedJob(job);
      setPollingJobId(null);
      toast.success("Video analysis complete!");
    },
    onError: (job) => {
      setPollingJobId(null);
      toast.error(job.error_msg || "Video analysis failed");
    },
  });

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const dropped = acceptedFiles[0];
    if (!dropped) return;
    const type = detectMediaType(dropped);
    if (!type) { toast.error("Unsupported file type"); return; }
    setFile(dropped);
    setMediaType(type);
    setImageResult(null);
    setCompletedJob(null);
    setPollingJobId(null);
    setUploadPct(0);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    multiple: false,
    accept: {
      "image/*": [".jpg", ".jpeg", ".png", ".webp", ".bmp"],
      "video/*": [".mp4", ".webm", ".avi", ".mov", ".mkv"],
      "audio/*": [".mp3", ".wav", ".ogg", ".flac", ".m4a"],
    },
  });

  const clearFile = () => {
    setFile(null); setMediaType(null);
    setImageResult(null); setCompletedJob(null);
    setPollingJobId(null); setUploadPct(0);
  };

  const handleAnalyze = async () => {
    if (!file || !mediaType) return;
    setIsUploading(true);
    setUploadPct(0);

    try {
      if (mediaType === "image") {
        const res = await detectApi.image(file, true, setUploadPct);
        setImageResult(res.data);
        toast.success("Analysis complete!");

      } else if (mediaType === "video") {
        const res = await detectApi.video(file, setUploadPct);
        const { job_id } = res.data;
        setPollingJobId(job_id);
        toast("Video uploaded — analyzing in background…", { icon: "🎬" });

      } else if (mediaType === "audio") {
        const res = await detectApi.audio(file, setUploadPct);
        const { job_id } = res.data;
        setPollingJobId(job_id);
        toast("Audio uploaded — analyzing in background…", { icon: "🎵" });
      }
    } catch {
      // Handled by interceptor
    } finally {
      setIsUploading(false);
    }
  };

  const FileIcon  = mediaType ? FILE_ICONS[mediaType] : UploadCloud;
  const hasResult = imageResult !== null || completedJob !== null;
  const isBusy    = isUploading || isPolling;

  // Extract video result from completed job
  const videoResult = completedJob?.result;
  const frameScores  = videoResult?.frame_scores ?? [];
  const suspiciousTs = completedJob?.result
    ? (completedJob.result as { raw_output?: { suspicious_timestamps?: number[] } })
        .raw_output?.suspicious_timestamps ?? []
    : [];

  return (
    <div className="p-8">
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">Analyze Media</h1>
        <p className="mt-1 text-sm text-gray-400">
          Upload a file to detect synthetic or AI-generated media.
        </p>
      </div>

      <div className="mx-auto max-w-2xl">
        {/* Dropzone */}
        {!file ? (
          <div
            {...getRootProps()}
            className={clsx(
              "cursor-pointer rounded-2xl border-2 border-dashed p-16 text-center transition",
              isDragActive
                ? "border-brand-500 bg-brand-500/10"
                : "border-gray-700 bg-gray-900 hover:border-gray-500 hover:bg-gray-800",
            )}
          >
            <input {...getInputProps()} aria-label="Upload media file" />
            <UploadCloud className="mx-auto mb-4 h-12 w-12 text-gray-500" />
            <p className="text-lg font-medium text-gray-200">
              {isDragActive ? "Drop it here!" : "Drag & drop or click to upload"}
            </p>
            <p className="mt-2 text-sm text-gray-500">
              Images (JPG, PNG, WebP) · Videos (MP4, WebM, MKV) · Audio (MP3, WAV, FLAC)
            </p>
            <p className="mt-1 text-xs text-gray-600">
              Max: 10 MB images · 500 MB videos · 50 MB audio
            </p>
          </div>
        ) : (
          <div className="space-y-4">
            {/* File card */}
            <div className="card">
              <div className="flex items-center gap-4">
                <div className="flex h-14 w-14 items-center justify-center rounded-xl bg-brand-600/20">
                  <FileIcon className="h-7 w-7 text-brand-400" />
                </div>
                <div className="min-w-0 flex-1">
                  <p className="truncate font-medium text-white">{file.name}</p>
                  <p className="text-sm text-gray-400">
                    {mediaType?.toUpperCase()} · {formatBytes(file.size)}
                  </p>
                </div>
                {!isBusy && (
                  <button onClick={clearFile} className="rounded-lg p-2 text-gray-500 hover:bg-gray-800 hover:text-gray-300" aria-label="Remove">
                    <X className="h-4 w-4" />
                  </button>
                )}
              </div>

              {/* Upload progress */}
              {isUploading && uploadPct > 0 && uploadPct < 100 && (
                <div className="mt-4">
                  <div className="mb-1 flex justify-between text-xs text-gray-400">
                    <span>Uploading…</span><span>{uploadPct}%</span>
                  </div>
                  <div className="h-1.5 overflow-hidden rounded-full bg-gray-800">
                    <div className="h-full rounded-full bg-brand-500 transition-all" style={{ width: `${uploadPct}%` }} />
                  </div>
                </div>
              )}

              {/* Analyze button */}
              {!hasResult && !isPolling && (
                <button onClick={handleAnalyze} disabled={isBusy} className="btn-primary mt-5 w-full justify-center py-3 text-base">
                  {isBusy ? (
                    <><span className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent" />
                    {uploadPct < 100 ? "Uploading…" : "Starting…"}</>
                  ) : (
                    <><FileIcon className="h-5 w-5" />Analyze {mediaType}</>
                  )}
                </button>
              )}

              {/* Reset */}
              {(hasResult) && (
                <button onClick={clearFile} className="btn-secondary mt-5 w-full justify-center">
                  <RotateCcw className="h-4 w-4" />Analyze another file
                </button>
              )}
            </div>

            {/* Video polling status */}
            {isPolling && <JobStatusBanner job={polledJob} />}

            {/* Image result */}
            {imageResult && (
              <ResultCard
                result={imageResult as Parameters<typeof ResultCard>[0]["result"]}
                filename={file.name}
              />
            )}

            {/* Video result */}
            {completedJob?.result && completedJob.media_type === "video" && (
              <>
                <ResultCard
                  result={{
                    job_id:            completedJob.id,
                    verdict:           completedJob.result.verdict,
                    confidence:        completedJob.result.confidence,
                    fake_probability:  (completedJob.result as { raw_output?: { fake_probability?: number } }).raw_output?.fake_probability ?? completedJob.result.confidence,
                    real_probability:  (completedJob.result as { raw_output?: { real_probability?: number } }).raw_output?.real_probability ?? (1 - completedJob.result.confidence),
                    faces_detected:    completedJob.result.faces_detected ?? 0,
                    branch_scores:     completedJob.result.branch_scores ?? {},
                    heatmap_url:       completedJob.result.heatmap_url ?? null,
                    processing_ms:     completedJob.ended_at && completedJob.started_at
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
            {completedJob?.result && completedJob.media_type === "audio" && (
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
                    suspiciousTimestamps={
                      (completedJob.result as { raw_output?: { suspicious_timestamps?: number[] } })
                        .raw_output?.suspicious_timestamps ?? []
                    }
                  />
                )}
              </>
            )}
          </div>
        )}

        {!hasResult && !isPolling && (
          <div className="mt-6 flex gap-3 rounded-lg border border-yellow-700/30 bg-yellow-900/10 p-4">
            <AlertCircle className="mt-0.5 h-4 w-4 shrink-0 text-yellow-500" />
            <p className="text-sm text-yellow-200/70">
              <strong className="text-yellow-300">Tip:</strong> Videos are analyzed asynchronously.
              Results appear automatically when processing is complete (typically 15–60 seconds).
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
