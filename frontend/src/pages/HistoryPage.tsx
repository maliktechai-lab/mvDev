import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { motion, AnimatePresence } from "framer-motion";
import { Trash2, RefreshCw, Clock, Filter, Search } from "lucide-react";
import { jobsApi } from "@/lib/api";
import type { DetectionJob, MediaType, JobStatus } from "@/types";
import toast from "react-hot-toast";

const TYPE_COLOR: Record<string, string> = {
  image: "#00f5ff", video: "#bf00ff", audio: "#00ff88",
};

function VerdictBadge({ verdict }: { verdict: string | undefined }) {
  if (!verdict) return <span className="badge-unknown">—</span>;
  if (verdict === "fake") return <span className="badge-fake">FAKE</span>;
  if (verdict === "real") return <span className="badge-real">REAL</span>;
  return <span className="badge-unknown">UNKNOWN</span>;
}

function StatusBadge({ status }: { status: JobStatus }) {
  if (status === "completed") return <span className="badge-real">{status}</span>;
  if (status === "failed")    return <span className="badge-fake">{status}</span>;
  if (status === "running")   return <span className="badge-running">{status}</span>;
  return <span className="badge-pending">{status}</span>;
}

export function HistoryPage() {
  const [mediaFilter,  setMediaFilter]  = useState<MediaType | "">("");
  const [statusFilter, setStatusFilter] = useState<JobStatus | "">("");
  const [search,       setSearch]       = useState("");

  const { data: jobs = [], isLoading, refetch } = useQuery<DetectionJob[]>({
    queryKey: ["jobs", mediaFilter, statusFilter],
    queryFn: () =>
      jobsApi.list({
        page_size:  50,
        media_type: mediaFilter  || undefined,
        status:     statusFilter || undefined,
      }).then((r) => r.data),
  });

  const filtered = search
    ? jobs.filter((j) => j.original_filename.toLowerCase().includes(search.toLowerCase()))
    : jobs;

  const handleDelete = async (id: string) => {
    await jobsApi.delete(id);
    toast.success("Job deleted");
    refetch();
  };

  return (
    <div
      className="min-h-full px-8 py-8"
      style={{ background: "linear-gradient(180deg, #0a0f1e 0%, #030712 100%)" }}
    >
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -15 }}
        animate={{ opacity: 1, y: 0 }}
        className="mb-6 flex items-center justify-between"
      >
        <div>
          <div className="flex items-center gap-2 text-xs text-gray-600 mb-1">
            <Clock className="h-3.5 w-3.5" />
            <span>Analysis History</span>
          </div>
          <h1 className="text-3xl font-bold text-white">History</h1>
          <motion.p
            key={filtered.length}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            className="mt-1 text-sm text-gray-500"
          >
            {filtered.length} {filtered.length === 1 ? "analysis" : "analyses"} found
          </motion.p>
        </div>
        <motion.button
          onClick={() => refetch()}
          className="btn-secondary"
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.97 }}
        >
          <RefreshCw className="h-4 w-4" />
          Refresh
        </motion.button>
      </motion.div>

      {/* Filters */}
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="mb-5 flex flex-wrap gap-3"
      >
        {/* Search */}
        <div className="relative">
          <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-gray-600" />
          <input
            type="text"
            placeholder="Search filenames…"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="input pl-9 text-sm"
            style={{ width: 220 }}
          />
        </div>

        <div className="flex items-center gap-2">
          <Filter className="h-4 w-4 text-gray-600" />
        </div>

        {/* Media type filter */}
        <div className="flex gap-1.5">
          {(["", "image", "video", "audio"] as const).map((t) => (
            <motion.button
              key={t || "all"}
              onClick={() => setMediaFilter(t)}
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              className="rounded-xl px-3 py-2 text-xs font-semibold transition-all"
              style={
                mediaFilter === t
                  ? { background: "rgba(99,102,241,0.2)", border: "1px solid rgba(99,102,241,0.4)", color: "#818cf8" }
                  : { background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", color: "#475569" }
              }
            >
              {t === "" ? "All types" : t.charAt(0).toUpperCase() + t.slice(1)}
            </motion.button>
          ))}
        </div>

        {/* Status filter */}
        <div className="flex gap-1.5">
          {(["", "completed", "running", "failed"] as const).map((s) => (
            <motion.button
              key={s || "all"}
              onClick={() => setStatusFilter(s)}
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              className="rounded-xl px-3 py-2 text-xs font-semibold transition-all"
              style={
                statusFilter === s
                  ? { background: "rgba(99,102,241,0.2)", border: "1px solid rgba(99,102,241,0.4)", color: "#818cf8" }
                  : { background: "rgba(255,255,255,0.03)", border: "1px solid rgba(255,255,255,0.07)", color: "#475569" }
              }
            >
              {s === "" ? "All statuses" : s}
            </motion.button>
          ))}
        </div>
      </motion.div>

      {/* Table */}
      <motion.div
        initial={{ opacity: 0, y: 15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.15 }}
        className="card-glow overflow-hidden p-0"
      >
        {isLoading ? (
          <div className="flex flex-col items-center justify-center gap-3 py-20">
            <motion.div
              className="h-10 w-10 rounded-full border-2 border-brand-500 border-t-transparent"
              animate={{ rotate: 360 }}
              transition={{ duration: 0.8, repeat: Infinity, ease: "linear" }}
            />
            <p className="text-sm text-gray-600">Loading analyses…</p>
          </div>
        ) : filtered.length === 0 ? (
          <div className="flex flex-col items-center justify-center gap-3 py-20">
            <Clock className="h-12 w-12 text-gray-800" />
            <p className="text-sm text-gray-600">No analyses found.</p>
          </div>
        ) : (
          <div className="overflow-x-auto">
            <table className="w-full text-sm" role="table">
              <thead>
                <tr style={{ background: "rgba(255,255,255,0.03)", borderBottom: "1px solid rgba(255,255,255,0.05)" }}>
                  {["File", "Type", "Status", "Verdict", "Confidence", "Date", ""].map((h) => (
                    <th key={h} className="px-5 py-4 text-left text-xs font-semibold uppercase tracking-wider text-gray-600">
                      {h}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                <AnimatePresence>
                  {filtered.map((job, i) => {
                    const c = TYPE_COLOR[job.media_type] ?? "#6366f1";
                    return (
                      <motion.tr
                        key={job.id}
                        initial={{ opacity: 0, x: -20 }}
                        animate={{ opacity: 1, x: 0 }}
                        exit={{ opacity: 0, x: 20 }}
                        transition={{ delay: i * 0.03 }}
                        className="group transition-all hover:bg-white/[0.02]"
                        style={{ borderBottom: "1px solid rgba(255,255,255,0.03)" }}
                      >
                        <td className="px-5 py-4">
                          <div className="flex items-center gap-2.5">
                            <div
                              className="flex h-7 w-7 shrink-0 items-center justify-center rounded-lg text-xs font-bold"
                              style={{ background: `${c}12`, color: c, border: `1px solid ${c}20` }}
                            >
                              {job.media_type[0].toUpperCase()}
                            </div>
                            <span className="max-w-[180px] truncate font-medium text-gray-200">
                              {job.original_filename}
                            </span>
                          </div>
                        </td>
                        <td className="px-5 py-4">
                          <span className="text-xs capitalize" style={{ color: c }}>{job.media_type}</span>
                        </td>
                        <td className="px-5 py-4">
                          <StatusBadge status={job.status} />
                        </td>
                        <td className="px-5 py-4">
                          <VerdictBadge verdict={job.result?.verdict} />
                        </td>
                        <td className="px-5 py-4 font-mono text-gray-400">
                          {job.result ? `${(job.result.confidence * 100).toFixed(1)}%` : "—"}
                        </td>
                        <td className="px-5 py-4 text-gray-600 text-xs">
                          {new Date(job.created_at).toLocaleDateString(undefined, {
                            month: "short", day: "numeric", year: "numeric",
                          })}
                        </td>
                        <td className="px-5 py-4 text-right">
                          <motion.button
                            onClick={() => handleDelete(job.id)}
                            whileHover={{ scale: 1.15 }}
                            whileTap={{ scale: 0.9 }}
                            className="rounded-lg p-1.5 text-gray-700 opacity-0 transition-all group-hover:opacity-100 hover:bg-red-500/10 hover:text-red-400"
                            aria-label="Delete job"
                          >
                            <Trash2 className="h-3.5 w-3.5" />
                          </motion.button>
                        </td>
                      </motion.tr>
                    );
                  })}
                </AnimatePresence>
              </tbody>
            </table>
          </div>
        )}
      </motion.div>
    </div>
  );
}
