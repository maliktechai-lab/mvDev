import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { Trash2, RefreshCw } from "lucide-react";
import { jobsApi } from "@/lib/api";
import type { DetectionJob, MediaType, JobStatus } from "@/types";
import toast from "react-hot-toast";
import { clsx } from "clsx";

function VerdictBadge({ verdict }: { verdict: string | undefined }) {
  if (!verdict) return <span className="badge-unknown">—</span>;
  if (verdict === "fake") return <span className="badge-fake">FAKE</span>;
  if (verdict === "real") return <span className="badge-real">REAL</span>;
  return <span className="badge-unknown">UNKNOWN</span>;
}

function StatusBadge({ status }: { status: JobStatus }) {
  const colors: Record<JobStatus, string> = {
    pending:   "text-yellow-400 bg-yellow-900/30",
    running:   "text-blue-400 bg-blue-900/30",
    completed: "text-green-400 bg-green-900/30",
    failed:    "text-red-400 bg-red-900/30",
  };
  return (
    <span className={clsx("rounded-full px-2.5 py-0.5 text-xs font-medium", colors[status])}>
      {status}
    </span>
  );
}

export function HistoryPage() {
  const [mediaFilter, setMediaFilter] = useState<MediaType | "">("");
  const [statusFilter, setStatusFilter] = useState<JobStatus | "">("");

  const { data: jobs = [], isLoading, refetch } = useQuery<DetectionJob[]>({
    queryKey: ["jobs", mediaFilter, statusFilter],
    queryFn: () =>
      jobsApi.list({
        page_size: 50,
        media_type: mediaFilter || undefined,
        status: statusFilter || undefined,
      }).then((r) => r.data),
  });

  const handleDelete = async (id: string) => {
    await jobsApi.delete(id);
    toast.success("Job deleted");
    refetch();
  };

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold text-white">History</h1>
          <p className="mt-1 text-sm text-gray-400">{jobs.length} analyses found</p>
        </div>
        <button onClick={() => refetch()} className="btn-secondary">
          <RefreshCw className="h-4 w-4" />
          Refresh
        </button>
      </div>

      {/* Filters */}
      <div className="mb-6 flex gap-3">
        <select
          className="input max-w-[160px]"
          value={mediaFilter}
          onChange={(e) => setMediaFilter(e.target.value as MediaType | "")}
          aria-label="Filter by media type"
        >
          <option value="">All types</option>
          <option value="image">Image</option>
          <option value="video">Video</option>
          <option value="audio">Audio</option>
        </select>

        <select
          className="input max-w-[160px]"
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value as JobStatus | "")}
          aria-label="Filter by status"
        >
          <option value="">All statuses</option>
          <option value="completed">Completed</option>
          <option value="running">Running</option>
          <option value="pending">Pending</option>
          <option value="failed">Failed</option>
        </select>
      </div>

      {/* Table */}
      <div className="card overflow-hidden p-0">
        {isLoading ? (
          <div className="flex items-center justify-center py-16">
            <span className="h-8 w-8 animate-spin rounded-full border-2 border-brand-500 border-t-transparent" />
          </div>
        ) : jobs.length === 0 ? (
          <p className="py-16 text-center text-sm text-gray-500">No jobs found.</p>
        ) : (
          <table className="w-full text-sm" role="table">
            <thead>
              <tr className="border-b border-gray-800 bg-gray-800/50">
                <th className="px-6 py-3 text-left font-medium text-gray-400">File</th>
                <th className="px-6 py-3 text-left font-medium text-gray-400">Type</th>
                <th className="px-6 py-3 text-left font-medium text-gray-400">Status</th>
                <th className="px-6 py-3 text-left font-medium text-gray-400">Verdict</th>
                <th className="px-6 py-3 text-left font-medium text-gray-400">Confidence</th>
                <th className="px-6 py-3 text-left font-medium text-gray-400">Date</th>
                <th className="px-6 py-3 text-right font-medium text-gray-400">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-800">
              {jobs.map((job) => (
                <tr key={job.id} className="hover:bg-gray-800/30 transition">
                  <td className="max-w-[200px] truncate px-6 py-3 font-medium text-gray-100">
                    {job.original_filename}
                  </td>
                  <td className="px-6 py-3 text-gray-400 capitalize">{job.media_type}</td>
                  <td className="px-6 py-3">
                    <StatusBadge status={job.status} />
                  </td>
                  <td className="px-6 py-3">
                    <VerdictBadge verdict={job.result?.verdict} />
                  </td>
                  <td className="px-6 py-3 text-gray-300">
                    {job.result
                      ? `${(job.result.confidence * 100).toFixed(1)}%`
                      : "—"}
                  </td>
                  <td className="px-6 py-3 text-gray-400">
                    {new Date(job.created_at).toLocaleDateString()}
                  </td>
                  <td className="px-6 py-3 text-right">
                    <button
                      onClick={() => handleDelete(job.id)}
                      className="rounded p-1.5 text-gray-500 transition hover:bg-red-900/30 hover:text-red-400"
                      aria-label="Delete job"
                    >
                      <Trash2 className="h-4 w-4" />
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
