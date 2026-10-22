import { useQuery } from "@tanstack/react-query";
import { Shield, Image, Video, Music, TrendingUp } from "lucide-react";
import { jobsApi } from "@/lib/api";
import { useAuthStore } from "@/store/authStore";
import type { DetectionJob } from "@/types";

function StatCard({
  icon: Icon,
  label,
  value,
  color,
}: {
  icon: React.ElementType;
  label: string;
  value: number | string;
  color: string;
}) {
  return (
    <div className="card flex items-center gap-4">
      <div className={`flex h-12 w-12 items-center justify-center rounded-xl ${color}`}>
        <Icon className="h-6 w-6" />
      </div>
      <div>
        <p className="text-sm text-gray-400">{label}</p>
        <p className="text-2xl font-bold text-white">{value}</p>
      </div>
    </div>
  );
}

export function DashboardPage() {
  const { user } = useAuthStore();

  const { data: jobs = [] } = useQuery<DetectionJob[]>({
    queryKey: ["jobs"],
    queryFn: () => jobsApi.list({ page_size: 100 }).then((r) => r.data),
  });

  const stats = {
    total: jobs.length,
    images: jobs.filter((j) => j.media_type === "image").length,
    videos: jobs.filter((j) => j.media_type === "video").length,
    audios: jobs.filter((j) => j.media_type === "audio").length,
    fakes: jobs.filter((j) => j.result?.verdict === "fake").length,
  };

  const recentJobs = jobs.slice(0, 5);

  return (
    <div className="p-8">
      {/* Header */}
      <div className="mb-8">
        <h1 className="text-2xl font-bold text-white">
          Welcome back, {user?.username} 👋
        </h1>
        <p className="mt-1 text-sm text-gray-400">
          Here's an overview of your detection activity.
        </p>
      </div>

      {/* Stats grid */}
      <div className="mb-8 grid grid-cols-2 gap-4 xl:grid-cols-4">
        <StatCard icon={Shield}      label="Total Analyses"  value={stats.total}  color="bg-brand-600/20 text-brand-400" />
        <StatCard icon={Image}       label="Images"          value={stats.images} color="bg-blue-600/20 text-blue-400" />
        <StatCard icon={Video}       label="Videos"          value={stats.videos} color="bg-purple-600/20 text-purple-400" />
        <StatCard icon={Music}       label="Audio"           value={stats.audios} color="bg-yellow-600/20 text-yellow-400" />
      </div>

      {/* Fake ratio */}
      {stats.total > 0 && (
        <div className="card mb-8">
          <div className="flex items-center justify-between">
            <div>
              <p className="text-sm text-gray-400">Deepfakes Detected</p>
              <p className="text-3xl font-bold text-red-400">{stats.fakes}</p>
              <p className="text-xs text-gray-500">
                {((stats.fakes / stats.total) * 100).toFixed(1)}% of total analyses
              </p>
            </div>
            <TrendingUp className="h-10 w-10 text-red-400/30" />
          </div>
          {/* Progress bar */}
          <div className="mt-4 h-2 overflow-hidden rounded-full bg-gray-800">
            <div
              className="h-full rounded-full bg-red-500 transition-all duration-500"
              style={{ width: `${(stats.fakes / stats.total) * 100}%` }}
            />
          </div>
        </div>
      )}

      {/* Recent jobs */}
      <div className="card">
        <h2 className="mb-4 text-lg font-semibold text-white">Recent Analyses</h2>
        {recentJobs.length === 0 ? (
          <p className="text-center text-sm text-gray-500 py-8">
            No analyses yet. Go to{" "}
            <a href="/detect" className="text-brand-400 hover:underline">Detect</a> to get started.
          </p>
        ) : (
          <div className="divide-y divide-gray-800">
            {recentJobs.map((job) => (
              <div key={job.id} className="flex items-center justify-between py-3">
                <div className="min-w-0 flex-1">
                  <p className="truncate text-sm font-medium text-gray-100">
                    {job.original_filename}
                  </p>
                  <p className="text-xs text-gray-500">
                    {new Date(job.created_at).toLocaleString()} · {job.media_type}
                  </p>
                </div>
                <div className="ml-4">
                  {job.result ? (
                    <span className={job.result.verdict === "fake" ? "badge-fake" : "badge-real"}>
                      {job.result.verdict.toUpperCase()}
                    </span>
                  ) : (
                    <span className="badge-unknown">{job.status.toUpperCase()}</span>
                  )}
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
