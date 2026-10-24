import { useQuery } from "@tanstack/react-query";
import { motion } from "framer-motion";
import { PieChart, Pie, Cell, Tooltip, ResponsiveContainer } from "recharts";
import {
  Shield, Image, Video, Music2, TrendingUp,
  Clock, ChevronRight, Zap, AlertTriangle,
} from "lucide-react";
import { Link } from "react-router-dom";
import { jobsApi } from "@/lib/api";
import { useAuthStore } from "@/store/authStore";
import type { DetectionJob } from "@/types";

// ── Animated number ──────────────────────────────────────────────────────
function AnimCount({ value }: { value: number }) {
  return (
    <motion.span
      key={value}
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4 }}
    >
      {value}
    </motion.span>
  );
}

// ── Stat card ────────────────────────────────────────────────────────────
function StatCard({
  icon: Icon, label, value, color, delay, trend,
}: {
  icon: React.ElementType;
  label: string;
  value: number;
  color: string;
  delay: number;
  trend?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay, duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className="card-glow relative overflow-hidden"
    >
      {/* Background glow */}
      <div
        className="pointer-events-none absolute -right-6 -top-6 h-20 w-20 rounded-full opacity-20"
        style={{ background: color, filter: "blur(20px)" }}
      />

      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-medium uppercase tracking-wider text-gray-600">{label}</p>
          <p className="mt-2 text-3xl font-black text-white">
            <AnimCount value={value} />
          </p>
          {trend && (
            <p className="mt-1 text-xs" style={{ color }}>
              {trend}
            </p>
          )}
        </div>
        <motion.div
          whileHover={{ scale: 1.1, rotate: 5 }}
          className="flex h-12 w-12 items-center justify-center rounded-2xl"
          style={{ background: `${color}18`, border: `1px solid ${color}25` }}
        >
          <Icon className="h-5 w-5" style={{ color }} />
        </motion.div>
      </div>

      {/* Bottom accent bar */}
      <motion.div
        className="absolute bottom-0 left-0 h-0.5"
        initial={{ width: 0 }}
        animate={{ width: "100%" }}
        transition={{ delay: delay + 0.3, duration: 0.8 }}
        style={{ background: `linear-gradient(90deg, transparent, ${color}, transparent)` }}
      />
    </motion.div>
  );
}

// ── Custom tooltip for chart ─────────────────────────────────────────────
function ChartTooltip({ active, payload }: { active?: boolean; payload?: Array<{ name: string; value: number; payload: { color: string } }> }) {
  if (!active || !payload?.length) return null;
  const { name, value, payload: { color } } = payload[0];
  return (
    <div
      className="rounded-xl px-3 py-2 text-sm"
      style={{ background: "rgba(10,15,30,0.95)", border: `1px solid ${color}40`, color }}
    >
      <p className="font-semibold">{name}</p>
      <p className="font-mono">{value} analyses</p>
    </div>
  );
}

// ── Verdict badge (inline) ───────────────────────────────────────────────
function VBadge({ verdict }: { verdict: string | undefined }) {
  if (!verdict) return <span className="badge-unknown">–</span>;
  if (verdict === "fake") return <span className="badge-fake">FAKE</span>;
  return <span className="badge-real">REAL</span>;
}

// ── Main page ────────────────────────────────────────────────────────────
export function DashboardPage() {
  const { user } = useAuthStore();

  const { data: jobs = [] } = useQuery<DetectionJob[]>({
    queryKey: ["jobs"],
    queryFn: () => jobsApi.list({ page_size: 100 }).then((r) => r.data),
  });

  const stats = {
    total:  jobs.length,
    images: jobs.filter((j) => j.media_type === "image").length,
    videos: jobs.filter((j) => j.media_type === "video").length,
    audios: jobs.filter((j) => j.media_type === "audio").length,
    fakes:  jobs.filter((j) => j.result?.verdict === "fake").length,
    real:   jobs.filter((j) => j.result?.verdict === "real").length,
  };

  const chartData = [
    { name: "Fake",    value: stats.fakes,  color: "#ff003c" },
    { name: "Real",    value: stats.real,   color: "#00ff88" },
    { name: "Pending", value: stats.total - stats.fakes - stats.real, color: "#6366f1" },
  ].filter((d) => d.value > 0);

  const fakeRatio = stats.total > 0 ? (stats.fakes / stats.total) * 100 : 0;
  const recentJobs = jobs.slice(0, 7);

  return (
    <div
      className="min-h-full px-8 py-8"
      style={{ background: "linear-gradient(180deg, #0a0f1e 0%, #030712 100%)" }}
    >
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: -15 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
        className="mb-8"
      >
        <div className="flex items-center gap-2 text-xs text-gray-600 mb-1">
          <Zap className="h-3.5 w-3.5" />
          <span>Overview</span>
        </div>
        <h1 className="text-3xl font-bold text-white">
          Welcome back,{" "}
          <span className="gradient-text">{user?.username}</span>
        </h1>
        <p className="mt-1 text-sm text-gray-500">
          Here's your detection activity at a glance.
        </p>
      </motion.div>

      {/* Stat cards */}
      <div className="mb-8 grid grid-cols-2 gap-4 xl:grid-cols-4">
        <StatCard icon={Shield}  label="Total Analyses"  value={stats.total}  color="#6366f1" delay={0.05} />
        <StatCard icon={Image}   label="Images"          value={stats.images} color="#00f5ff" delay={0.10} />
        <StatCard icon={Video}   label="Videos"          value={stats.videos} color="#bf00ff" delay={0.15} />
        <StatCard icon={Music2}  label="Audio"           value={stats.audios} color="#00ff88" delay={0.20} />
      </div>

      <div className="grid gap-6 lg:grid-cols-5">
        {/* ── Chart card ──────────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.25, duration: 0.5 }}
          className="card-glow lg:col-span-2"
        >
          <div className="mb-1 flex items-center gap-2">
            <TrendingUp className="h-4 w-4" style={{ color: "#ff003c" }} />
            <h2 className="text-sm font-bold text-gray-200">Verdict Distribution</h2>
          </div>

          {stats.total === 0 ? (
            <div className="flex h-48 flex-col items-center justify-center gap-3">
              <Shield className="h-10 w-10 text-gray-800" />
              <p className="text-sm text-gray-700">No analyses yet</p>
              <Link to="/detect" className="btn-primary text-xs">
                Run First Analysis
              </Link>
            </div>
          ) : (
            <>
              <div style={{ height: 200 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <PieChart>
                    <Pie
                      data={chartData}
                      cx="50%"
                      cy="50%"
                      innerRadius={55}
                      outerRadius={80}
                      paddingAngle={3}
                      dataKey="value"
                      animationBegin={300}
                      animationDuration={800}
                    >
                      {chartData.map((d) => (
                        <Cell key={d.name} fill={d.color} stroke="none" />
                      ))}
                    </Pie>
                    <Tooltip content={<ChartTooltip />} />
                  </PieChart>
                </ResponsiveContainer>
              </div>

              {/* Legend */}
              <div className="mt-2 space-y-2">
                {chartData.map((d) => (
                  <div key={d.name} className="flex items-center justify-between text-sm">
                    <div className="flex items-center gap-2">
                      <div className="h-2.5 w-2.5 rounded-full" style={{ background: d.color }} />
                      <span className="text-gray-400">{d.name}</span>
                    </div>
                    <span className="font-mono font-semibold text-gray-300">{d.value}</span>
                  </div>
                ))}
              </div>

              {/* Fake ratio */}
              {fakeRatio > 0 && (
                <div className="mt-4 rounded-xl p-3" style={{ background: "rgba(255,0,60,0.06)", border: "1px solid rgba(255,0,60,0.15)" }}>
                  <div className="flex items-center gap-2 mb-1">
                    <AlertTriangle className="h-3.5 w-3.5 text-red-400" />
                    <span className="text-xs font-semibold text-red-400">
                      {fakeRatio.toFixed(1)}% deepfake rate
                    </span>
                  </div>
                  <div className="progress-bar">
                    <motion.div
                      className="progress-fill"
                      initial={{ width: 0 }}
                      animate={{ width: `${fakeRatio}%` }}
                      transition={{ delay: 0.6, duration: 1 }}
                      style={{ background: "linear-gradient(90deg, #9f1239, #ff003c)" }}
                    />
                  </div>
                </div>
              )}
            </>
          )}
        </motion.div>

        {/* ── Recent jobs ──────────────────────────────────────────── */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ delay: 0.3, duration: 0.5 }}
          className="card-glow lg:col-span-3"
        >
          <div className="mb-4 flex items-center justify-between">
            <div className="flex items-center gap-2">
              <Clock className="h-4 w-4 text-gray-500" />
              <h2 className="text-sm font-bold text-gray-200">Recent Analyses</h2>
            </div>
            <Link
              to="/history"
              className="flex items-center gap-1 text-xs transition-colors hover:text-brand-400"
              style={{ color: "#334155" }}
            >
              View all <ChevronRight className="h-3 w-3" />
            </Link>
          </div>

          {recentJobs.length === 0 ? (
            <div className="flex flex-col items-center justify-center py-12 gap-3">
              <UploadCloud className="h-10 w-10 text-gray-800" />
              <p className="text-sm text-gray-700">No analyses yet.</p>
              <Link to="/detect" className="btn-primary text-xs">
                Analyze Media
              </Link>
            </div>
          ) : (
            <div className="space-y-1">
              {recentJobs.map((job, i) => {
                const typeColor: Record<string, string> = {
                  image: "#00f5ff", video: "#bf00ff", audio: "#00ff88",
                };
                const c = typeColor[job.media_type] ?? "#6366f1";
                return (
                  <motion.div
                    key={job.id}
                    initial={{ opacity: 0, x: 20 }}
                    animate={{ opacity: 1, x: 0 }}
                    transition={{ delay: 0.35 + i * 0.05 }}
                    className="flex items-center gap-4 rounded-xl px-4 py-3 transition-all hover:bg-white/[0.02]"
                    style={{ border: "1px solid transparent" }}
                  >
                    <div
                      className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-xs font-bold"
                      style={{ background: `${c}15`, color: c, border: `1px solid ${c}25` }}
                    >
                      {job.media_type[0].toUpperCase()}
                    </div>
                    <div className="min-w-0 flex-1">
                      <p className="truncate text-sm font-medium text-gray-200">{job.original_filename}</p>
                      <p className="text-xs text-gray-600">
                        {new Date(job.created_at).toLocaleString(undefined, {
                          month: "short", day: "numeric", hour: "2-digit", minute: "2-digit",
                        })}
                      </p>
                    </div>
                    <div className="flex items-center gap-2 text-xs text-gray-600">
                      {job.result && (
                        <span className="font-mono">{(job.result.confidence * 100).toFixed(0)}%</span>
                      )}
                      <VBadge verdict={job.result?.verdict} />
                    </div>
                  </motion.div>
                );
              })}
            </div>
          )}
        </motion.div>
      </div>
    </div>
  );
}

// eslint-disable-next-line react-refresh/only-export-components
function UploadCloud({ className, ...p }: React.SVGProps<SVGSVGElement> & { className?: string }) {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round"
      className={className} {...p}>
      <polyline points="16 16 12 12 8 16"/><line x1="12" y1="12" x2="12" y2="21"/>
      <path d="M20.39 18.39A5 5 0 0 0 18 9h-1.26A8 8 0 1 0 3 16.3"/>
    </svg>
  );
}
