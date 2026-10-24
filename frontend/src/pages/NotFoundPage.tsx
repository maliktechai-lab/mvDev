import { motion } from "framer-motion";
import { Link } from "react-router-dom";
import { Shield, Home, ArrowLeft } from "lucide-react";
import { ParticleField } from "@/components/3d/ParticleField";

export function NotFoundPage() {
  return (
    <div
      className="relative flex min-h-screen flex-col items-center justify-center overflow-hidden px-4"
      style={{ background: "#030712" }}
    >
      <ParticleField />
      <div className="pointer-events-none absolute inset-0 bg-grid opacity-30" />

      <div className="relative z-10 text-center">
        {/* Glitchy 404 */}
        <motion.div
          initial={{ scale: 0.8, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          transition={{ duration: 0.7, type: "spring", stiffness: 200 }}
          className="mb-4"
        >
          <motion.h1
            className="font-black leading-none tracking-tighter"
            style={{
              fontSize: "clamp(80px, 20vw, 180px)",
              background: "linear-gradient(135deg, #4f46e5 0%, #00f5ff 50%, #bf00ff 100%)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              filter: "drop-shadow(0 0 40px rgba(99,102,241,0.4))",
            }}
            animate={{
              filter: [
                "drop-shadow(0 0 40px rgba(99,102,241,0.4))",
                "drop-shadow(0 0 80px rgba(99,102,241,0.7))",
                "drop-shadow(0 0 40px rgba(99,102,241,0.4))",
              ],
            }}
            transition={{ duration: 3, repeat: Infinity }}
          >
            404
          </motion.h1>
        </motion.div>

        {/* Logo */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="mb-5 flex items-center justify-center gap-3"
        >
          <motion.div
            animate={{ rotate: [0, 360] }}
            transition={{ duration: 20, repeat: Infinity, ease: "linear" }}
            className="flex h-10 w-10 items-center justify-center rounded-xl"
            style={{ background: "linear-gradient(135deg, #4f46e5, #7c3aed)" }}
          >
            <Shield className="h-5 w-5 text-white" />
          </motion.div>
          <span className="text-xl font-bold text-white">DeepGuard</span>
        </motion.div>

        <motion.p
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="mb-2 text-xl font-semibold text-white"
        >
          Page not found
        </motion.p>
        <motion.p
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.4 }}
          className="mb-8 text-sm text-gray-500"
        >
          The page you're looking for doesn't exist or has been moved.
        </motion.p>

        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="flex flex-col items-center gap-3 sm:flex-row sm:justify-center"
        >
          <Link to="/dashboard">
            <motion.button
              className="btn-primary"
              whileHover={{ scale: 1.04 }}
              whileTap={{ scale: 0.97 }}
            >
              <Home className="h-4 w-4" />
              Go to Dashboard
            </motion.button>
          </Link>
          <button
            onClick={() => window.history.back()}
            className="btn-secondary"
          >
            <ArrowLeft className="h-4 w-4" />
            Go Back
          </button>
        </motion.div>
      </div>
    </div>
  );
}
