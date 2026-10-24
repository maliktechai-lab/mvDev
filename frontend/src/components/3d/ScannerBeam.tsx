import { motion } from "framer-motion";

interface ScannerBeamProps {
  active?: boolean;
  color?: "cyan" | "brand" | "red" | "green";
  className?: string;
}

const colorMap = {
  cyan:  { line: "rgba(0,245,255,0.8)",   glow: "rgba(0,245,255,0.3)",   side: "rgba(0,245,255,0.2)" },
  brand: { line: "rgba(99,102,241,0.8)",  glow: "rgba(99,102,241,0.3)",  side: "rgba(99,102,241,0.2)" },
  red:   { line: "rgba(255,0,60,0.8)",    glow: "rgba(255,0,60,0.3)",    side: "rgba(255,0,60,0.2)" },
  green: { line: "rgba(0,255,136,0.8)",   glow: "rgba(0,255,136,0.3)",   side: "rgba(0,255,136,0.2)" },
};

export function ScannerBeam({ active = true, color = "cyan", className = "" }: ScannerBeamProps) {
  const c = colorMap[color];

  if (!active) return null;

  return (
    <div className={`pointer-events-none absolute inset-0 overflow-hidden rounded-2xl ${className}`}>
      {/* Corner brackets */}
      <div className="absolute inset-0">
        {/* Top-left */}
        <div className="absolute left-3 top-3 h-6 w-6"
          style={{ borderTop: `2px solid ${c.line}`, borderLeft: `2px solid ${c.line}` }} />
        {/* Top-right */}
        <div className="absolute right-3 top-3 h-6 w-6"
          style={{ borderTop: `2px solid ${c.line}`, borderRight: `2px solid ${c.line}` }} />
        {/* Bottom-left */}
        <div className="absolute bottom-3 left-3 h-6 w-6"
          style={{ borderBottom: `2px solid ${c.line}`, borderLeft: `2px solid ${c.line}` }} />
        {/* Bottom-right */}
        <div className="absolute bottom-3 right-3 h-6 w-6"
          style={{ borderBottom: `2px solid ${c.line}`, borderRight: `2px solid ${c.line}` }} />
      </div>

      {/* Horizontal scan line */}
      <motion.div
        className="absolute left-0 right-0"
        style={{
          height: 2,
          background: `linear-gradient(90deg, transparent 0%, ${c.glow} 20%, ${c.line} 50%, ${c.glow} 80%, transparent 100%)`,
          boxShadow: `0 0 15px ${c.glow}, 0 0 30px ${c.side}`,
        }}
        animate={{ y: ["0%", "100%", "0%"] }}
        transition={{ duration: 2.4, ease: "easeInOut", repeat: Infinity }}
      />

      {/* Sweep overlay */}
      <motion.div
        className="absolute left-0 right-0"
        style={{
          height: "30%",
          background: `linear-gradient(180deg, transparent 0%, ${c.side} 50%, transparent 100%)`,
        }}
        animate={{ y: ["-30%", "130%", "-30%"] }}
        transition={{ duration: 2.4, ease: "easeInOut", repeat: Infinity }}
      />

      {/* Side scan pulses */}
      <motion.div
        className="absolute top-0 bottom-0 left-0"
        style={{
          width: 2,
          background: `linear-gradient(180deg, transparent, ${c.line}, transparent)`,
          boxShadow: `0 0 8px ${c.glow}`,
        }}
        animate={{ opacity: [0.3, 1, 0.3] }}
        transition={{ duration: 1.2, repeat: Infinity }}
      />
      <motion.div
        className="absolute top-0 bottom-0 right-0"
        style={{
          width: 2,
          background: `linear-gradient(180deg, transparent, ${c.line}, transparent)`,
          boxShadow: `0 0 8px ${c.glow}`,
        }}
        animate={{ opacity: [1, 0.3, 1] }}
        transition={{ duration: 1.2, repeat: Infinity }}
      />
    </div>
  );
}

// ── Upload drop zone scan variant ────────────────────────────────────────
export function DropzoneScan({ scanning }: { scanning: boolean }) {
  return (
    <div className="pointer-events-none absolute inset-0 overflow-hidden rounded-2xl">
      {scanning && (
        <>
          {/* Animated border */}
          <motion.div
            className="absolute inset-0 rounded-2xl"
            style={{
              background: "transparent",
              border: "1px solid rgba(0,245,255,0.0)",
            }}
            animate={{
              boxShadow: [
                "0 0 0px rgba(0,245,255,0)",
                "0 0 25px rgba(0,245,255,0.3), inset 0 0 25px rgba(0,245,255,0.05)",
                "0 0 0px rgba(0,245,255,0)",
              ],
              borderColor: [
                "rgba(0,245,255,0.0)",
                "rgba(0,245,255,0.5)",
                "rgba(0,245,255,0.0)",
              ],
            }}
            transition={{ duration: 1.8, repeat: Infinity }}
          />

          {/* Scan line */}
          <motion.div
            className="absolute left-4 right-4"
            style={{
              height: 1.5,
              background:
                "linear-gradient(90deg, transparent, rgba(0,245,255,0.9), transparent)",
              boxShadow: "0 0 12px rgba(0,245,255,0.6)",
            }}
            animate={{ y: [0, 200, 0] }}
            transition={{ duration: 2, ease: "linear", repeat: Infinity }}
          />
        </>
      )}
    </div>
  );
}
