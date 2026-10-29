import { useEffect, useRef, useState, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Props {
  imageUrl?: string;
  label: 'REAL' | 'FAKE' | 'UNKNOWN';
  /** Backend GradCAM heatmap URL — when provided, display the real heatmap overlay. */
  heatmapUrl?: string;
}

interface HeatZone {
  x: number; y: number;
  radius: number; intensity: number;
  label: string; color: string;
}

const FAKE_ZONES: HeatZone[] = [
  { x: 0.38, y: 0.28, radius: 0.14, intensity: 0.92, label: 'Face boundary artifact', color: '#ff2020' },
  { x: 0.62, y: 0.22, radius: 0.10, intensity: 0.78, label: 'Eye region inconsistency', color: '#ff2020' },
  { x: 0.50, y: 0.45, radius: 0.09, intensity: 0.65, label: 'Skin texture anomaly',    color: '#f59e0b' },
  { x: 0.28, y: 0.55, radius: 0.08, intensity: 0.55, label: 'Lighting mismatch',        color: '#f59e0b' },
  { x: 0.72, y: 0.60, radius: 0.07, intensity: 0.44, label: 'Edge blending seam',       color: '#888' },
];

const REAL_ZONES: HeatZone[] = [
  { x: 0.50, y: 0.40, radius: 0.20, intensity: 0.18, label: 'Natural texture', color: '#00ff88' },
  { x: 0.30, y: 0.30, radius: 0.10, intensity: 0.12, label: 'Authentic lighting', color: '#00ff88' },
];

function drawHeatmap(
  canvas: HTMLCanvasElement,
  zones: HeatZone[],
  scanY: number,
  progress: number,
  revealed: boolean,
) {
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  // Use canvas's actual pixel dimensions (not CSS size)
  const W = canvas.width;
  const H = canvas.height;

  ctx.clearRect(0, 0, W, H);

  const drawZones = (clipH: number | null) => {
    if (clipH !== null) {
      ctx.save();
      ctx.beginPath();
      ctx.rect(0, 0, W, clipH);
      ctx.clip();
    }

    zones.forEach((zone) => {
      const cx = zone.x * W;
      const cy = zone.y * H;
      const r = zone.radius * Math.min(W, H) * progress;
      if (r <= 0) return;

      const alpha = zone.intensity * (clipH !== null ? Math.min(1, (clipH / H) / zone.y) : 1);
      const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, r);

      const [r1, g1, b1] = hexToRgb(zone.color);
      grad.addColorStop(0,   `rgba(${r1},${g1},${b1},${Math.min(1, alpha * 0.85)})`);
      grad.addColorStop(0.4, `rgba(${r1},${g1},${b1},${Math.min(1, alpha * 0.45)})`);
      grad.addColorStop(1,   `rgba(${r1},${g1},${b1},0)`);

      ctx.globalCompositeOperation = 'screen';
      ctx.fillStyle = grad;
      ctx.beginPath();
      ctx.arc(cx, cy, r, 0, Math.PI * 2);
      ctx.fill();
    });

    if (clipH !== null) ctx.restore();
  };

  drawZones(revealed ? null : scanY);

  ctx.globalCompositeOperation = 'source-over';

  // Scan beam
  if (!revealed && scanY > 0) {
    const lg = ctx.createLinearGradient(0, scanY - 4, 0, scanY + 4);
    lg.addColorStop(0, 'rgba(255,255,255,0)');
    lg.addColorStop(0.5, 'rgba(255,255,255,0.8)');
    lg.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = lg;
    ctx.fillRect(0, scanY - 4, W, 8);

    // Glow below beam
    const glow = ctx.createLinearGradient(0, scanY, 0, scanY + 24);
    glow.addColorStop(0, 'rgba(255,255,255,0.1)');
    glow.addColorStop(1, 'rgba(255,255,255,0)');
    ctx.fillStyle = glow;
    ctx.fillRect(0, scanY, W, 24);
  }

  // Grid
  ctx.strokeStyle = 'rgba(255,255,255,0.04)';
  ctx.lineWidth = 0.5;
  const gridSize = 32;
  for (let x = 0; x <= W; x += gridSize) {
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  }
  for (let y = 0; y <= H; y += gridSize) {
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(W, y); ctx.stroke();
  }
}

function hexToRgb(hex: string): [number, number, number] {
  const result = /^#?([a-f\d]{2})([a-f\d]{2})([a-f\d]{2})$/i.exec(hex);
  return result
    ? [parseInt(result[1], 16), parseInt(result[2], 16), parseInt(result[3], 16)]
    : [255, 255, 255];
}

// ─── Canvas component ─────────────────────────────────────────
function HeatmapCanvas({ label, imageUrl }: { label: 'REAL' | 'FAKE' | 'UNKNOWN'; imageUrl?: string; heatmapUrl?: string }) {
  const containerRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [scanY, setScanY] = useState(0);
  const [revealed, setRevealed] = useState(false);
  const [progress, setProgress] = useState(0);
  const [activeZone, setActiveZone] = useState<HeatZone | null>(null);
  const [scanning, setScanning] = useState(false);
  const [canvasSize, setCanvasSize] = useState({ w: 640, h: 360 });
  const animRef = useRef<number>(0);

  const zones = label === 'FAKE' ? FAKE_ZONES : REAL_ZONES;

  // Sync canvas pixel size to CSS rendered size
  useEffect(() => {
    const ro = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect;
        if (width > 0 && height > 0) {
          setCanvasSize({ w: Math.round(width), h: Math.round(height) });
        }
      }
    });
    if (containerRef.current) ro.observe(containerRef.current);
    return () => ro.disconnect();
  }, []);

  // Update canvas internal resolution
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    canvas.width  = canvasSize.w;
    canvas.height = canvasSize.h;
  }, [canvasSize]);

  // Run scan animation
  useEffect(() => {
    setRevealed(false);
    setScanY(0);
    setProgress(0);
    setScanning(false);
    cancelAnimationFrame(animRef.current);

    const t = setTimeout(() => {
      setScanning(true);
      const start = performance.now();
      const duration = 2000;

      const animate = (now: number) => {
        const p = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - p, 2.2);
        const canvas = canvasRef.current;
        if (!canvas) return;
        setScanY(eased * canvas.height);
        setProgress(eased);
        if (p < 1) {
          animRef.current = requestAnimationFrame(animate);
        } else {
          setRevealed(true);
          setScanning(false);
        }
      };
      animRef.current = requestAnimationFrame(animate);
    }, 500);

    return () => { clearTimeout(t); cancelAnimationFrame(animRef.current); };
  }, [label, imageUrl]);

  // Redraw on every state change
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    drawHeatmap(canvas, zones, scanY, progress, revealed);
  }, [scanY, progress, revealed, zones]);

  const handleMouseMove = useCallback((e: React.MouseEvent<HTMLDivElement>) => {
    if (!revealed || !containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    const mx = (e.clientX - rect.left) / rect.width;
    const my = (e.clientY - rect.top) / rect.height;
    const hit = zones.find((z) => {
      const dx = mx - z.x;
      const dy = my - z.y;
      return Math.sqrt(dx * dx + dy * dy) < z.radius * 1.4;
    });
    setActiveZone(hit ?? null);
  }, [revealed, zones]);

  return (
    <div
      ref={containerRef}
      style={{
        position: 'relative',
        width: '100%',
        aspectRatio: '16/9',
        background: '#000',
        overflow: 'hidden',
      }}
      onMouseMove={handleMouseMove}
      onMouseLeave={() => setActiveZone(null)}
    >
      {/* Base image */}
      {imageUrl ? (
        <img
          src={imageUrl}
          alt="Analyzed media"
          style={{
            position: 'absolute', inset: 0,
            width: '100%', height: '100%',
            objectFit: 'cover',
            opacity: 0.6,
          }}
        />
      ) : (
        <div style={{
          position: 'absolute', inset: 0,
          background: 'linear-gradient(135deg, #0a0a10 0%, #000 100%)',
          display: 'flex', alignItems: 'center', justifyContent: 'center',
        }}>
          <svg viewBox="0 0 200 200" width="100" height="100" opacity={0.08}>
            <ellipse cx="100" cy="80" rx="44" ry="54" fill="#fff" />
            <ellipse cx="100" cy="168" rx="58" ry="38" fill="#fff" />
            <ellipse cx="82" cy="70" rx="8" ry="10" fill="#000" />
            <ellipse cx="118" cy="70" rx="8" ry="10" fill="#000" />
            <path d="M85 96 Q100 108 115 96" stroke="#000" strokeWidth="3" fill="none" strokeLinecap="round" />
          </svg>
        </div>
      )}

      {/* Heatmap canvas — must match container exactly */}
      <canvas
        ref={canvasRef}
        style={{
          position: 'absolute', inset: 0,
          width: '100%', height: '100%',
          mixBlendMode: imageUrl ? 'screen' : 'normal',
        }}
      />

      {/* Corner brackets */}
      {(['tl','tr','bl','br'] as const).map((c) => (
        <div key={c} style={{
          position: 'absolute',
          width: 16, height: 16,
          top:    c[0] === 't' ? 10 : 'auto',
          bottom: c[0] === 'b' ? 10 : 'auto',
          left:   c[1] === 'l' ? 10 : 'auto',
          right:  c[1] === 'r' ? 10 : 'auto',
          borderTop:    c[0] === 't' ? '1px solid rgba(255,255,255,0.5)' : 'none',
          borderBottom: c[0] === 'b' ? '1px solid rgba(255,255,255,0.5)' : 'none',
          borderLeft:   c[1] === 'l' ? '1px solid rgba(255,255,255,0.5)' : 'none',
          borderRight:  c[1] === 'r' ? '1px solid rgba(255,255,255,0.5)' : 'none',
        }} />
      ))}

      {/* Scanning badge */}
      <AnimatePresence>
        {scanning && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
              position: 'absolute', top: 12, left: '50%', transform: 'translateX(-50%)',
              display: 'flex', alignItems: 'center', gap: 7,
              padding: '5px 14px',
              background: 'rgba(0,0,0,0.8)',
              border: '1px solid rgba(255,255,255,0.15)',
              fontFamily: 'JetBrains Mono',
              fontSize: 9,
              letterSpacing: '0.18em',
              color: 'rgba(255,255,255,0.7)',
            }}
          >
            <motion.div
              animate={{ opacity: [1, 0.2, 1] }}
              transition={{ repeat: Infinity, duration: 0.7 }}
              style={{ width: 4, height: 4, borderRadius: '50%', background: '#fff' }}
            />
            SCANNING
          </motion.div>
        )}
      </AnimatePresence>

      {/* Verdict badge */}
      <AnimatePresence>
        {revealed && (
          <motion.div
            initial={{ opacity: 0, scale: 0.85 }}
            animate={{ opacity: 1, scale: 1 }}
            transition={{ type: 'spring', stiffness: 300, delay: 0.15 }}
            style={{
              position: 'absolute', top: 12, right: 12,
              padding: '5px 14px',
              background: label === 'FAKE' ? 'rgba(255,32,32,0.9)' : 'rgba(0,255,136,0.9)',
              fontFamily: 'JetBrains Mono',
              fontSize: 9,
              letterSpacing: '0.18em',
              color: '#000',
              fontWeight: 700,
            }}
          >
            {label === 'FAKE' ? '⚠ SYNTHETIC' : '✓ AUTHENTIC'}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Zone tooltip */}
      <AnimatePresence>
        {activeZone && (
          <motion.div
            initial={{ opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0 }}
            style={{
              position: 'absolute', bottom: 12, left: 12, right: 12,
              padding: '8px 14px',
              background: 'rgba(0,0,0,0.88)',
              border: `1px solid ${activeZone.color}44`,
              display: 'flex', alignItems: 'center', gap: 10,
              fontFamily: 'JetBrains Mono',
              fontSize: 10,
              color: 'var(--off-white)',
            }}
          >
            <div style={{ width: 6, height: 6, borderRadius: '50%', background: activeZone.color, flexShrink: 0 }} />
            <span style={{ color: activeZone.color }}>{Math.round(activeZone.intensity * 100)}%</span>
            <span style={{ color: 'var(--grey-3)' }}>—</span>
            <span>{activeZone.label}</span>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

// ─── Main export ──────────────────────────────────────────────
export default function HeatmapViewer({ imageUrl, label, heatmapUrl }: Props) {
  const isFake = label === 'FAKE';
  const isUnknown = label === 'UNKNOWN';
  const zones = isUnknown ? [] : isFake ? FAKE_ZONES : REAL_ZONES;
  const hasRealHeatmap = !!heatmapUrl;

  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ delay: 0.3 }}
      style={{ padding: '0' }}
    >
      {/* Header */}
      <div style={{
        display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        marginBottom: 16,
      }}>
        <div>
          <div style={{ fontFamily: 'Syne', fontWeight: 700, fontSize: 15, color: 'var(--white)', marginBottom: 2 }}>
            Grad-CAM++ Heatmap
          </div>
          <div style={{ fontFamily: 'JetBrains Mono', fontSize: 10, color: 'var(--grey-2)', letterSpacing: '0.06em' }}>
            {hasRealHeatmap
              ? 'Backend-generated activation map'
              : isUnknown
              ? 'No heatmap — inconclusive result'
              : 'Indicative overlay — no backend heatmap available'}
          </div>
        </div>
        <div style={{
          fontFamily: 'JetBrains Mono', fontSize: 9,
          letterSpacing: '0.14em', color: 'var(--grey-3)',
          padding: '4px 10px', border: '1px solid var(--border)',
        }}>
          {hasRealHeatmap ? 'REAL' : 'INDICATIVE'}
        </div>
      </div>

      {/* Canvas or real heatmap image */}
      {hasRealHeatmap ? (
        // Display the actual GradCAM heatmap from the backend
        <div style={{
          position: 'relative',
          width: '100%',
          aspectRatio: '16/9',
          background: '#000',
          overflow: 'hidden',
        }}>
          {imageUrl && (
            <img
              src={imageUrl}
              alt="Analyzed media"
              style={{
                position: 'absolute', inset: 0,
                width: '100%', height: '100%',
                objectFit: 'cover',
                opacity: 0.5,
              }}
            />
          )}
          <img
            src={heatmapUrl}
            alt="GradCAM heatmap"
            style={{
              position: 'absolute', inset: 0,
              width: '100%', height: '100%',
              objectFit: 'cover',
              mixBlendMode: 'screen',
            }}
          />
          {/* Verdict badge */}
          <div style={{
            position: 'absolute', top: 12, right: 12,
            padding: '5px 14px',
            background: isFake ? 'rgba(255,32,32,0.9)' : 'rgba(0,255,136,0.9)',
            fontFamily: 'JetBrains Mono',
            fontSize: 9,
            letterSpacing: '0.18em',
            color: '#000',
            fontWeight: 700,
          }}>
            {isFake ? '⚠ SYNTHETIC' : '✓ AUTHENTIC'}
          </div>
        </div>
      ) : isUnknown ? (
        // No heatmap for unknown verdict
        <div style={{
          position: 'relative',
          width: '100%',
          aspectRatio: '16/9',
          background: 'linear-gradient(135deg, #0a0a10 0%, #000 100%)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          border: '1px solid var(--border)',
        }}>
          <div style={{ textAlign: 'center' }}>
            <div style={{
              fontFamily: 'Syne', fontSize: 24, fontWeight: 800,
              color: '#f59e0b', letterSpacing: '-0.03em',
            }}>N/A</div>
            <div style={{
              fontFamily: 'JetBrains Mono', fontSize: 10,
              color: 'var(--grey-2)', letterSpacing: '0.1em',
              textTransform: 'uppercase', marginTop: 8,
            }}>
              No heatmap — inconclusive
            </div>
          </div>
        </div>
      ) : (
        // Fall back to canvas-based indicative heatmap
        <HeatmapCanvas label={label} imageUrl={imageUrl} heatmapUrl={heatmapUrl} />
      )}

      {/* Zone list — only show for indicative mode */}
      {!hasRealHeatmap && !isUnknown && (
        <div style={{ marginTop: 20 }}>
          <div style={{
            fontFamily: 'JetBrains Mono', fontSize: 9,
            letterSpacing: '0.18em', color: 'var(--grey-3)',
            textTransform: 'uppercase', marginBottom: 12,
          }}>
            Activation Zones
          </div>
          {zones.map((zone, i) => (
            <motion.div
              key={zone.label}
              initial={{ opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              transition={{ delay: 2.6 + i * 0.1 }}
              style={{
                display: 'flex', alignItems: 'center', gap: 12,
                padding: '10px 0',
                borderBottom: '1px solid var(--border)',
              }}
            >
              <div style={{
                width: 6, height: 6, borderRadius: '50%',
                background: zone.color, flexShrink: 0,
                boxShadow: `0 0 6px ${zone.color}`,
              }} />
              <span style={{ fontFamily: 'Inter', fontSize: 12, color: 'var(--grey-1)', flex: 1 }}>
                {zone.label}
              </span>
              {/* Bar */}
              <div style={{ width: 72, height: 1, background: 'var(--grey-3)', position: 'relative' }}>
                <motion.div
                  initial={{ width: 0 }}
                  animate={{ width: `${zone.intensity * 100}%` }}
                  transition={{ delay: 2.8 + i * 0.1, duration: 0.5 }}
                  style={{ position: 'absolute', top: 0, left: 0, height: '100%', background: zone.color }}
                />
              </div>
              <span style={{
                fontFamily: 'JetBrains Mono', fontSize: 10,
                color: zone.color, minWidth: 32, textAlign: 'right',
              }}>
                {Math.round(zone.intensity * 100)}%
              </span>
            </motion.div>
          ))}
        </div>
      )}
    </motion.div>
  );
}
