import { useRef, useState, useEffect, useCallback } from 'react';
import { motion, useScroll, useTransform, AnimatePresence } from 'framer-motion';
import { IMAGES, GALLERY_LABELS } from '../assets/images';

// ─── Slice tear canvas ─────────────────────────────────────────
// Draws the image sliced into N horizontal strips that drift apart
// and reveals detection heatmap zones underneath
interface SliceCanvasProps {
  src: string;
  label: string;
  pct: number;
  color: string;
  active: boolean;
}

const SLICE_COUNT = 14;

function SliceCanvas({ src, label, active }: SliceCanvasProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imgRef = useRef<HTMLImageElement | null>(null);
  const imgLoadedRef = useRef(false);
  const progressRef = useRef(0); // 0 = closed, 1 = fully torn
  const rafRef = useRef<number>(0);

  // Preload image
  useEffect(() => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = src;
    img.onload = () => { imgRef.current = img; imgLoadedRef.current = true; };
    imgRef.current = img;
  }, [src]);

  // Animation loop
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const target = active ? 1 : 0;
    const speed = active ? 0.032 : 0.055; // open slower, close faster

    const animate = () => {
      const diff = target - progressRef.current;
      if (Math.abs(diff) > 0.002) {
        progressRef.current += diff * speed;
      } else {
        progressRef.current = target;
      }
      draw(canvas);
      rafRef.current = requestAnimationFrame(animate);
    };

    rafRef.current = requestAnimationFrame(animate);
    return () => cancelAnimationFrame(rafRef.current);
  }, [active, src]);

  // Sync canvas to CSS size
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ro = new ResizeObserver(() => {
      canvas.width = canvas.offsetWidth;
      canvas.height = canvas.offsetHeight;
    });
    ro.observe(canvas);
    return () => ro.disconnect();
  }, []);

  const draw = useCallback((canvas: HTMLCanvasElement) => {
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const W = canvas.width;
    const H = canvas.height;
    if (W === 0 || H === 0) return;

    const p = progressRef.current; // 0–1
    ctx.clearRect(0, 0, W, H);

    // ── Base background ──
    ctx.fillStyle = '#050505';
    ctx.fillRect(0, 0, W, H);

    const img = imgRef.current;
    const sliceH = H / SLICE_COUNT;

    if (imgLoadedRef.current && img) {
      // ── Draw image in slices — each drifts by a different amount ──
      for (let i = 0; i < SLICE_COUNT; i++) {
        const sy = i * sliceH; // destination Y

        // Each slice has a unique drift — odd/even go opposite directions
        // amount increases toward center slices for organic tear feel
        const centerDist = Math.abs(i - SLICE_COUNT / 2) / (SLICE_COUNT / 2);
        const driftMag = (1 - centerDist * 0.4) * W * 0.18;
        const dir = i % 2 === 0 ? 1 : -1;
        const drift = dir * driftMag * p;

        // Vertical gap between slices grows with p
        const gap = p * 3.5;
        const dy = sy + gap * (i - SLICE_COUNT / 2);

        // Source rect on image
        const srcY = (i / SLICE_COUNT) * img.naturalHeight;
        const srcH = img.naturalHeight / SLICE_COUNT;

        ctx.save();
        ctx.globalAlpha = 1 - p * 0.15; // slight fade as tears
        ctx.drawImage(img, 0, srcY, img.naturalWidth, srcH, drift, dy, W, sliceH + 1);
        ctx.restore();
      }

      // ── Heatmap zones — revealed as slices tear ──
      if (p > 0.15) {
        const heatAlpha = Math.min(1, (p - 0.15) / 0.5);
        const isFake = label === 'DEEPFAKE';

        // Zone definitions
        const zones = isFake
          ? [
              { x: 0.38, y: 0.25, r: 0.18, a: 0.9, c: [255, 20,  20] },
              { x: 0.62, y: 0.20, r: 0.12, a: 0.75, c: [255, 20,  20] },
              { x: 0.50, y: 0.44, r: 0.10, a: 0.60, c: [245, 158, 11] },
              { x: 0.25, y: 0.58, r: 0.08, a: 0.45, c: [245, 158, 11] },
              { x: 0.74, y: 0.62, r: 0.07, a: 0.35, c: [150, 150, 0]  },
            ]
          : [
              { x: 0.50, y: 0.40, r: 0.22, a: 0.25, c: [0, 255, 136] },
              { x: 0.30, y: 0.28, r: 0.12, a: 0.18, c: [0, 255, 136] },
            ];

        ctx.globalCompositeOperation = 'screen';
        zones.forEach(({ x, y, r, a, c }) => {
          const cx = x * W;
          const cy = y * H;
          const radius = r * Math.min(W, H);
          const grad = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
          const alpha = a * heatAlpha;
          grad.addColorStop(0,   `rgba(${c[0]},${c[1]},${c[2]},${Math.min(1, alpha * 0.9)})`);
          grad.addColorStop(0.4, `rgba(${c[0]},${c[1]},${c[2]},${Math.min(1, alpha * 0.5)})`);
          grad.addColorStop(1,   `rgba(${c[0]},${c[1]},${c[2]},0)`);
          ctx.fillStyle = grad;
          ctx.beginPath();
          ctx.arc(cx, cy, radius, 0, Math.PI * 2);
          ctx.fill();
        });
        ctx.globalCompositeOperation = 'source-over';
      }

      // ── Scan line sweep ──
      if (p > 0 && p < 0.98) {
        const scanY = p * H * 1.1;
        ctx.globalCompositeOperation = 'source-over';
        const sg = ctx.createLinearGradient(0, scanY - 3, 0, scanY + 3);
        sg.addColorStop(0,   'rgba(255,255,255,0)');
        sg.addColorStop(0.5, `rgba(255,255,255,${0.7 * p})`);
        sg.addColorStop(1,   'rgba(255,255,255,0)');
        ctx.fillStyle = sg;
        ctx.fillRect(0, scanY - 3, W, 6);
      }

      // ── Glitch noise bands ── occasional horizontal distortion
      if (p > 0.3 && p < 0.85) {
        const numBands = Math.floor(p * 3);
        ctx.globalCompositeOperation = 'source-over';
        for (let b = 0; b < numBands; b++) {
          const bandY = ((b + 1) / (numBands + 1)) * H;
          const bandH = 2 + Math.random() * 4;
          ctx.fillStyle = `rgba(255,255,255,${0.03 * p})`;
          ctx.fillRect(0, bandY, W, bandH);
        }
      }
    }

    // ── Grid overlay ──
    ctx.globalCompositeOperation = 'source-over';
    ctx.strokeStyle = `rgba(255,255,255,${0.04 + p * 0.04})`;
    ctx.lineWidth = 0.5;
    const g = 28;
    for (let x = 0; x <= W; x += g) { ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }
    for (let y = 0; y <= H; y += g) { ctx.beginPath(); ctx.moveTo(0,y); ctx.lineTo(W,y); ctx.stroke(); }

    // ── Vignette ──
    const vig = ctx.createRadialGradient(W/2, H/2, H*0.2, W/2, H/2, H*0.85);
    vig.addColorStop(0, 'rgba(0,0,0,0)');
    vig.addColorStop(1, 'rgba(0,0,0,0.7)');
    ctx.fillStyle = vig;
    ctx.fillRect(0, 0, W, H);

    // ── Scanlines texture ──
    ctx.fillStyle = 'rgba(0,0,0,0)';
    for (let y = 0; y < H; y += 4) {
      ctx.fillStyle = 'rgba(0,0,0,0.08)';
      ctx.fillRect(0, y, W, 1);
    }

  }, [label]);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'absolute', inset: 0,
        width: '100%', height: '100%',
        display: 'block',
      }}
    />
  );
}

// ─── Single card ───────────────────────────────────────────────
function GalleryCard({ src, label, pct, color, index }: {
  src: string; label: string; pct: number; color: string; index: number;
}) {
  const [hovered, setHovered] = useState(false);
  // Show stamp only after tear is ~80% complete (~0.5s after hover)
  const [stampVisible, setStampVisible] = useState(false);
  const stampTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const handleEnter = () => {
    setHovered(true);
    // Delay stamp until slices have torn apart
    stampTimerRef.current = setTimeout(() => setStampVisible(true), 520);
  };

  const handleLeave = () => {
    setHovered(false);
    setStampVisible(false);
    if (stampTimerRef.current) clearTimeout(stampTimerRef.current);
  };

  return (
    <motion.div
      initial={{ opacity: 0, y: 28 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: '-60px' }}
      transition={{ delay: index * 0.07, duration: 0.55, ease: [0.16, 1, 0.3, 1] }}
      onMouseEnter={handleEnter}
      onMouseLeave={handleLeave}
      style={{
        position: 'relative',
        flexShrink: 0,
        width: 'clamp(200px, 20vw, 280px)',
        aspectRatio: '3/4',
        overflow: 'hidden',
        background: '#050505',
        isolation: 'isolate',
      }}
    >
      {/* Layer 0 — Slice-tear canvas (z-index 0) */}
      <div style={{ position: 'absolute', inset: 0, zIndex: 0 }}>
        <SliceCanvas
          src={src}
          label={label}
          pct={pct}
          color={color}
          active={hovered}
        />
      </div>

      {/* Layer 1 — Corner brackets (z-index 2) */}
      {(['tl','tr','bl','br'] as const).map((c) => (
        <motion.div
          key={c}
          animate={{ opacity: hovered ? 0.8 : 0.2, scale: hovered ? 1 : 0.85 }}
          transition={{ duration: 0.3 }}
          style={{
            position: 'absolute',
            width: 14, height: 14,
            top:    c[0] === 't' ? 10 : 'auto',
            bottom: c[0] === 'b' ? 10 : 'auto',
            left:   c[1] === 'l' ? 10 : 'auto',
            right:  c[1] === 'r' ? 10 : 'auto',
            borderTop:    c[0] === 't' ? `1px solid ${color}` : 'none',
            borderBottom: c[0] === 'b' ? `1px solid ${color}` : 'none',
            borderLeft:   c[1] === 'l' ? `1px solid ${color}` : 'none',
            borderRight:  c[1] === 'r' ? `1px solid ${color}` : 'none',
            zIndex: 2,
            pointerEvents: 'none',
          }}
        />
      ))}

      {/* Layer 2 — Verdict stamp (z-index 10, above everything, appears after tear) */}
      <AnimatePresence>
        {stampVisible && (
          <motion.div
            key="stamp"
            initial={{ opacity: 0, scale: 1.15 }}
            animate={{ opacity: 1, scale: 1 }}
            exit={{ opacity: 0, scale: 0.9, transition: { duration: 0.15 } }}
            transition={{ duration: 0.3, ease: [0.16, 1, 0.3, 1] }}
            style={{
              position: 'absolute',
              top: 0, left: 0, right: 0, bottom: 0,
              zIndex: 10,
              pointerEvents: 'none',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            {/* Dark blur backdrop so stamp is always readable */}
            <div style={{
              position: 'absolute', inset: 0,
              background: 'rgba(0,0,0,0.45)',
              backdropFilter: 'blur(1px)',
            }} />

            {/* Stamp box */}
            <motion.div
              initial={{ rotate: label === 'DEEPFAKE' ? -6 : 4 }}
              animate={{ rotate: label === 'DEEPFAKE' ? -2 : 1.5 }}
              transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
              style={{
                position: 'relative',
                padding: '10px 20px',
                border: `2px solid ${color}`,
                background: `${color}14`,
                textAlign: 'center',
              }}
            >
              {/* Top scan line on stamp */}
              <motion.div
                initial={{ scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ duration: 0.35, delay: 0.05 }}
                style={{
                  position: 'absolute', top: -2, left: 0, right: 0,
                  height: 2, background: color,
                  transformOrigin: 'left',
                  boxShadow: `0 0 8px ${color}`,
                }}
              />
              {/* Bottom scan line */}
              <motion.div
                initial={{ scaleX: 0 }}
                animate={{ scaleX: 1 }}
                transition={{ duration: 0.35, delay: 0.1 }}
                style={{
                  position: 'absolute', bottom: -2, left: 0, right: 0,
                  height: 2, background: color,
                  transformOrigin: 'right',
                  boxShadow: `0 0 8px ${color}`,
                }}
              />

              <div style={{
                fontFamily: 'Syne',
                fontWeight: 800,
                fontSize: 'clamp(18px, 2.8vw, 26px)',
                letterSpacing: '0.08em',
                color: color,
                lineHeight: 1,
                textShadow: `0 0 24px ${color}90`,
              }}>
                {label}
              </div>
              <div style={{
                fontFamily: 'JetBrains Mono',
                fontSize: 9,
                letterSpacing: '0.18em',
                color: `${color}cc`,
                marginTop: 5,
              }}>
                {pct}% CONFIDENCE
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Layer 3 — Bottom meta bar (z-index 3) */}
      <div style={{
        position: 'absolute', bottom: 0, left: 0, right: 0,
        padding: '28px 14px 12px',
        background: 'linear-gradient(to bottom, transparent, rgba(0,0,0,0.92))',
        zIndex: 3,
        pointerEvents: 'none',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 6 }}>
          <span style={{
            fontFamily: 'JetBrains Mono', fontSize: 9,
            letterSpacing: '0.14em', color: 'var(--grey-3)',
          }}>
            SAMPLE {String(index + 1).padStart(2, '0')}
          </span>
          <motion.span
            animate={{ color: hovered ? color : 'var(--grey-3)' }}
            style={{ fontFamily: 'JetBrains Mono', fontSize: 9, letterSpacing: '0.1em' }}
          >
            {hovered ? `${pct}%` : '——'}
          </motion.span>
        </div>
        <div style={{ height: 1, background: 'var(--grey-3)' }}>
          <motion.div
            animate={{ width: hovered ? `${pct}%` : '0%', backgroundColor: hovered ? color : 'var(--grey-3)' }}
            transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
            style={{ height: '100%' }}
          />
        </div>
      </div>

      {/* Index number */}
      <div style={{
        position: 'absolute', top: 12, left: 14, zIndex: 3,
        fontFamily: 'JetBrains Mono', fontSize: 9,
        color: 'rgba(255,255,255,0.2)', letterSpacing: '0.1em',
        pointerEvents: 'none',
      }}>
        {String(index + 1).padStart(2, '0')}
      </div>
    </motion.div>
  );
}

// ─── Main section ──────────────────────────────────────────────
export default function SampleGallery() {
  const sectionRef = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({
    target: sectionRef,
    offset: ['start end', 'end start'],
  });
  const x = useTransform(scrollYProgress, [0, 1], ['0%', '-8%']);

  return (
    <section
      ref={sectionRef}
      style={{
        background: '#000',
        borderTop: '1px solid var(--border)',
        borderBottom: '1px solid var(--border)',
        padding: '80px 0',
        overflow: 'hidden',
      }}
    >
      {/* Header */}
      <div style={{
        maxWidth: 1320, margin: '0 auto',
        padding: '0 40px 48px',
        display: 'grid', gridTemplateColumns: '1fr auto',
        gap: 40, alignItems: 'flex-end',
      }}>
        <div>
          <div style={{
            fontFamily: 'JetBrains Mono', fontSize: 10,
            letterSpacing: '0.2em', color: 'var(--grey-2)',
            textTransform: 'uppercase', marginBottom: 14,
          }}>
            — Sample Analysis
          </div>
          <div style={{
            fontFamily: 'Syne', fontWeight: 800,
            fontSize: 'clamp(32px, 5vw, 64px)',
            letterSpacing: '-0.04em', color: 'var(--white)', lineHeight: 0.92,
          }}>
            Every fake<br />
            <span style={{ WebkitTextStroke: '1px rgba(255,255,255,0.3)', color: 'transparent' }}>
              leaves a trace.
            </span>
          </div>
        </div>
        <div style={{
          fontFamily: 'JetBrains Mono', fontSize: 10,
          color: 'var(--grey-3)', letterSpacing: '0.1em',
          textAlign: 'right', lineHeight: 2,
        }}>
          HOVER TO REVEAL<br />
          {IMAGES.gallery.length} SAMPLES
        </div>
      </div>

      {/* Scroll-parallax track */}
      <div style={{ paddingLeft: 40, overflow: 'visible' }}>
        <motion.div style={{ x, display: 'flex', gap: 12, width: 'max-content' }}>
          {IMAGES.gallery.map((src, i) => (
            <GalleryCard
              key={i}
              src={src}
              label={GALLERY_LABELS[i].label}
              pct={GALLERY_LABELS[i].pct}
              color={GALLERY_LABELS[i].color}
              index={i}
            />
          ))}
        </motion.div>
      </div>
    </section>
  );
}
