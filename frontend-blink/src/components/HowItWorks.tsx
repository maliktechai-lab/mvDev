import { useRef } from 'react';
import { motion, useInView } from 'framer-motion';

// ─── Step data ────────────────────────────────────────────────
const STEPS = [
  {
    n: '01', title: 'Upload',
    body: 'Drag and drop images or videos. JPEG, PNG, WebP, MP4, WebM — up to 100MB. Every pixel matters.',
    tag: 'INPUT',
    accent: '#ffffff',
    glow:   'rgba(255,255,255,0.07)',
  },
  {
    n: '02', title: 'Spatial Analysis',
    body: 'EfficientNet-B4 scans pixel-level textures, GAN artifacts, facial blending seams, and lighting inconsistencies invisible to the human eye.',
    tag: 'BRANCH A',
    accent: '#a855f7',
    glow:   'rgba(168,85,247,0.10)',
  },
  {
    n: '03', title: 'Frequency Domain',
    body: '2D Fast Fourier Transform reveals spectral anomalies characteristic of synthetic generation — patterns that exist only in fabricated media.',
    tag: 'BRANCH B',
    accent: '#3b82f6',
    glow:   'rgba(59,130,246,0.10)',
  },
  {
    n: '04', title: 'Dual-Branch Fusion',
    body: 'Spatial and frequency outputs are combined via a learned fusion layer, weighting each branch by its confidence to produce a final score.',
    tag: 'FUSION',
    accent: '#facc15',
    glow:   'rgba(250,204,21,0.10)',
  },
  {
    n: '05', title: 'Grad-CAM++ Maps',
    body: 'Saliency maps highlight every region that influenced the decision. Transparent, human-reviewable, not a black box.',
    tag: 'EXPLAIN',
    accent: '#f97316',
    glow:   'rgba(249,115,22,0.10)',
  },
  {
    n: '06', title: 'Forensic Report',
    body: 'Structured verdict, confidence score, per-branch breakdown, frame analysis for video, and downloadable JSON evidence package.',
    tag: 'OUTPUT',
    accent: '#00ff88',
    glow:   'rgba(0,255,136,0.10)',
  },
];

// ─── Limitation data ──────────────────────────────────────────
const LIMITS = [
  {
    title: 'Unseen Generators',
    body:  'Novel synthesis methods outside training data may reduce accuracy. The model is retrained quarterly.',
    icon:  '⚠',
    accent: '#facc15',
  },
  {
    title: 'Frame Sampling',
    body:  'Video analyzed via uniform samples — brief manipulations under 0.3s may be missed.',
    icon:  '⏱',
    accent: '#3b82f6',
  },
  {
    title: 'No Audio Analysis',
    body:  'Voice cloning and audio deepfakes are outside current scope. Image and video only.',
    icon:  '🔇',
    accent: '#a855f7',
  },
  {
    title: 'Probabilistic Output',
    body:  'AI assessment — not forensic or legal evidence. Always apply human judgment.',
    icon:  '⚖',
    accent: '#ff2020',
  },
];

// ─── Step card ────────────────────────────────────────────────
function StepCard({ s, i }: { s: typeof STEPS[0]; i: number }) {
  const ref    = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: '-60px' });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 36 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ delay: i * 0.08, duration: 0.65, ease: [0.16, 1, 0.3, 1] }}
      whileHover={{ y: -5, transition: { duration: 0.2 } }}
      style={{
        position: 'relative',
        background: '#080808',
        border: '1px solid var(--border)',
        padding: '36px 32px 32px',
        overflow: 'hidden',
        cursor: 'default',
      }}
    >
      {/* Ambient glow */}
      <div style={{
        position: 'absolute', top: -50, left: -50,
        width: 200, height: 200,
        background: s.glow,
        borderRadius: '50%', filter: 'blur(60px)',
        pointerEvents: 'none',
      }} />

      {/* Top accent line — sweeps in */}
      <motion.div
        initial={{ scaleX: 0 }}
        animate={inView ? { scaleX: 1 } : {}}
        transition={{ delay: i * 0.08 + 0.2, duration: 0.55, ease: [0.16, 1, 0.3, 1] }}
        style={{
          position: 'absolute', top: 0, left: 0, right: 0, height: 2,
          background: s.accent,
          transformOrigin: 'left',
          boxShadow: `0 0 16px ${s.accent}80`,
          zIndex: 2,
        }}
      />

      {/* Ghost step number watermark */}
      <div style={{
        position: 'absolute', bottom: -20, right: 12,
        fontFamily: 'Syne', fontWeight: 800,
        fontSize: 130, lineHeight: 1,
        color: 'rgba(255,255,255,0.025)',
        letterSpacing: '-0.06em',
        userSelect: 'none', pointerEvents: 'none',
      }}>
        {s.n}
      </div>

      {/* Content */}
      <div style={{ position: 'relative', zIndex: 1 }}>
        {/* Tag */}
        <div style={{
          display: 'inline-block',
          fontFamily: 'JetBrains Mono', fontSize: 9,
          letterSpacing: '0.18em', color: 'var(--grey-3)',
          textTransform: 'uppercase',
          border: '1px solid var(--border)',
          padding: '3px 10px', marginBottom: 24,
        }}>
          {s.tag}
        </div>

        {/* Step number */}
        <div style={{
          fontFamily: 'JetBrains Mono', fontSize: 10,
          color: 'var(--grey-3)', letterSpacing: '0.1em', marginBottom: 10,
        }}>
          {s.n} /
        </div>

        {/* Title */}
        <div style={{
          fontFamily: 'Syne', fontWeight: 800,
          fontSize: 'clamp(18px, 1.8vw, 24px)',
          letterSpacing: '-0.03em', color: 'var(--white)',
          lineHeight: 1.1, marginBottom: 14,
        }}>
          {s.title}
        </div>

        {/* Body */}
        <p style={{
          fontFamily: 'Inter', fontSize: 13,
          color: 'var(--grey-2)', lineHeight: 1.75, margin: 0,
        }}>
          {s.body}
        </p>
      </div>
    </motion.div>
  );
}

// ─── Limit card ───────────────────────────────────────────────
function LimitCard({ l, i }: { l: typeof LIMITS[0]; i: number }) {
  const ref    = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, margin: '-40px' });

  return (
    <motion.div
      ref={ref}
      initial={{ opacity: 0, y: 20 }}
      animate={inView ? { opacity: 1, y: 0 } : {}}
      transition={{ delay: i * 0.1, duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
      whileHover={{ y: -4, boxShadow: `0 8px 40px ${l.accent}18`, borderColor: `${l.accent}40` }}
      style={{
        background: '#080808',
        border: '1px solid var(--border)',
        padding: '32px 28px',
        position: 'relative', overflow: 'hidden',
        transition: 'border-color 0.25s, box-shadow 0.25s',
      }}
    >
      {/* Glow blob */}
      <div style={{
        position: 'absolute', top: -40, right: -40,
        width: 160, height: 160,
        background: `${l.accent}0c`,
        borderRadius: '50%', filter: 'blur(50px)',
        pointerEvents: 'none',
      }} />

      {/* Left accent bar */}
      <motion.div
        initial={{ scaleY: 0 }}
        animate={inView ? { scaleY: 1 } : {}}
        transition={{ delay: i * 0.1 + 0.2, duration: 0.4 }}
        style={{
          position: 'absolute', top: 0, left: 0, bottom: 0, width: 2,
          background: l.accent,
          transformOrigin: 'top',
          boxShadow: `0 0 12px ${l.accent}60`,
        }}
      />

      <div style={{ position: 'relative', zIndex: 1 }}>
        {/* Icon + title row */}
        <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14 }}>
          <span style={{ fontSize: 20, lineHeight: 1 }}>{l.icon}</span>
          <div style={{
            fontFamily: 'Syne', fontWeight: 700, fontSize: 15,
            color: 'var(--white)', letterSpacing: '-0.01em',
          }}>
            {l.title}
          </div>
        </div>

        <p style={{
          fontFamily: 'Inter', fontSize: 13,
          color: 'var(--grey-2)', lineHeight: 1.7, margin: 0,
        }}>
          {l.body}
        </p>
      </div>
    </motion.div>
  );
}

// ─── Main export ──────────────────────────────────────────────
export default function HowItWorks() {
  const headRef = useRef<HTMLDivElement>(null);
  const inView  = useInView(headRef, { once: true });

  return (
    <section id="how-it-works" style={{ background: '#000', paddingBottom: 120 }}>

      {/* ── Section heading ── */}
      <div style={{ borderTop: '1px solid var(--border)', borderBottom: '1px solid var(--border)' }}>
        <motion.div
          ref={headRef}
          initial={{ opacity: 0, y: 24 }}
          animate={inView ? { opacity: 1, y: 0 } : {}}
          transition={{ duration: 0.7, ease: [0.16, 1, 0.3, 1] }}
          style={{
            maxWidth: 1320, margin: '0 auto',
            padding: '80px 40px',
            display: 'grid', gridTemplateColumns: '1fr 1fr',
            gap: 60, alignItems: 'end',
          }}
        >
          <div>
            <div style={{ fontFamily: 'JetBrains Mono', fontSize: 10, letterSpacing: '0.2em', color: 'var(--grey-2)', textTransform: 'uppercase', marginBottom: 20 }}>
              — 02 / Pipeline
            </div>
            <div style={{ fontFamily: 'Syne', fontWeight: 800, fontSize: 'clamp(52px, 8vw, 112px)', letterSpacing: '-0.045em', lineHeight: 0.88 }}>
              <span style={{ color: 'var(--white)', display: 'block' }}>HOW IT</span>
              <span style={{ WebkitTextStroke: '1px rgba(255,255,255,0.25)', color: 'transparent', display: 'block' }}>WORKS.</span>
            </div>
          </div>
          <p style={{ fontFamily: 'Inter', fontSize: 15, color: 'var(--grey-2)', lineHeight: 1.7, maxWidth: 420, alignSelf: 'end' }}>
            Six stages. Under 2 seconds. Every step is modular, auditable, and explainable — because trust in AI requires transparency.
          </p>
        </motion.div>
      </div>

      {/* ── 3×2 card grid ── */}
      <div style={{ maxWidth: 1320, margin: '0 auto', padding: '60px 40px 0' }}>
        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(3, 1fr)',
          gap: 1,
          background: 'var(--border)',
        }}>
          {STEPS.map((s, i) => <StepCard key={s.n} s={s} i={i} />)}
        </div>
      </div>

      {/* ── Architecture flow strip ── */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        whileInView={{ opacity: 1, y: 0 }}
        viewport={{ once: true }}
        transition={{ duration: 0.6 }}
        style={{ maxWidth: 1320, margin: '60px auto 0', padding: '0 40px' }}
      >
        <div style={{
          borderTop: '1px solid var(--border)', borderBottom: '1px solid var(--border)',
          padding: '24px 0', display: 'flex', alignItems: 'center', gap: 0, overflowX: 'auto',
        }}>
          {[
            { l: 'Input', dim: false }, { l: '→', dim: true },
            { l: 'Preprocess', dim: false }, { l: '→', dim: true },
            { l: 'EfficientNet-B4', dim: false }, { l: '+', dim: true },
            { l: 'FFT + CNN', dim: false }, { l: '→', dim: true },
            { l: 'Fusion', dim: false }, { l: '→', dim: true },
            { l: 'Grad-CAM++', dim: false }, { l: '→', dim: true },
            { l: 'Report', dim: false },
          ].map((item, idx) => (
            <motion.span key={idx} whileHover={!item.dim ? { color: 'var(--white)' } : {}} style={{ fontFamily: 'JetBrains Mono', fontSize: 11, letterSpacing: item.dim ? '0' : '0.06em', color: item.dim ? 'var(--grey-3)' : 'var(--grey-2)', padding: item.dim ? '0 12px' : '6px 16px', border: item.dim ? 'none' : '1px solid var(--border)', whiteSpace: 'nowrap', transition: 'color 0.2s' }}>
              {item.l}
            </motion.span>
          ))}
        </div>
      </motion.div>

      {/* ── Full Transparency / Known Limits ── */}
      <div style={{ maxWidth: 1320, margin: '80px auto 0', padding: '0 40px' }}>

        {/* Heading row */}
        <div style={{
          display: 'grid', gridTemplateColumns: '1fr 3fr', gap: 60,
          borderTop: '1px solid var(--border)', paddingTop: 60, marginBottom: 40,
        }}>
          <div>
            <div style={{ fontFamily: 'JetBrains Mono', fontSize: 10, letterSpacing: '0.2em', color: 'var(--grey-2)', textTransform: 'uppercase', marginBottom: 16 }}>
              Known Limits
            </div>
            <div style={{ fontFamily: 'Syne', fontWeight: 800, fontSize: 36, letterSpacing: '-0.04em', color: 'var(--white)', lineHeight: 1.05 }}>
              Full<br />transparency.
            </div>
          </div>
          <p style={{ fontFamily: 'Inter', fontSize: 14, color: 'var(--grey-2)', lineHeight: 1.75, maxWidth: 520, alignSelf: 'end', paddingBottom: 4 }}>
            No model is perfect. These are the known failure modes, stated plainly — because trust requires honesty, not marketing.
          </p>
        </div>

        {/* Limit cards — 2×2 grid */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1, background: 'var(--border)' }}>
          {LIMITS.map((l, i) => <LimitCard key={l.title} l={l} i={i} />)}
        </div>
      </div>
    </section>
  );
}
