import { useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import UploadZone from './UploadZone';
import AnalysisResultCard from './AnalysisResult';
import BranchScoresPanel from './BranchScores';
import HeatmapViewer from './HeatmapViewer';
import FrameResults from './FrameResults';
import FrameSampling from './FrameSampling';
import ForensicReport from './ForensicReport';
import AuthModal from './AuthModal';
import { useAppStore } from '../store/useAppStore';
import { analyzeMedia, isLoggedIn, getErrorMessage } from '../api/realApi';

const STAGES = [
  { key: 'upload',    label: 'Upload',    color: '#fff' },
  { key: 'decode',    label: 'Decode',    color: '#888' },
  { key: 'spatial',   label: 'Spatial',   color: '#fff' },
  { key: 'frequency', label: 'Frequency', color: '#fff' },
  { key: 'fusion',    label: 'Fusion',    color: '#fff' },
  { key: 'gradcam',   label: 'Grad-CAM',  color: '#fff' },
];

// ─── Raw pipeline — horizontal text flow ─────────────────────
function RawPipeline({ stage, pct }: { stage: string; pct: number }) {
  const cur = Math.max(0, STAGES.findIndex((s) => stage.toLowerCase().includes(s.key)));

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      exit={{ opacity: 0 }}
      style={{ padding: '32px 0' }}
    >
      {/* Progress bar — raw */}
      <div style={{ height: 1, background: 'var(--grey-3)', marginBottom: 32, position: 'relative', overflow: 'hidden' }}>
        <motion.div
          animate={{ width: `${pct}%` }}
          transition={{ duration: 0.4, ease: 'easeOut' }}
          style={{
            position: 'absolute', top: 0, left: 0, bottom: 0,
            background: 'var(--white)',
            boxShadow: '0 0 12px rgba(255,255,255,0.6)',
          }}
        />
      </div>

      {/* Stage flow */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 0 }}>
        {STAGES.map((s, i) => {
          const done = i < cur;
          const running = i === cur;
          return (
            <div key={s.key} style={{ display: 'flex', alignItems: 'center', flex: 1 }}>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                <span style={{
                  fontFamily: 'JetBrains Mono',
                  fontSize: 9,
                  letterSpacing: '0.14em',
                  textTransform: 'uppercase',
                  color: done ? 'var(--grey-2)' : running ? 'var(--white)' : 'var(--grey-3)',
                  transition: 'color 0.3s',
                }}>
                  {done ? '✓' : running ? '▶' : '○'} {s.label}
                </span>
                {running && (
                  <motion.div
                    animate={{ opacity: [1, 0.3, 1] }}
                    transition={{ repeat: Infinity, duration: 0.7 }}
                    style={{ height: 1, background: 'var(--white)', width: '100%' }}
                  />
                )}
              </div>
              {i < STAGES.length - 1 && (
                <div style={{
                  flex: 1, height: 1,
                  background: i < cur ? 'var(--grey-2)' : 'var(--grey-3)',
                  margin: '0 12px', marginBottom: 8,
                  transition: 'background 0.4s',
                }} />
              )}
            </div>
          );
        })}
      </div>

      {/* Percentage */}
      <div style={{ marginTop: 24, display: 'flex', alignItems: 'baseline', gap: 8 }}>
        <motion.span
          animate={{ opacity: [1, 0.4, 1] }}
          transition={{ repeat: Infinity, duration: 0.9 }}
          style={{
            fontFamily: 'Syne',
            fontSize: 48,
            fontWeight: 800,
            color: 'var(--white)',
            letterSpacing: '-0.04em',
            lineHeight: 1,
          }}
        >
          {pct}
        </motion.span>
        <span style={{ fontFamily: 'JetBrains Mono', fontSize: 14, color: 'var(--grey-2)' }}>%</span>
        <span style={{ fontFamily: 'JetBrains Mono', fontSize: 11, color: 'var(--grey-3)', letterSpacing: '0.1em', marginLeft: 8 }}>
          ANALYZING
        </span>
      </div>
    </motion.div>
  );
}

// ─── Full-screen verdict flash ─────────────────────────────────
function VerdictFlash({ label, onDone }: { label: 'REAL' | 'FAKE' | 'UNKNOWN'; onDone: () => void }) {
  const isFake = label === 'FAKE';
  const isUnknown = label === 'UNKNOWN';
  const accent = isFake ? '#ff2020' : isUnknown ? '#f59e0b' : '#00ff88';
  const accentDim = isFake ? 'rgba(255,32,32,0.6)' : isUnknown ? 'rgba(245,158,11,0.6)' : 'rgba(0,255,136,0.6)';

  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: [0, 1, 1, 1, 0] }}
      transition={{ duration: 1.6, times: [0, 0.08, 0.5, 0.88, 1] }}
      onAnimationComplete={onDone}
      style={{
        position: 'fixed', inset: 0, zIndex: 9000,
        background: '#000',
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        pointerEvents: 'none',
      }}
    >
      {/* RGB split layers */}
      <motion.div
        initial={{ x: 0 }}
        animate={{ x: [-8, 8, -4, 0] }}
        transition={{ duration: 0.3, repeat: 2 }}
        style={{
          position: 'absolute',
          fontFamily: 'Syne',
          fontSize: 'clamp(100px, 20vw, 240px)',
          fontWeight: 800,
          letterSpacing: '-0.05em',
          color: isUnknown ? 'rgba(245,158,11,0.6)' : 'rgba(255,0,40,0.6)',
          userSelect: 'none',
        }}
      >{label}</motion.div>
      <motion.div
        initial={{ x: 0 }}
        animate={{ x: [8, -8, 4, 0] }}
        transition={{ duration: 0.3, repeat: 2, delay: 0.05 }}
        style={{
          position: 'absolute',
          fontFamily: 'Syne',
          fontSize: 'clamp(100px, 20vw, 240px)',
          fontWeight: 800,
          letterSpacing: '-0.05em',
          color: isUnknown ? 'rgba(245,158,11,0.6)' : 'rgba(0,120,255,0.6)',
          userSelect: 'none',
        }}
      >{label}</motion.div>

      {/* Main text */}
      <motion.div
        initial={{ scale: 0.7, opacity: 0 }}
        animate={{ scale: [0.7, 1.05, 1], opacity: [0, 1, 1] }}
        transition={{ duration: 0.5, times: [0, 0.6, 1] }}
        style={{ textAlign: 'center', position: 'relative' }}
      >
        <div style={{
          fontFamily: 'Syne',
          fontSize: 'clamp(100px, 20vw, 240px)',
          fontWeight: 800,
          letterSpacing: '-0.05em',
          color: accent,
          lineHeight: 0.85,
          textShadow: isUnknown
            ? '0 0 120px rgba(245,158,11,0.5)'
            : isFake
            ? '0 0 120px rgba(255,32,32,0.5)'
            : '0 0 120px rgba(0,255,136,0.5)',
        }}>
          {label}
        </div>
        <motion.div
          initial={{ scaleX: 0 }}
          animate={{ scaleX: 1 }}
          transition={{ delay: 0.4, duration: 0.35 }}
          style={{
            height: 2,
            background: accent,
            transformOrigin: 'left',
            marginTop: 16,
          }}
        />
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ delay: 0.6 }}
          style={{
            fontFamily: 'JetBrains Mono',
            fontSize: 11,
            letterSpacing: '0.22em',
            color: accentDim,
            marginTop: 12,
            textTransform: 'uppercase',
          }}
        >
          {isUnknown
            ? 'Inconclusive — insufficient evidence'
            : isFake
            ? 'Synthetic content detected'
            : 'Authentic media verified'}
        </motion.div>
      </motion.div>

      {/* Scanlines */}
      <div style={{
        position: 'absolute', inset: 0,
        backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(0,0,0,0.2) 3px, rgba(0,0,0,0.2) 4px)',
        pointerEvents: 'none',
      }} />
    </motion.div>
  );
}

// ─── Main ─────────────────────────────────────────────────────
export default function AnalyzerSection() {
  const { status, uploadedFile, previewUrl, result, error, setFile, setStatus, setResult, setError, reset } = useAppStore();
  const [prog, setProg] = useState({ stage: '', pct: 0 });
  const [showFlash, setShowFlash] = useState(false);
  const [flashDone, setFlashDone] = useState(false);
  const [authOpen, setAuthOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  const analyze = async () => {
    if (!uploadedFile) return;

    // Gate on authentication — backend requires JWT on all detect endpoints
    if (!isLoggedIn()) {
      setAuthOpen(true);
      return;
    }

    setStatus('analyzing');
    setFlashDone(false);
    try {
      const r = await analyzeMedia(uploadedFile, (stage, pct) => setProg({ stage, pct }));
      setResult(r);
      setShowFlash(true);
    } catch (err) {
      setError(getErrorMessage(err));
    }
  };

  const analyzing = status === 'analyzing';
  const complete = status === 'complete' && flashDone;

  return (
    <section id="detector" ref={ref} style={{ background: '#000', paddingBottom: 120 }}>
      {/* Auth modal */}
      <AuthModal
        open={authOpen}
        onClose={() => setAuthOpen(false)}
        onAuthed={() => { setAuthOpen(false); analyze(); }}
      />

      {/* Verdict flash */}
      <AnimatePresence>
        {showFlash && result && !flashDone && (
          <VerdictFlash label={result.label} onDone={() => { setShowFlash(false); setFlashDone(true); }} />
        )}
      </AnimatePresence>

      {/* ── Section heading — split panel ── */}
      <div style={{ borderTop: '1px solid var(--border)', borderBottom: '1px solid var(--border)', overflow: 'hidden' }}>
        <div style={{ maxWidth: 1320, margin: '0 auto', display: 'grid', gridTemplateColumns: '1fr 380px', minHeight: 340 }}>

          {/* Left — typography + stats */}
          <motion.div
            initial={{ opacity: 0, x: -32 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            transition={{ duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
            style={{
              padding: '72px 40px',
              borderRight: '1px solid var(--border)',
              display: 'flex', flexDirection: 'column', justifyContent: 'space-between',
            }}
          >
            <div style={{
              fontFamily: 'JetBrains Mono', fontSize: 10,
              letterSpacing: '0.2em', color: 'var(--grey-2)',
              textTransform: 'uppercase', marginBottom: 28,
            }}>
              — 01 / Live Detector
            </div>

            <div style={{
              fontFamily: 'Syne', fontWeight: 800,
              fontSize: 'clamp(52px, 7vw, 104px)',
              letterSpacing: '-0.045em', lineHeight: 0.88,
            }}>
              <span style={{ color: 'var(--white)', display: 'block' }}>ANALYZE</span>
              <span style={{ WebkitTextStroke: '1px rgba(255,255,255,0.22)', color: 'transparent', display: 'block' }}>
                MEDIA.
              </span>
            </div>

            <div>
              <p style={{
                fontFamily: 'Inter', fontSize: 14, color: 'var(--grey-2)',
                lineHeight: 1.75, maxWidth: 400, marginBottom: 32,
              }}>
                Upload an image or video. Dual-branch AI returns a verdict,
                confidence score, and full visual explanation in under 2 seconds.
              </p>

              {/* Stat strip */}
              <div style={{ display: 'flex', gap: 40 }}>
                {[
                  { val: '99.2%', label: 'Accuracy' },
                  { val: '<2s',   label: 'Inference' },
                  { val: '2',     label: 'Branches' },
                ].map(({ val, label }, i) => (
                  <motion.div
                    key={label}
                    initial={{ opacity: 0, y: 12 }}
                    whileInView={{ opacity: 1, y: 0 }}
                    viewport={{ once: true }}
                    transition={{ delay: 0.4 + i * 0.1 }}
                  >
                    <div style={{
                      fontFamily: 'Syne', fontWeight: 800, fontSize: 24,
                      color: 'var(--white)', letterSpacing: '-0.03em',
                    }}>{val}</div>
                    <div style={{
                      fontFamily: 'JetBrains Mono', fontSize: 9,
                      letterSpacing: '0.14em', color: 'var(--grey-3)',
                      textTransform: 'uppercase', marginTop: 4,
                    }}>{label}</div>
                  </motion.div>
                ))}
              </div>
            </div>
          </motion.div>

          {/* Right — forensic face panel */}
          <motion.div
            initial={{ opacity: 0 }}
            whileInView={{ opacity: 1 }}
            viewport={{ once: true }}
            transition={{ duration: 1, ease: 'easeOut' }}
            style={{ position: 'relative', overflow: 'hidden', background: '#050505' }}
          >
            {/* Face image */}
            <img
              src="https://images.unsplash.com/photo-1590086782957-93c06ef21604?w=600&q=80&fit=crop&crop=faces,top"
              alt=""
              style={{
                position: 'absolute', inset: 0,
                width: '100%', height: '100%',
                objectFit: 'cover', objectPosition: 'center top',
                filter: 'grayscale(40%) contrast(1.1) brightness(0.65)',
              }}
            />

            {/* Red sweeping scan beam */}
            <motion.div
              animate={{ top: ['-2%', '104%'] }}
              transition={{ repeat: Infinity, duration: 3.2, ease: 'linear', repeatDelay: 1.4 }}
              style={{
                position: 'absolute', left: 0, right: 0, height: 2, zIndex: 3,
                background: 'linear-gradient(90deg, transparent 0%, rgba(255,20,20,0.9) 40%, rgba(255,20,20,0.9) 60%, transparent 100%)',
                boxShadow: '0 0 16px rgba(255,20,20,0.5)',
                pointerEvents: 'none',
              }}
            />

            {/* Forensic corner brackets */}
            {(['tl','tr','bl','br'] as const).map((c) => (
              <div key={c} style={{
                position: 'absolute', width: 20, height: 20, zIndex: 4,
                top:    c[0] === 't' ? 16 : 'auto',
                bottom: c[0] === 'b' ? 16 : 'auto',
                left:   c[1] === 'l' ? 16 : 'auto',
                right:  c[1] === 'r' ? 16 : 'auto',
                borderTop:    c[0] === 't' ? '1px solid rgba(255,20,20,0.7)' : 'none',
                borderBottom: c[0] === 'b' ? '1px solid rgba(255,20,20,0.7)' : 'none',
                borderLeft:   c[1] === 'l' ? '1px solid rgba(255,20,20,0.7)' : 'none',
                borderRight:  c[1] === 'r' ? '1px solid rgba(255,20,20,0.7)' : 'none',
              }} />
            ))}

            {/* SUBJECT UNKNOWN badge */}
            <div style={{
              position: 'absolute', bottom: 16, left: 16, zIndex: 5,
              fontFamily: 'JetBrains Mono', fontSize: 9,
              letterSpacing: '0.18em', color: 'rgba(255,20,20,0.9)',
              textTransform: 'uppercase',
              background: 'rgba(0,0,0,0.7)',
              padding: '4px 10px',
              border: '1px solid rgba(255,20,20,0.35)',
            }}>
              SUBJECT UNKNOWN
            </div>

            {/* Overlays */}
            <div style={{ position: 'absolute', inset: 0, zIndex: 2, background: 'linear-gradient(to right, rgba(0,0,0,0.45), transparent 40%)' }} />
            <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0, height: '30%', zIndex: 2, background: 'linear-gradient(to top, #000, transparent)' }} />
            <div style={{
              position: 'absolute', inset: 0, zIndex: 3, pointerEvents: 'none',
              backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.07) 2px, rgba(0,0,0,0.07) 4px)',
            }} />
          </motion.div>
        </div>
      </div>

      {/* ── Upload area — full width, raw ── */}
      <div style={{ maxWidth: 1320, margin: '0 auto', padding: '60px 40px 0' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 60, alignItems: 'start' }}>

          {/* Left — upload */}
          <div>
            <div className="upload-zone" data-cursor="upload" data-cursor-label="DROP">
              <UploadZone onFileSelected={setFile} isAnalyzing={analyzing} />
            </div>

            {/* Analyze button */}
            <AnimatePresence>
              {uploadedFile && !analyzing && status !== 'complete' && (
                <motion.button
                  initial={{ opacity: 0, y: 8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  onClick={analyze}
                  data-cursor="scan"
                  whileHover={{ scale: 1.02 }}
                  whileTap={{ scale: 0.98 }}
                  style={{
                    marginTop: 20,
                    width: '100%',
                    padding: '18px',
                    background: 'var(--white)',
                    color: 'var(--black)',
                    border: 'none',
                    borderRadius: 0,
                    fontFamily: 'Syne',
                    fontWeight: 800,
                    fontSize: 13,
                    letterSpacing: '0.12em',
                    textTransform: 'uppercase',
                    position: 'relative',
                    overflow: 'hidden',
                  }}
                >
                  <span style={{ position: 'relative', zIndex: 1 }}>Run Analysis</span>
                  <div style={{
                    position: 'absolute', inset: 0,
                    background: 'linear-gradient(90deg, transparent, rgba(0,0,0,0.08), transparent)',
                    animation: 'shimmer 2.5s infinite',
                  }} />
                </motion.button>
              )}
            </AnimatePresence>

            {/* Pipeline */}
            <AnimatePresence>
              {analyzing && <RawPipeline stage={prog.stage} pct={prog.pct} />}
            </AnimatePresence>

            {/* Error */}
            <AnimatePresence>
              {error && (
                <motion.div
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  style={{
                    marginTop: 16,
                    padding: '14px 0',
                    borderTop: '1px solid rgba(255,32,32,0.3)',
                    fontFamily: 'JetBrains Mono',
                    fontSize: 12,
                    color: '#ff2020',
                    letterSpacing: '0.04em',
                  }}
                >
                  ✕ {error}
                </motion.div>
              )}
            </AnimatePresence>

            {/* Reset */}
            <AnimatePresence>
              {complete && (
                <motion.button
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  onClick={reset}
                  whileHover={{ x: 4 }}
                  style={{
                    marginTop: 20,
                    background: 'none',
                    border: 'none',
                    padding: 0,
                    fontFamily: 'JetBrains Mono',
                    fontSize: 11,
                    color: 'var(--grey-2)',
                    letterSpacing: '0.12em',
                    textTransform: 'uppercase',
                    display: 'flex',
                    alignItems: 'center',
                    gap: 8,
                    transition: 'color 0.2s',
                  }}
                >
                  ← Analyze another file
                </motion.button>
              )}
            </AnimatePresence>
          </div>

          {/* Right — result */}
          <div>
            <AnimatePresence mode="wait">
              {!complete && !analyzing && (
                <motion.div
                  key="idle"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  style={{ paddingTop: 24 }}
                >
                  <div style={{
                    borderTop: '1px solid var(--border)',
                    paddingTop: 28,
                  }}>
                    <div style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: 'var(--grey-3)', letterSpacing: '0.2em', marginBottom: 16 }}>
                      AWAITING INPUT
                    </div>
                    {['EfficientNet-B4 spatial branch', 'FFT frequency domain branch', 'Dual-branch fusion layer', 'Grad-CAM++ explainability'].map((item, i) => (
                      <motion.div
                        key={item}
                        initial={{ opacity: 0, x: -8 }}
                        animate={{ opacity: 1, x: 0 }}
                        transition={{ delay: i * 0.1 }}
                        style={{
                          padding: '14px 0',
                          borderBottom: '1px solid var(--border)',
                          fontFamily: 'JetBrains Mono',
                          fontSize: 11,
                          color: 'var(--grey-3)',
                          letterSpacing: '0.06em',
                          display: 'flex',
                          alignItems: 'center',
                          gap: 10,
                        }}
                      >
                        <span style={{ color: 'var(--grey-3)', fontSize: 8 }}>○</span>
                        {item}
                      </motion.div>
                    ))}
                  </div>
                </motion.div>
              )}

              {analyzing && (
                <motion.div
                  key="running"
                  initial={{ opacity: 0 }}
                  animate={{ opacity: 1 }}
                  exit={{ opacity: 0 }}
                  style={{ paddingTop: 24 }}
                >
                  <div style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: 'var(--grey-3)', letterSpacing: '0.2em', marginBottom: 24 }}>
                    INFERENCE RUNNING
                  </div>
                  {STAGES.map((s, i) => (
                    <motion.div
                      key={s.key}
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      transition={{ delay: i * 0.15 }}
                      style={{
                        padding: '12px 0',
                        borderBottom: '1px solid var(--border)',
                        fontFamily: 'JetBrains Mono',
                        fontSize: 11,
                        color: 'var(--grey-2)',
                        display: 'flex',
                        justifyContent: 'space-between',
                      }}
                    >
                      <span>{s.label}</span>
                      <motion.span
                        animate={{ opacity: [1, 0.3, 1] }}
                        transition={{ repeat: Infinity, duration: 1.2, delay: i * 0.2 }}
                        style={{ color: 'var(--grey-3)' }}
                      >
                        processing
                      </motion.span>
                    </motion.div>
                  ))}
                </motion.div>
              )}

              {complete && result && (
                <motion.div
                  key="result"
                  initial={{ opacity: 0, y: 12 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.5 }}
                >
                  <AnalysisResultCard result={result} />
                </motion.div>
              )}
            </AnimatePresence>
          </div>
        </div>

        {/* Expanded results */}
        <AnimatePresence>
          {complete && result && (
            <motion.div
              initial={{ opacity: 0, y: 32 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              transition={{ delay: 0.3, duration: 0.6 }}
              style={{ marginTop: 60, display: 'flex', flexDirection: 'column', gap: 1 }}
            >
              {/* Section label */}
              <div style={{
                borderTop: '1px solid var(--border)',
                padding: '20px 0',
                fontFamily: 'JetBrains Mono',
                fontSize: 9,
                color: 'var(--grey-3)',
                letterSpacing: '0.2em',
              }}>
                DETAILED ANALYSIS
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 1, background: 'var(--border)' }}>
                <div style={{ background: '#000', padding: '32px 0' }}>
                  <BranchScoresPanel scores={result.branchScores} label={result.label} />
                </div>
                <div style={{ background: '#000', padding: '32px 0' }}>
                  <HeatmapViewer imageUrl={previewUrl ?? undefined} label={result.label} heatmapUrl={result.heatmapUrl} />
                </div>
              </div>

              {result.frameResults && result.frameResults.length > 0 && (
                <div style={{ borderTop: '1px solid var(--border)', paddingTop: 32 }}>
                  <FrameSampling frames={result.frameResults} />
                  <FrameResults frames={result.frameResults} />
                </div>
              )}
              <div style={{ borderTop: '1px solid var(--border)', paddingTop: 32 }}>
                <ForensicReport result={result} />
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </section>
  );
}
