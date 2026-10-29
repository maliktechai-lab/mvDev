import { useEffect, useState, useRef } from 'react';
import { motion, AnimatePresence } from 'framer-motion';

interface Props {
  onComplete: () => void;
}

const LETTERS = ['B', 'L', 'I', 'N', 'K'];
// Corrupted alternatives for each letter
const CORRUPTS = ['8', '|', '!', 'И', 'X'];
const GLITCH_CHARS = ['@', '#', '%', '&', '/', '\\', '?', '!', '0', '1', '_', '—'];

function randomGlitch() {
  return GLITCH_CHARS[Math.floor(Math.random() * GLITCH_CHARS.length)];
}

export default function Loader({ onComplete }: Props) {
  const [phase, setPhase] = useState<'init' | 'corrupt' | 'resolve' | 'scan' | 'tear'>('init');
  const [letterStates, setLetterStates] = useState(LETTERS.map((l) => ({ char: ' ', final: l, corrupt: CORRUPTS[LETTERS.indexOf(l)] })));
  const [scanPos, setScanPos] = useState(0);
  const [progress, setProgress] = useState(0);
  const rafRef = useRef<number>(0);

  useEffect(() => {
    // Phase 1 — letters flicker in one by one
    const timers: ReturnType<typeof setTimeout>[] = [];

    LETTERS.forEach((_, i) => {
      // Rapid corruption before settling
      timers.push(setTimeout(() => {
        let ticks = 0;
        const corrupt = setInterval(() => {
          setLetterStates((prev) => {
            const next = [...prev];
            next[i] = { ...next[i], char: randomGlitch() };
            return next;
          });
          ticks++;
          if (ticks > 6) {
            clearInterval(corrupt);
            setLetterStates((prev) => {
              const next = [...prev];
              next[i] = { ...next[i], char: LETTERS[i] };
              return next;
            });
          }
        }, 60);
      }, 200 + i * 180));
    });

    // Phase 2 — all letters resolved, enter corrupt phase
    timers.push(setTimeout(() => setPhase('corrupt'), 200 + LETTERS.length * 180 + 500));

    // Phase 3 — scan beam
    timers.push(setTimeout(() => {
      setPhase('scan');
      const start = performance.now();
      const dur = 900;
      const animate = (now: number) => {
        const p = Math.min((now - start) / dur, 1);
        setScanPos(p * 100);
        setProgress(Math.round(p * 100));
        if (p < 1) {
          rafRef.current = requestAnimationFrame(animate);
        } else {
          setPhase('tear');
        }
      };
      rafRef.current = requestAnimationFrame(animate);
    }, 200 + LETTERS.length * 180 + 1200));

    // Phase 4 — tear open, call onComplete
    timers.push(setTimeout(() => {
      onComplete();
    }, 200 + LETTERS.length * 180 + 1200 + 900 + 600));

    return () => {
      timers.forEach(clearTimeout);
      cancelAnimationFrame(rafRef.current);
    };
  }, [onComplete]);

  return (
    <AnimatePresence>
      {phase !== 'tear' ? (
        <motion.div
          key="loader"
          exit={{ opacity: 0 }}
          style={{
            position: 'fixed', inset: 0, zIndex: 99999,
            background: '#000',
            display: 'flex', flexDirection: 'column',
            alignItems: 'center', justifyContent: 'center',
            overflow: 'hidden',
          }}
        >
          {/* Scanlines overlay */}
          <div style={{
            position: 'absolute', inset: 0, pointerEvents: 'none',
            backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 3px, rgba(0,0,0,0.18) 3px, rgba(0,0,0,0.18) 4px)',
            zIndex: 2,
          }} />

          {/* BLINK letters */}
          <div style={{
            display: 'flex', gap: 'clamp(8px, 2vw, 24px)',
            position: 'relative', zIndex: 3,
          }}>
            {letterStates.map((l, i) => (
              <motion.span
                key={i}
                initial={{ opacity: 0 }}
                animate={{ opacity: l.char === ' ' ? 0 : 1 }}
                style={{
                  fontFamily: 'Syne',
                  fontSize: 'clamp(72px, 14vw, 180px)',
                  fontWeight: 800,
                  letterSpacing: '-0.04em',
                  lineHeight: 1,
                  color: '#fff',
                  position: 'relative',
                  display: 'inline-block',
                  textShadow: phase === 'corrupt'
                    ? '0 0 40px rgba(255,255,255,0.6), 0 0 80px rgba(255,255,255,0.2)'
                    : 'none',
                  transition: 'text-shadow 0.3s',
                }}
              >
                {/* RGB split ghost layers during corrupt phase */}
                {phase === 'corrupt' && (
                  <>
                    <span style={{
                      position: 'absolute', inset: 0,
                      color: 'rgba(255,30,60,0.7)',
                      animation: 'rgb-r 0.4s infinite',
                      pointerEvents: 'none',
                    }}>{l.char}</span>
                    <span style={{
                      position: 'absolute', inset: 0,
                      color: 'rgba(0,180,255,0.7)',
                      animation: 'rgb-b 0.4s infinite',
                      animationDelay: '0.05s',
                      pointerEvents: 'none',
                    }}>{l.char}</span>
                  </>
                )}
                {l.char}
              </motion.span>
            ))}
          </div>

          {/* Tagline */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: phase === 'init' ? 0 : 0.5 }}
            transition={{ delay: 0.3 }}
            style={{
              fontFamily: 'JetBrains Mono',
              fontSize: 11,
              letterSpacing: '0.25em',
              color: '#fff',
              marginTop: 24,
              textTransform: 'uppercase',
            }}
          >
            Deepfake Detection Engine
          </motion.div>

          {/* Scan beam */}
          {phase === 'scan' && (
            <motion.div
              style={{
                position: 'absolute',
                left: 0, right: 0,
                top: `${scanPos}%`,
                height: 2,
                background: 'linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.9) 20%, #fff 50%, rgba(255,255,255,0.9) 80%, transparent 100%)',
                boxShadow: '0 0 20px 4px rgba(255,255,255,0.4)',
                zIndex: 4,
                pointerEvents: 'none',
              }}
            />
          )}

          {/* Progress counter */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: phase === 'scan' ? 1 : 0 }}
            style={{
              position: 'absolute',
              bottom: 48,
              right: 48,
              fontFamily: 'JetBrains Mono',
              fontSize: 13,
              color: 'rgba(255,255,255,0.6)',
              letterSpacing: '0.04em',
              zIndex: 5,
            }}
          >
            {String(progress).padStart(3, '0')}%
          </motion.div>

          {/* Bottom line */}
          <div style={{
            position: 'absolute', bottom: 0, left: 0, right: 0,
            height: 1, background: 'var(--border)',
            zIndex: 5,
          }} />
        </motion.div>
      ) : (
        /* Tear-open panels */
        <motion.div key="tear" style={{ position: 'fixed', inset: 0, zIndex: 99999, pointerEvents: 'none' }}>
          <motion.div
            initial={{ y: 0 }}
            animate={{ y: '-100%' }}
            transition={{ duration: 0.7, ease: [0.76, 0, 0.24, 1] }}
            style={{
              position: 'absolute', top: 0, left: 0, right: 0,
              height: '50%', background: '#000',
            }}
          />
          <motion.div
            initial={{ y: 0 }}
            animate={{ y: '100%' }}
            transition={{ duration: 0.7, ease: [0.76, 0, 0.24, 1] }}
            style={{
              position: 'absolute', bottom: 0, left: 0, right: 0,
              height: '50%', background: '#000',
            }}
          />
        </motion.div>
      )}
    </AnimatePresence>
  );
}
