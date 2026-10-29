import { useEffect, useRef, useState } from 'react';
import { motion, useScroll, useTransform } from 'framer-motion';
import GlitchFace from './GlitchFace';

// ─── Marquee strip ─────────────────────────────────────────────
function MarqueeStrip() {
  const items = [
    'DEEPFAKE DETECTION', '—', 'DUAL-BRANCH AI', '—',
    'EFFICIENTNET-B4', '—', 'FFT FREQUENCY', '—',
    'GRAD-CAM++', '—', 'BLINK ENGINE v2', '—',
  ];
  const full = [...items, ...items];

  return (
    <div style={{
      overflow: 'hidden',
      borderTop: '1px solid var(--border)',
      borderBottom: '1px solid var(--border)',
      padding: '10px 0',
      background: 'rgba(0,0,0,0.6)',
      backdropFilter: 'blur(8px)',
    }}>
      <div className="marquee-track">
        {full.map((item, i) => (
          <span key={i} style={{
            fontFamily: 'JetBrains Mono', fontSize: 10,
            letterSpacing: '0.18em',
            color: item === '—' ? 'var(--grey-3)' : 'var(--grey-2)',
            textTransform: 'uppercase', paddingRight: 40, whiteSpace: 'nowrap',
          }}>
            {item}
          </span>
        ))}
      </div>
    </div>
  );
}

// ─── Corrupted counter ─────────────────────────────────────────
function CorruptCounter({ value, suffix = '' }: { value: string; suffix?: string }) {
  const [display, setDisplay] = useState('___');
  const chars = '0123456789%.<>';

  useEffect(() => {
    let ticks = 0;
    const max = 12;
    const iv = setInterval(() => {
      if (ticks < max) {
        setDisplay(value.split('').map(() => chars[Math.floor(Math.random() * chars.length)]).join(''));
        ticks++;
      } else {
        setDisplay(value);
        clearInterval(iv);
      }
    }, 60);
    return () => clearInterval(iv);
  }, [value]);

  return <span style={{ fontFamily: 'JetBrains Mono' }}>{display}{suffix}</span>;
}

// ─── Main Hero ─────────────────────────────────────────────────
export default function Hero({ onScrollToDetector }: { onScrollToDetector: () => void }) {
  const ref = useRef<HTMLElement>(null);
  const { scrollYProgress } = useScroll({ target: ref, offset: ['start start', 'end start'] });

  const faceY  = useTransform(scrollYProgress, [0, 1], ['0%', '15%']);
  const textY  = useTransform(scrollYProgress, [0, 1], ['0%', '8%']);
  const opacity = useTransform(scrollYProgress, [0, 0.7], [1, 0]);

  const [mounted, setMounted] = useState(false);
  useEffect(() => { setTimeout(() => setMounted(true), 100); }, []);

  return (
    <section ref={ref} style={{
      position: 'relative', height: '100vh', minHeight: 700,
      overflow: 'hidden', background: '#000',
    }}>
      {/* Full-bleed glitch face */}
      <motion.div style={{ position: 'absolute', inset: 0, y: faceY }}>
        <GlitchFace intensity={0.6} />
      </motion.div>

      {/* Content layer */}
      <motion.div style={{
        position: 'absolute', inset: 0,
        y: textY, opacity,
        display: 'flex', flexDirection: 'column',
        justifyContent: 'flex-end', padding: '0 0 80px',
      }}>
        {/* Top-left meta tag */}
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={mounted ? { opacity: 1, x: 0 } : {}}
          transition={{ delay: 0.2, duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
          style={{ position: 'absolute', top: 100, left: 40, display: 'flex', alignItems: 'center', gap: 10 }}
        >
          <div className="dot-live" />
          <span style={{
            fontFamily: 'JetBrains Mono', fontSize: 10,
            letterSpacing: '0.2em', color: 'var(--grey-2)', textTransform: 'uppercase',
          }}>
            AI DETECTION ENGINE — ACTIVE
          </span>
        </motion.div>

        {/* Top-right accuracy counter */}
        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={mounted ? { opacity: 1, x: 0 } : {}}
          transition={{ delay: 0.3, duration: 0.6 }}
          style={{
            position: 'absolute', top: 100, right: 40,
            fontFamily: 'JetBrains Mono', fontSize: 10,
            letterSpacing: '0.12em', color: 'var(--grey-2)', textAlign: 'right',
          }}
        >
          <div style={{ marginBottom: 4 }}>ACCURACY</div>
          <div style={{ fontSize: 22, fontWeight: 700, color: 'var(--white)', letterSpacing: '-0.02em' }}>
            {mounted && <CorruptCounter value="99.2" suffix="%" />}
          </div>
        </motion.div>

        {/* Main typography */}
        <div style={{ padding: '0 40px' }}>
          <motion.div
            initial={{ opacity: 0, y: 16 }}
            animate={mounted ? { opacity: 1, y: 0 } : {}}
            transition={{ delay: 0.3, duration: 0.6 }}
            style={{
              fontFamily: 'JetBrains Mono', fontSize: 11,
              letterSpacing: '0.22em', color: 'var(--grey-2)',
              textTransform: 'uppercase', marginBottom: 20,
            }}
          >
            In the blink of an eye —
          </motion.div>

          <div style={{ lineHeight: 0.9 }}>
            {['DETECT', 'SYNTHETIC', 'MEDIA.'].map((word, wi) => (
              <div key={word} style={{ overflow: 'hidden', display: 'block' }}>
                <motion.div
                  initial={{ y: '110%' }}
                  animate={mounted ? { y: '0%' } : {}}
                  transition={{ delay: 0.4 + wi * 0.12, duration: 0.8, ease: [0.16, 1, 0.3, 1] }}
                  style={{
                    fontFamily: 'Syne', fontSize: 'clamp(44px, 7vw, 108px)',
                    fontWeight: 800, letterSpacing: '-0.04em', lineHeight: 0.9,
                    color: wi === 1 ? 'transparent' : 'var(--white)',
                    WebkitTextStroke: wi === 1 ? '1px rgba(255,255,255,0.45)' : 'none',
                    whiteSpace: 'nowrap', display: 'block',
                  }}
                >
                  {word}
                </motion.div>
              </div>
            ))}
          </div>

          {/* CTA row */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={mounted ? { opacity: 1, y: 0 } : {}}
            transition={{ delay: 0.85, duration: 0.6, ease: [0.16, 1, 0.3, 1] }}
            style={{
              display: 'flex', alignItems: 'flex-end',
              justifyContent: 'space-between', marginTop: 40,
              flexWrap: 'wrap', gap: 24,
            }}
          >
            <p style={{
              fontFamily: 'Inter', fontSize: 15, lineHeight: 1.7,
              color: 'var(--grey-1)', maxWidth: 380,
            }}>
              Dual-branch deep learning —
              EfficientNet-B4 spatial analysis
              combined with FFT frequency detection.
              Every fake leaves a trace.
            </p>

            <div style={{ display: 'flex', gap: 14, alignItems: 'center' }}>
              <motion.button
                onClick={onScrollToDetector}
                whileHover={{ scale: 1.04 }}
                whileTap={{ scale: 0.97 }}
                style={{
                  padding: '14px 36px',
                  background: 'var(--white)', color: 'var(--black)',
                  border: 'none', borderRadius: 0,
                  fontFamily: 'Syne', fontWeight: 800,
                  fontSize: 13, letterSpacing: '0.1em',
                  textTransform: 'uppercase',
                  position: 'relative', overflow: 'hidden',
                }}
              >
                <span style={{ position: 'relative', zIndex: 1 }}>Analyze Media</span>
                <div style={{
                  position: 'absolute', inset: 0,
                  background: 'linear-gradient(90deg, transparent, rgba(0,0,0,0.1), transparent)',
                  animation: 'shimmer 2.5s infinite',
                }} />
              </motion.button>

              <motion.a
                href="#how-it-works"
                whileHover={{ x: 4 }}
                style={{
                  fontFamily: 'JetBrains Mono', fontSize: 11,
                  letterSpacing: '0.14em', color: 'var(--grey-2)',
                  textDecoration: 'none', textTransform: 'uppercase',
                  display: 'flex', alignItems: 'center', gap: 8, transition: 'color 0.2s',
                }}
                onMouseEnter={(e) => (e.currentTarget.style.color = 'var(--white)')}
                onMouseLeave={(e) => (e.currentTarget.style.color = 'var(--grey-2)')}
              >
                How it works <span style={{ fontSize: 14 }}>→</span>
              </motion.a>
            </div>
          </motion.div>
        </div>
      </motion.div>

      {/* Marquee strip at bottom */}
      <div style={{ position: 'absolute', bottom: 0, left: 0, right: 0 }}>
        <MarqueeStrip />
      </div>

      {/* Scroll indicator */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={mounted ? { opacity: 1 } : {}}
        transition={{ delay: 1.8 }}
        style={{
          position: 'absolute', bottom: 60, right: 40,
          display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 8, zIndex: 10,
        }}
      >
        <span style={{
          fontFamily: 'JetBrains Mono', fontSize: 9,
          letterSpacing: '0.22em', color: 'var(--grey-3)', writingMode: 'vertical-rl',
        }}>SCROLL</span>
        <motion.div
          animate={{ scaleY: [0, 1, 0] }}
          transition={{ repeat: Infinity, duration: 1.8, ease: 'easeInOut' }}
          style={{
            width: 1, height: 48,
            background: 'linear-gradient(to bottom, var(--grey-3), transparent)',
            transformOrigin: 'top',
          }}
        />
      </motion.div>
    </section>
  );
}
