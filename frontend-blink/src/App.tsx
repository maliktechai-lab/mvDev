import { useRef, useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import Loader from './components/Loader';
import Header from './components/Header';
import Hero from './components/Hero';
import SampleGallery from './components/SampleGallery';
import AnalyzerSection from './components/AnalyzerSection';
import HowItWorks from './components/HowItWorks';

// ─── Footer ───────────────────────────────────────────────────
function Footer() {
  return (
    <footer style={{ background: '#000', borderTop: '1px solid var(--border)' }}>
      <div style={{ maxWidth: 1320, margin: '0 auto', padding: '80px 40px 60px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 40, marginBottom: 80 }}>
          <div style={{ gridColumn: 'span 1' }}>
            <div style={{
              fontFamily: 'Syne', fontWeight: 800, fontSize: 28,
              letterSpacing: '-0.04em', color: 'var(--white)', marginBottom: 16, lineHeight: 1,
            }}>BLINK</div>
            <p style={{
              fontFamily: 'Inter', fontSize: 13,
              color: 'var(--grey-2)', lineHeight: 1.7, maxWidth: 260, marginBottom: 24,
            }}>
              State-of-the-art synthetic media detection. Every fake leaves a trace.
            </p>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
              {['PyTorch', 'FastAPI', 'React', 'Three.js'].map((t) => (
                <span key={t} style={{
                  fontFamily: 'JetBrains Mono', fontSize: 10,
                  padding: '3px 10px', border: '1px solid var(--border)',
                  color: 'var(--grey-2)', letterSpacing: '0.06em',
                }}>{t}</span>
              ))}
            </div>
          </div>

          {[
            { heading: 'Product',    items: ['Detector', 'API Docs', 'Changelog', 'Status'] },
            { heading: 'Technology', items: ['EfficientNet-B4', 'FFT Analysis', 'Grad-CAM++', 'Benchmarks'] },
            { heading: 'Company',    items: ['About', 'Blog', 'Privacy', 'Terms'] },
          ].map(({ heading, items }) => (
            <div key={heading}>
              <div style={{
                fontFamily: 'JetBrains Mono', fontSize: 9, letterSpacing: '0.18em',
                color: 'var(--grey-3)', textTransform: 'uppercase', marginBottom: 20,
              }}>{heading}</div>
              {items.map((item) => (
                <div key={item} style={{ marginBottom: 12 }}>
                  <motion.a
                    href="#"
                    whileHover={{ x: 4, color: 'var(--white)' }}
                    style={{
                      fontFamily: 'Inter', fontSize: 13,
                      color: 'var(--grey-2)', textDecoration: 'none',
                      display: 'inline-block', transition: 'color 0.15s',
                    }}
                  >{item}</motion.a>
                </div>
              ))}
            </div>
          ))}
        </div>

        <div style={{
          borderTop: '1px solid var(--border)', paddingTop: 32,
          display: 'flex', justifyContent: 'space-between',
          alignItems: 'center', flexWrap: 'wrap', gap: 16,
        }}>
          <span style={{ fontFamily: 'JetBrains Mono', fontSize: 10, color: 'var(--grey-3)', letterSpacing: '0.08em' }}>
            © 2025 BLINK. ALL RIGHTS RESERVED.
          </span>
          <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
            <div className="dot-live" />
            <span style={{ fontFamily: 'JetBrains Mono', fontSize: 10, color: 'var(--grey-3)' }}>
              ALL SYSTEMS OPERATIONAL
            </span>
          </div>
          <span style={{ fontFamily: 'JetBrains Mono', fontSize: 10, color: 'var(--grey-3)', letterSpacing: '0.04em' }}>
            PROBABILISTIC OUTPUT — NOT FORENSIC EVIDENCE
          </span>
        </div>
      </div>
    </footer>
  );
}

// ─── App ──────────────────────────────────────────────────────
export default function App() {
  const [loaded, setLoaded] = useState(false);
  const detectorRef = useRef<HTMLDivElement>(null);

  const scrollToDetector = () =>
    detectorRef.current?.scrollIntoView({ behavior: 'smooth' });

  return (
    <div style={{ position: 'relative', background: '#000', minHeight: '100vh' }}>
      <AnimatePresence>
        {!loaded && <Loader onComplete={() => setLoaded(true)} />}
      </AnimatePresence>

      <AnimatePresence>
        {loaded && (
          <motion.div
            key="site"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={{ duration: 0.5, ease: 'easeOut' }}
          >
            <Header />
            <Hero onScrollToDetector={scrollToDetector} />
            <div style={{ height: 1, background: 'var(--border)' }} />
            <SampleGallery />
            <div style={{ height: 1, background: 'var(--border)' }} />
            <div ref={detectorRef}>
              <AnalyzerSection />
            </div>
            <div style={{ height: 1, background: 'var(--border)' }} />
            <HowItWorks />
            <Footer />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
