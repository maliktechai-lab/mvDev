import { useEffect, useRef, useState } from 'react';
import { motion, useMotionValue, useSpring, AnimatePresence } from 'framer-motion';

const NAV = [
  { label: 'Detector',     href: '#detector' },
  { label: 'How It Works', href: '#how-it-works' },
  { label: 'Technology',   href: '#how-it-works' },
];

function MagneticLink({ label, href }: { label: string; href: string }) {
  const ref = useRef<HTMLAnchorElement>(null);
  const x = useMotionValue(0);
  const y = useMotionValue(0);
  const sx = useSpring(x, { stiffness: 250, damping: 20 });
  const sy = useSpring(y, { stiffness: 250, damping: 20 });
  const [hovered, setHovered] = useState(false);

  const onMove = (e: React.MouseEvent) => {
    if (!ref.current) return;
    const rect = ref.current.getBoundingClientRect();
    const dx = e.clientX - (rect.left + rect.width / 2);
    const dy = e.clientY - (rect.top + rect.height / 2);
    const dist = Math.sqrt(dx * dx + dy * dy);
    if (dist < 70) { x.set(dx * 0.3); y.set(dy * 0.3); }
  };

  const onLeave = () => { x.set(0); y.set(0); setHovered(false); };

  return (
    <motion.a
      ref={ref}
      href={href}
      style={{
        x: sx, y: sy,
        fontFamily: 'JetBrains Mono',
        fontSize: 11,
        letterSpacing: '0.12em',
        textTransform: 'uppercase',
        color: hovered ? 'var(--white)' : 'var(--grey-2)',
        textDecoration: 'none',
        padding: '6px 14px',
        transition: 'color 0.2s',
        position: 'relative',
      }}
      onMouseMove={onMove}
      onMouseLeave={onLeave}
      onMouseEnter={() => setHovered(true)}
    >
      {label}
      {/* Underline */}
      <motion.div
        style={{
          position: 'absolute', bottom: 2, left: 14, right: 14,
          height: 1, background: 'var(--white)',
          scaleX: hovered ? 1 : 0,
          transformOrigin: 'left',
          transition: 'transform 0.3s ease',
        }}
      />
    </motion.a>
  );
}

export default function Header() {
  const [scrolled, setScrolled] = useState(false);
  const [open, setOpen] = useState(false);
  const [scrollPct, setScrollPct] = useState(0);

  useEffect(() => {
    const fn = () => {
      const s = window.scrollY;
      setScrolled(s > 60);
      const docH = document.documentElement.scrollHeight - window.innerHeight;
      setScrollPct(docH > 0 ? (s / docH) * 100 : 0);
    };
    window.addEventListener('scroll', fn, { passive: true });
    return () => window.removeEventListener('scroll', fn);
  }, []);

  return (
    <>
      <motion.header
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.6, delay: 0.2 }}
        style={{
          position: 'fixed',
          top: 0, left: 0, right: 0,
          zIndex: 1000,
          background: scrolled ? 'rgba(0,0,0,0.92)' : 'transparent',
          backdropFilter: scrolled ? 'blur(20px)' : 'none',
          borderBottom: scrolled ? '1px solid var(--border)' : '1px solid transparent',
          transition: 'background 0.4s, border-color 0.4s',
        }}
      >
        <div style={{
          maxWidth: 1320, margin: '0 auto',
          padding: '0 40px',
          height: 70,
          display: 'flex', alignItems: 'center', justifyContent: 'space-between',
        }}>

          {/* Logo — BLINK wordmark */}
          <a href="/" style={{ textDecoration: 'none', display: 'flex', alignItems: 'center', gap: 12 }}>
            <motion.div
              whileHover={{ scale: 1.02 }}
              style={{
                fontFamily: 'Syne',
                fontWeight: 800,
                fontSize: 22,
                letterSpacing: '-0.04em',
                color: 'var(--white)',
                lineHeight: 1,
              }}
              data-text="BLINK"
              className="glitch"
            >
              BLINK
            </motion.div>
            <div style={{
              width: 1, height: 16,
              background: 'var(--grey-3)',
            }} />
            <span style={{
              fontFamily: 'JetBrains Mono',
              fontSize: 9,
              letterSpacing: '0.14em',
              color: 'var(--grey-2)',
              textTransform: 'uppercase',
            }}>
              Detect
            </span>
          </a>

          {/* Desktop nav */}
          <nav style={{ display: 'flex', alignItems: 'center', gap: 0 }}>
            {NAV.map(({ label, href }) => (
              <MagneticLink key={label} label={label} href={href} />
            ))}
          </nav>

          {/* Right */}
          <div style={{ display: 'flex', alignItems: 'center', gap: 20 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 7 }}>
              <div className="dot-live" />
              <span style={{ fontFamily: 'JetBrains Mono', fontSize: 9, color: 'var(--grey-2)', letterSpacing: '0.12em' }}>
                LIVE
              </span>
            </div>

            <motion.a
              href="#detector"
              data-cursor="scan"
              whileHover={{ scale: 1.03 }}
              whileTap={{ scale: 0.97 }}
              style={{
                display: 'inline-flex',
                alignItems: 'center',
                gap: 8,
                padding: '9px 22px',
                background: 'var(--white)',
                color: 'var(--black)',
                border: 'none',
                borderRadius: 0,
                fontFamily: 'Syne',
                fontWeight: 800,
                fontSize: 11,
                letterSpacing: '0.1em',
                textTransform: 'uppercase',
                textDecoration: 'none',
                position: 'relative',
                overflow: 'hidden',
              }}
            >
              Try Blink
              <div style={{
                position: 'absolute', inset: 0,
                background: 'linear-gradient(90deg, transparent, rgba(0,0,0,0.08), transparent)',
                animation: 'shimmer 2.5s infinite',
              }} />
            </motion.a>

            {/* Mobile */}
            <button
              onClick={() => setOpen(!open)}
              style={{
                display: 'none',
                background: 'none', border: 'none',
                color: 'var(--grey-1)', padding: 4,
              }}
            >
              <svg width="20" height="20" fill="none" stroke="currentColor" strokeWidth="1.5">
                {open
                  ? <path d="M5 5l14 14M5 19L19 5" strokeLinecap="round" />
                  : <><line x1="3" y1="7" x2="21" y2="7" strokeLinecap="round" />
                     <line x1="3" y1="14" x2="21" y2="14" strokeLinecap="round" /></>
                }
              </svg>
            </button>
          </div>
        </div>

        {/* Scroll progress */}
        <div style={{ height: 1, background: 'var(--border)' }}>
          <div style={{
            height: '100%',
            background: 'var(--white)',
            width: `${scrollPct}%`,
            transition: 'width 0.1s linear',
            opacity: 0.5,
          }} />
        </div>
      </motion.header>

      {/* Mobile drawer */}
      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -8 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -8 }}
            style={{
              position: 'fixed', top: 71, left: 0, right: 0,
              zIndex: 999,
              background: '#000',
              borderBottom: '1px solid var(--border)',
              padding: '20px 40px 28px',
            }}
          >
            {NAV.map(({ label, href }) => (
              <a
                key={label}
                href={href}
                onClick={() => setOpen(false)}
                style={{
                  display: 'block',
                  padding: '14px 0',
                  fontFamily: 'Syne',
                  fontWeight: 700,
                  fontSize: 20,
                  color: 'var(--grey-1)',
                  textDecoration: 'none',
                  borderBottom: '1px solid var(--border)',
                  letterSpacing: '-0.02em',
                }}
              >{label}</a>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}
