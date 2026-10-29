import { useEffect, useState, useCallback } from 'react';
import { motion, useMotionValue, useSpring } from 'framer-motion';

type CursorState = 'default' | 'hover' | 'click' | 'scan' | 'text' | 'upload' | 'drag';

export default function CustomCursor() {
  const [state, setState] = useState<CursorState>('default');
  const [label, setLabel] = useState('');
  const [visible, setVisible] = useState(false);
  const [clicking, setClicking] = useState(false);

  const mx = useMotionValue(0);
  const my = useMotionValue(0);

  // Main cursor — tight spring
  const cx = useSpring(mx, { stiffness: 700, damping: 40, mass: 0.3 });
  const cy = useSpring(my, { stiffness: 700, damping: 40, mass: 0.3 });

  // Trail dot — looser spring
  const tx = useSpring(mx, { stiffness: 120, damping: 22, mass: 0.5 });
  const ty = useSpring(my, { stiffness: 120, damping: 22, mass: 0.5 });

  const updateState = useCallback((el: Element | null) => {
    if (!el) { setState('default'); setLabel(''); return; }

    // Walk up DOM to find meaningful target
    let target: Element | null = el;
    while (target) {
      const tag = target.tagName?.toLowerCase();
      const role = target.getAttribute('role');
      const dataCursor = target.getAttribute('data-cursor');
      const dataLabel = target.getAttribute('data-cursor-label');

      if (dataCursor) {
        setState(dataCursor as CursorState);
        setLabel(dataLabel ?? '');
        return;
      }
      if (tag === 'button' || tag === 'a' || role === 'button') {
        setState('hover');
        setLabel(dataLabel ?? '');
        return;
      }
      if (tag === 'input' || tag === 'textarea') {
        setState('text');
        setLabel('');
        return;
      }
      if (target.classList.contains('upload-zone')) {
        setState('upload');
        setLabel('DROP');
        return;
      }
      if (target.classList.contains('scan-trigger')) {
        setState('scan');
        setLabel('SCAN');
        return;
      }
      target = target.parentElement;
    }
    setState('default');
    setLabel('');
  }, []);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      mx.set(e.clientX);
      my.set(e.clientY);
      if (!visible) setVisible(true);
      updateState(document.elementFromPoint(e.clientX, e.clientY));
    };
    const onDown = () => setClicking(true);
    const onUp = () => setClicking(false);
    const onLeave = () => setVisible(false);
    const onEnter = () => setVisible(true);

    window.addEventListener('mousemove', onMove);
    window.addEventListener('mousedown', onDown);
    window.addEventListener('mouseup', onUp);
    document.documentElement.addEventListener('mouseleave', onLeave);
    document.documentElement.addEventListener('mouseenter', onEnter);

    return () => {
      window.removeEventListener('mousemove', onMove);
      window.removeEventListener('mousedown', onDown);
      window.removeEventListener('mouseup', onUp);
      document.documentElement.removeEventListener('mouseleave', onLeave);
      document.documentElement.removeEventListener('mouseenter', onEnter);
    };
  }, [mx, my, visible, updateState]);

  if (!visible) return null;

  // ─── State-based ring config ───────────────────────────────────
  const configs: Record<CursorState, {
    dotSize: number;
    ringSize: number;
    ringColor: string;
    ringOpacity: number;
    dotColor: string;
    ringBorderColor: string;
    ringBorderWidth: number;
    mixBlend?: string;
  }> = {
    default: {
      dotSize: 6, ringSize: 36,
      dotColor: '#ffffff',
      ringColor: 'transparent',
      ringBorderColor: 'rgba(255,255,255,0.35)',
      ringBorderWidth: 1.5,
      ringOpacity: 1,
    },
    hover: {
      dotSize: 0, ringSize: 56,
      dotColor: 'transparent',
      ringColor: 'rgba(167,139,250,0.12)',
      ringBorderColor: 'rgba(167,139,250,0.8)',
      ringBorderWidth: 1.5,
      ringOpacity: 1,
      mixBlend: 'screen',
    },
    click: {
      dotSize: 14, ringSize: 24,
      dotColor: 'rgba(167,139,250,0.9)',
      ringColor: 'rgba(167,139,250,0.2)',
      ringBorderColor: '#a78bfa',
      ringBorderWidth: 1.5,
      ringOpacity: 1,
    },
    scan: {
      dotSize: 4, ringSize: 64,
      dotColor: '#00e5ff',
      ringColor: 'rgba(0,229,255,0.06)',
      ringBorderColor: 'rgba(0,229,255,0.7)',
      ringBorderWidth: 1,
      ringOpacity: 1,
      mixBlend: 'screen',
    },
    text: {
      dotSize: 2, ringSize: 4,
      dotColor: '#ffffff',
      ringColor: 'transparent',
      ringBorderColor: 'transparent',
      ringBorderWidth: 0,
      ringOpacity: 0,
    },
    upload: {
      dotSize: 8, ringSize: 72,
      dotColor: '#a78bfa',
      ringColor: 'rgba(167,139,250,0.06)',
      ringBorderColor: 'rgba(167,139,250,0.6)',
      ringBorderWidth: 1.5,
      ringOpacity: 1,
    },
    drag: {
      dotSize: 6, ringSize: 44,
      dotColor: '#00e5ff',
      ringColor: 'rgba(0,229,255,0.08)',
      ringBorderColor: 'rgba(0,229,255,0.5)',
      ringBorderWidth: 1.5,
      ringOpacity: 1,
    },
  };

  const cfg = configs[clicking ? 'click' : state];

  return (
    <>
      {/* Outer ring */}
      <motion.div
        style={{
          position: 'fixed',
          top: 0, left: 0,
          x: tx,
          y: ty,
          translateX: '-50%',
          translateY: '-50%',
          pointerEvents: 'none',
          zIndex: 99998,
          borderRadius: '50%',
          borderStyle: 'solid',
          mixBlendMode: (cfg.mixBlend as React.CSSProperties['mixBlendMode']) ?? 'normal',
        }}
        animate={{
          width: cfg.ringSize,
          height: cfg.ringSize,
          opacity: cfg.ringOpacity,
          backgroundColor: cfg.ringColor,
          borderColor: cfg.ringBorderColor,
          borderWidth: cfg.ringBorderWidth,
        }}
        transition={{ type: 'spring', stiffness: 300, damping: 28, mass: 0.5 }}
        className="cursor-ring"
      >
        <div style={{
          position: 'absolute', inset: 0,
          borderRadius: '50%',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
        }}>
          {label && (
            <motion.span
              initial={{ opacity: 0, scale: 0.6 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0.6 }}
              style={{
                fontFamily: 'JetBrains Mono',
                fontSize: 9,
                fontWeight: 700,
                letterSpacing: '0.1em',
                color: cfg.dotColor,
                userSelect: 'none',
                whiteSpace: 'nowrap',
              }}
            >
              {label}
            </motion.span>
          )}
        </div>
      </motion.div>

      {/* Inner dot */}
      <motion.div
        style={{
          position: 'fixed',
          top: 0, left: 0,
          x: cx,
          y: cy,
          translateX: '-50%',
          translateY: '-50%',
          pointerEvents: 'none',
          zIndex: 99999,
          borderRadius: '50%',
          backgroundColor: cfg.dotColor,
        }}
        animate={{
          width: cfg.dotSize,
          height: cfg.dotSize,
          opacity: cfg.dotSize > 0 ? 1 : 0,
        }}
        transition={{ type: 'spring', stiffness: 700, damping: 35, mass: 0.2 }}
      />

      {/* Scan state crosshair lines */}
      {(state === 'scan') && (
        <>
          <motion.div
            initial={{ opacity: 0, scaleX: 0 }}
            animate={{ opacity: 0.4, scaleX: 1 }}
            style={{
              position: 'fixed', top: 0, left: 0,
              x: tx, y: ty,
              translateX: '-50%', translateY: '-50%',
              pointerEvents: 'none', zIndex: 99997,
              width: 80, height: 1,
              background: 'linear-gradient(90deg, transparent, var(--cyan), transparent)',
            }}
          />
          <motion.div
            initial={{ opacity: 0, scaleY: 0 }}
            animate={{ opacity: 0.4, scaleY: 1 }}
            style={{
              position: 'fixed', top: 0, left: 0,
              x: tx, y: ty,
              translateX: '-50%', translateY: '-50%',
              pointerEvents: 'none', zIndex: 99997,
              width: 1, height: 80,
              background: 'linear-gradient(180deg, transparent, var(--cyan), transparent)',
            }}
          />
        </>
      )}
    </>
  );
}
