import { useEffect, useRef, useState } from 'react';

// ─── GlitchFace ───────────────────────────────────────────────
// Full-bleed hero canvas: hero image + RGB-split glitch bursts +
// scan-line sweep + vignette. No external scroll wiring.

export default function GlitchFace({ intensity: _intensity = 0.5 }: { intensity?: number }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [glitching, setGlitching] = useState(false);

  // ── Random glitch burst scheduler ────────────────────────────
  useEffect(() => {
    let cancelled = false;

    const schedule = () => {
      if (cancelled) return;
      const delay    = 2000 + Math.random() * 4000;  // 2–6 s
      const duration =  120 + Math.random() * 200;   // 120–320 ms

      setTimeout(() => {
        if (cancelled) return;
        setGlitching(true);
        setTimeout(() => {
          if (cancelled) return;
          setGlitching(false);
          schedule();
        }, duration);
      }, delay);
    };

    schedule();
    return () => { cancelled = true; };
  }, []);

  // ── Canvas render loop ────────────────────────────────────────
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const W = canvas.width;
    const H = canvas.height;

    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.src = 'https://images.unsplash.com/photo-1531746020798-e6953c6e8e04?w=1920&q=85&fit=crop';

    let animId: number;
    let t = 0;

    const drawFrame = () => {
      t += 0.016;
      ctx.clearRect(0, 0, W, H);

      ctx.fillStyle = '#000';
      ctx.fillRect(0, 0, W, H);

      if (img.complete && img.naturalWidth > 0) {
        const scale = Math.max(W / img.naturalWidth, H / img.naturalHeight);
        const sw = img.naturalWidth  * scale;
        const sh = img.naturalHeight * scale;
        const sx = (W - sw) / 2;
        const sy = (H - sh) / 2;

        // Base image — slightly darkened
        ctx.globalAlpha = 0.55;
        ctx.drawImage(img, sx, sy, sw, sh);
        ctx.globalAlpha = 1;

        if (glitching) {
          // RGB channel split
          ctx.globalCompositeOperation = 'screen';
          ctx.globalAlpha = 0.35;
          ctx.drawImage(img, sx - 8, sy, sw, sh); // red ghost left
          ctx.drawImage(img, sx + 8, sy, sw, sh); // blue ghost right
          ctx.globalAlpha = 1;
          ctx.globalCompositeOperation = 'source-over';

          // Horizontal slice displacements
          const numSlices = 6 + Math.floor(Math.random() * 8);
          for (let i = 0; i < numSlices; i++) {
            const sliceY = Math.random() * H;
            const sliceH = 2 + Math.random() * 14;
            const shift  = (Math.random() - 0.5) * 40;
            ctx.drawImage(
              img,
              0,
              (sliceY / H) * img.naturalHeight,
              img.naturalWidth,
              (sliceH / H) * img.naturalHeight,
              sx + shift, sliceY, sw, sliceH,
            );
          }
        }

        // Vertical scan lines
        ctx.globalCompositeOperation = 'multiply';
        ctx.fillStyle = 'rgba(0,0,0,0.25)';
        for (let x = 0; x < W; x += 3) ctx.fillRect(x, 0, 1, H);
        ctx.globalCompositeOperation = 'source-over';

        // Slow horizontal scan sweep
        const scanY = (Math.sin(t * 0.4) * 0.5 + 0.5) * H;
        const sg = ctx.createLinearGradient(0, scanY - 40, 0, scanY + 40);
        sg.addColorStop(0,    'rgba(255,255,255,0)');
        sg.addColorStop(0.45, 'rgba(255,255,255,0.02)');
        sg.addColorStop(0.5,  'rgba(255,255,255,0.06)');
        sg.addColorStop(0.55, 'rgba(255,255,255,0.02)');
        sg.addColorStop(1,    'rgba(255,255,255,0)');
        ctx.fillStyle = sg;
        ctx.fillRect(0, scanY - 40, W, 80);

        // Edge vignette
        const vign = ctx.createRadialGradient(W / 2, H / 2, H * 0.25, W / 2, H / 2, H * 0.85);
        vign.addColorStop(0, 'rgba(0,0,0,0)');
        vign.addColorStop(1, 'rgba(0,0,0,0.85)');
        ctx.fillStyle = vign;
        ctx.fillRect(0, 0, W, H);

        // Bottom fade to black
        const bot = ctx.createLinearGradient(0, H * 0.4, 0, H);
        bot.addColorStop(0, 'rgba(0,0,0,0)');
        bot.addColorStop(1, 'rgba(0,0,0,1)');
        ctx.fillStyle = bot;
        ctx.fillRect(0, 0, W, H);

        // Top fade
        const top = ctx.createLinearGradient(0, 0, 0, H * 0.25);
        top.addColorStop(0, 'rgba(0,0,0,0.7)');
        top.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = top;
        ctx.fillRect(0, 0, W, H);

      } else {
        // Procedural placeholder while image loads
        const grad = ctx.createRadialGradient(W * 0.5, H * 0.38, 0, W * 0.5, H * 0.38, H * 0.55);
        grad.addColorStop(0, 'rgba(30,30,35,0.9)');
        grad.addColorStop(0.5, 'rgba(12,12,15,0.95)');
        grad.addColorStop(1, 'rgba(0,0,0,1)');
        ctx.fillStyle = grad;
        ctx.fillRect(0, 0, W, H);

        ctx.strokeStyle = 'rgba(255,255,255,0.06)';
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.ellipse(W * 0.5, H * 0.38, W * 0.18, H * 0.26, 0, 0, Math.PI * 2);
        ctx.stroke();
        [W * 0.435, W * 0.565].forEach((ex) => {
          ctx.beginPath();
          ctx.ellipse(ex, H * 0.32, W * 0.028, H * 0.018, 0, 0, Math.PI * 2);
          ctx.stroke();
        });

        const vign2 = ctx.createRadialGradient(W / 2, H / 2, H * 0.2, W / 2, H / 2, H * 0.8);
        vign2.addColorStop(0, 'rgba(0,0,0,0)');
        vign2.addColorStop(1, 'rgba(0,0,0,0.9)');
        ctx.fillStyle = vign2;
        ctx.fillRect(0, 0, W, H);
      }

      animId = requestAnimationFrame(drawFrame);
    };

    animId = requestAnimationFrame(drawFrame);
    return () => cancelAnimationFrame(animId);
  }, [glitching]);

  return (
    <div style={{ position: 'absolute', inset: 0, overflow: 'hidden' }}>
      <canvas
        ref={canvasRef}
        width={1920}
        height={1080}
        style={{
          position: 'absolute', inset: 0,
          width: '100%', height: '100%',
          objectFit: 'cover',
          filter: glitching ? 'contrast(1.15) brightness(1.05)' : 'contrast(1.05)',
          transition: glitching ? 'none' : 'filter 0.3s',
        }}
      />

      {/* CSS RGB ghost overlays during glitch */}
      {glitching && (
        <>
          <div style={{
            position: 'absolute', inset: 0, pointerEvents: 'none',
            background: 'rgba(255,0,40,0.04)',
            transform: 'translateX(-5px)',
            mixBlendMode: 'screen',
          }} />
          <div style={{
            position: 'absolute', inset: 0, pointerEvents: 'none',
            background: 'rgba(0,120,255,0.04)',
            transform: 'translateX(5px)',
            mixBlendMode: 'screen',
          }} />
        </>
      )}

      {/* Persistent scanlines */}
      <div style={{
        position: 'absolute', inset: 0, pointerEvents: 'none',
        backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.12) 2px, rgba(0,0,0,0.12) 4px)',
        zIndex: 2,
      }} />
    </div>
  );
}
