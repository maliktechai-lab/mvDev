// ─── BLINK — Curated image set ────────────────────────────────
// All images from Unsplash (free for commercial use — Unsplash License).

const CDN = 'https://images.unsplash.com';
const CARD = 'w=800&q=82&fit=crop&crop=faces,top';

export const IMAGES = {
  // Hero background — neon-lit face, feeds GlitchFace canvas
  heroBg: `${CDN}/photo-1617791160505-6f00504e3519?w=1920&q=88&fit=crop&crop=faces,center`,

  // Gallery — 8 editorial portraits
  gallery: [
    `${CDN}/photo-1508214751196-bcfd4ca60f91?${CARD}`,
    `${CDN}/photo-1507003211169-0a1dd7228f2d?${CARD}`,
    `${CDN}/photo-1488426862026-3ee34a7d66df?${CARD}`,
    `${CDN}/photo-1600486913747-55e5470d6f40?${CARD}`,
    `${CDN}/photo-1567532939604-b6b5b0db2604?${CARD}`,
    `${CDN}/photo-1552058544-f2b08422138a?${CARD}`,
    `${CDN}/photo-1573496359142-b8d87734a5a2?${CARD}`,
    `${CDN}/photo-1619895862022-09114b41f16f?${CARD}`,
  ],
};

export const GALLERY_LABELS = [
  { label: 'AUTHENTIC', pct: 96, color: '#00ff88' },
  { label: 'DEEPFAKE',  pct: 98, color: '#ff2020' },
  { label: 'AUTHENTIC', pct: 89, color: '#00ff88' },
  { label: 'DEEPFAKE',  pct: 91, color: '#ff2020' },
  { label: 'DEEPFAKE',  pct: 94, color: '#ff2020' },
  { label: 'AUTHENTIC', pct: 83, color: '#00ff88' },
  { label: 'AUTHENTIC', pct: 92, color: '#00ff88' },
  { label: 'DEEPFAKE',  pct: 87, color: '#ff2020' },
];
