import { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';
import { motion, AnimatePresence } from 'framer-motion';

interface Props {
  onFileSelected: (file: File) => void;
  isAnalyzing: boolean;
}

const ACCEPTED = {
  'image/jpeg': [], 'image/png': [], 'image/webp': [], 'image/bmp': [],
  'video/mp4':  [], 'video/webm': [], 'video/avi': [], 'video/quicktime': [],
};

const FORMATS = [
  { ext: 'JPG',  color: '#ffffff' },
  { ext: 'PNG',  color: '#a78bfa' },
  { ext: 'WebP', color: '#3b82f6' },
  { ext: 'MP4',  color: '#ff2020' },
  { ext: 'WebM', color: '#facc15' },
  { ext: 'MOV',  color: '#00ff88' },
];

export default function UploadZone({ onFileSelected, isAnalyzing }: Props) {
  const [preview, setPreview] = useState<{
    url: string; type: 'image' | 'video'; name: string; size: number;
  } | null>(null);

  const onDrop = useCallback((files: File[]) => {
    const f = files[0];
    if (!f) return;
    const url = URL.createObjectURL(f);
    setPreview({ url, type: f.type.startsWith('video/') ? 'video' : 'image', name: f.name, size: f.size });
    onFileSelected(f);
  }, [onFileSelected]);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop, accept: ACCEPTED, maxFiles: 1,
    maxSize: 100 * 1024 * 1024, disabled: isAnalyzing,
  });

  const accentColor = isDragReject
    ? '#ff2020'
    : isDragActive
    ? '#a78bfa'
    : preview
    ? 'rgba(255,255,255,0.18)'
    : 'rgba(255,255,255,0.08)';

  return (
    <motion.div
      {...(getRootProps() as object)}
      animate={{
        borderColor: accentColor,
        boxShadow: isDragActive
          ? '0 0 40px rgba(167,139,250,0.18), inset 0 0 40px rgba(167,139,250,0.04)'
          : '0 0 0px transparent',
      }}
      transition={{ duration: 0.25 }}
      style={{
        border: `1px solid ${accentColor}`,
        background: isDragActive ? 'rgba(167,139,250,0.04)' : '#080808',
        cursor: isAnalyzing ? 'default' : 'pointer',
        minHeight: 280,
        position: 'relative',
        overflow: 'hidden',
      }}
    >
      <input {...getInputProps()} />

      {/* Animated scanline */}
      {!preview && !isAnalyzing && (
        <motion.div
          animate={{ top: ['0%', '102%'] }}
          transition={{ repeat: Infinity, duration: 3, ease: 'linear', repeatDelay: 1.5 }}
          style={{
            position: 'absolute', left: 0, right: 0, height: 1,
            background: isDragActive
              ? 'linear-gradient(90deg, transparent, rgba(167,139,250,0.8), transparent)'
              : 'linear-gradient(90deg, transparent, rgba(255,255,255,0.15), transparent)',
            pointerEvents: 'none', zIndex: 2,
          }}
        />
      )}

      {/* Corner brackets */}
      {(['tl','tr','bl','br'] as const).map((c) => (
        <motion.div
          key={c}
          animate={{ opacity: isDragActive ? 1 : 0.5, scale: isDragActive ? 1.1 : 1 }}
          transition={{ duration: 0.2 }}
          style={{
            position: 'absolute', width: 18, height: 18,
            top:    c[0] === 't' ? 12 : 'auto',
            bottom: c[0] === 'b' ? 12 : 'auto',
            left:   c[1] === 'l' ? 12 : 'auto',
            right:  c[1] === 'r' ? 12 : 'auto',
            borderTop:    c[0] === 't' ? `1px solid ${accentColor}` : 'none',
            borderBottom: c[0] === 'b' ? `1px solid ${accentColor}` : 'none',
            borderLeft:   c[1] === 'l' ? `1px solid ${accentColor}` : 'none',
            borderRight:  c[1] === 'r' ? `1px solid ${accentColor}` : 'none',
            zIndex: 3, pointerEvents: 'none',
            transition: 'border-color 0.25s',
          }}
        />
      ))}

      {/* CRT scanlines texture */}
      <div style={{
        position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 1,
        backgroundImage: 'repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0,0,0,0.08) 2px, rgba(0,0,0,0.08) 4px)',
      }} />

      {/* Grid overlay */}
      <div style={{
        position: 'absolute', inset: 0, pointerEvents: 'none', zIndex: 1,
        backgroundImage: `
          linear-gradient(rgba(255,255,255,0.02) 1px, transparent 1px),
          linear-gradient(90deg, rgba(255,255,255,0.02) 1px, transparent 1px)
        `,
        backgroundSize: '40px 40px',
      }} />

      <AnimatePresence mode="wait">

        {/* Preview state */}
        {preview ? (
          <motion.div
            key="preview"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
              padding: 24, display: 'flex', flexDirection: 'column',
              alignItems: 'center', gap: 16, position: 'relative', zIndex: 4,
            }}
          >
            {preview.type === 'image' ? (
              <img
                src={preview.url} alt="Preview"
                style={{
                  maxHeight: 180, maxWidth: '100%',
                  objectFit: 'contain',
                  filter: 'contrast(1.05)',
                }}
              />
            ) : (
              <video
                src={preview.url} muted
                style={{ maxHeight: 180, maxWidth: '100%' }}
              />
            )}
            <div style={{ textAlign: 'center' }}>
              <div style={{
                fontFamily: 'JetBrains Mono', fontSize: 11,
                color: 'var(--white)', letterSpacing: '0.06em', marginBottom: 4,
              }}>
                {preview.name}
              </div>
              <div style={{
                fontFamily: 'JetBrains Mono', fontSize: 9,
                color: 'var(--grey-2)', letterSpacing: '0.1em', textTransform: 'uppercase',
              }}>
                {preview.type} · {(preview.size / 1024 / 1024).toFixed(2)} MB
              </div>
            </div>
            {!isAnalyzing && (
              <div style={{
                fontFamily: 'JetBrains Mono', fontSize: 9,
                color: 'var(--grey-3)', letterSpacing: '0.1em', textTransform: 'uppercase',
              }}>
                Drop another to replace
              </div>
            )}
          </motion.div>
        ) : (
          /* Empty / drag state */
          <motion.div
            key="empty"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
              padding: '48px 32px',
              display: 'flex', flexDirection: 'column',
              alignItems: 'center', gap: 20, textAlign: 'center',
              position: 'relative', zIndex: 4,
            }}
          >
            {/* Upload icon */}
            <motion.div
              animate={isDragActive
                ? { scale: 1.2, rotate: -8 }
                : { scale: 1, rotate: 0 }
              }
              transition={{ type: 'spring', stiffness: 280, damping: 18 }}
            >
              <svg
                width="48" height="48" viewBox="0 0 24 24" fill="none"
                stroke={isDragActive ? '#a78bfa' : 'var(--grey-2)'}
                strokeWidth="1.2"
                style={{ transition: 'stroke 0.2s', display: 'block' }}
              >
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" strokeLinecap="round" strokeLinejoin="round" />
                <polyline points="17 8 12 3 7 8" strokeLinecap="round" strokeLinejoin="round" />
                <line x1="12" y1="3" x2="12" y2="15" strokeLinecap="round" />
              </svg>
            </motion.div>

            {/* Instruction text */}
            <div>
              <div style={{
                fontFamily: 'Syne', fontWeight: 800,
                fontSize: 18, letterSpacing: '-0.02em',
                color: isDragActive ? '#a78bfa' : 'var(--white)',
                marginBottom: 8, transition: 'color 0.2s',
              }}>
                {isDragActive ? 'Release to scan' : 'Drop media here'}
              </div>
              <div style={{
                fontFamily: 'Inter', fontSize: 13, color: 'var(--grey-2)', lineHeight: 1.6,
              }}>
                or <span style={{ color: 'var(--white)', textDecoration: 'underline', textUnderlineOffset: 3 }}>click to browse</span>
              </div>
            </div>

            {/* Format tags */}
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6, justifyContent: 'center' }}>
              {FORMATS.map(({ ext, color }) => (
                <motion.span
                  key={ext}
                  whileHover={{ y: -3, scale: 1.06 }}
                  style={{
                    fontFamily: 'JetBrains Mono', fontSize: 10,
                    letterSpacing: '0.1em',
                    padding: '4px 10px',
                    border: `1px solid ${color}30`,
                    color: color,
                    background: `${color}0a`,
                  }}
                >
                  {ext}
                </motion.span>
              ))}
            </div>

            <div style={{
              fontFamily: 'JetBrains Mono', fontSize: 9,
              color: 'var(--grey-3)', letterSpacing: '0.14em', textTransform: 'uppercase',
            }}>
              Max 100 MB
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Analyzing overlay */}
      <AnimatePresence>
        {isAnalyzing && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            style={{
              position: 'absolute', inset: 0, zIndex: 10,
              display: 'flex', flexDirection: 'column',
              alignItems: 'center', justifyContent: 'center', gap: 16,
              background: 'rgba(0,0,0,0.85)', backdropFilter: 'blur(4px)',
            }}
          >
            {/* Dual-ring spinner */}
            <div style={{ position: 'relative', width: 48, height: 48 }}>
              <div style={{
                position: 'absolute', inset: 0, borderRadius: '50%',
                border: '2px solid transparent',
                borderTopColor: '#a78bfa',
                borderRightColor: '#3b82f6',
                animation: 'spin 1s linear infinite',
              }} />
              <div style={{
                position: 'absolute', inset: 8, borderRadius: '50%',
                border: '2px solid transparent',
                borderTopColor: '#ff2020',
                animation: 'spin 1.4s linear infinite reverse',
              }} />
            </div>
            <span style={{
              fontFamily: 'JetBrains Mono', fontSize: 10,
              letterSpacing: '0.2em', color: 'var(--grey-2)', textTransform: 'uppercase',
            }}>
              Scanning…
            </span>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
}
