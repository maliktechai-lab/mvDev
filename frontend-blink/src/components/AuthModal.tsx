/**
 * AuthModal
 * =========
 * Login / Register modal that calls the DeepGuard backend auth API.
 *
 * Matches the BLINK design language: monochrome, monospaced, raw borders.
 * Uses the realApi login/register functions which store JWT in localStorage.
 */

import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { login, register, getErrorMessage } from '../api/realApi';

interface Props {
  open: boolean;
  onClose: () => void;
  onAuthed: () => void;
}

type Mode = 'login' | 'register';

export default function AuthModal({ open, onClose, onAuthed }: Props) {
  const [mode, setMode] = useState<Mode>('login');
  const [email, setEmail] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [fullName, setFullName] = useState('');
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const reset = () => {
    setEmail('');
    setUsername('');
    setPassword('');
    setFullName('');
    setError(null);
    setLoading(false);
  };

  const close = () => {
    reset();
    onClose();
  };

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      if (mode === 'login') {
        await login(email, password);
      } else {
        await register(email, username, password, fullName || undefined);
        // Auto-login after registration
        await login(email, password);
      }
      reset();
      onAuthed();
    } catch (err) {
      setError(getErrorMessage(err));
    } finally {
      setLoading(false);
    }
  };

  return (
    <AnimatePresence>
      {open && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2 }}
          style={{
            position: 'fixed',
            inset: 0,
            zIndex: 10000,
            background: 'rgba(0,0,0,0.85)',
            backdropFilter: 'blur(8px)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
          }}
          onClick={close}
        >
          <motion.div
            initial={{ opacity: 0, scale: 0.95, y: 12 }}
            animate={{ opacity: 1, scale: 1, y: 0 }}
            exit={{ opacity: 0, scale: 0.95, y: 12 }}
            transition={{ duration: 0.25, ease: [0.16, 1, 0.3, 1] }}
            onClick={(e) => e.stopPropagation()}
            style={{
              width: '100%',
              maxWidth: 400,
              background: 'var(--bg-1)',
              border: '1px solid var(--border-2)',
              position: 'relative',
              overflow: 'hidden',
            }}
          >
            {/* Top scanline accent */}
            <div style={{
              height: 1,
              background: 'linear-gradient(90deg, transparent, var(--white), transparent)',
              opacity: 0.3,
            }} />

            <div style={{ padding: '32px 28px' }}>
              {/* Header */}
              <div style={{ marginBottom: 28 }}>
                <div style={{
                  fontFamily: 'JetBrains Mono', fontSize: 9,
                  letterSpacing: '0.2em', color: 'var(--grey-2)',
                  textTransform: 'uppercase', marginBottom: 12,
                }}>
                  — DeepGuard Auth
                </div>
                <div style={{
                  fontFamily: 'Syne', fontWeight: 800,
                  fontSize: 28, letterSpacing: '-0.03em',
                  color: 'var(--white)',
                }}>
                  {mode === 'login' ? 'SIGN IN' : 'CREATE ACCOUNT'}
                </div>
              </div>

              {/* Mode toggle */}
              <div style={{
                display: 'flex',
                marginBottom: 24,
                borderBottom: '1px solid var(--border)',
              }}>
                {(['login', 'register'] as const).map((m) => (
                  <button
                    key={m}
                    onClick={() => { setMode(m); setError(null); }}
                    style={{
                      flex: 1,
                      padding: '10px 0',
                      background: 'none',
                      border: 'none',
                      borderBottom: mode === m ? '1px solid var(--white)' : '1px solid transparent',
                      fontFamily: 'JetBrains Mono',
                      fontSize: 11,
                      letterSpacing: '0.1em',
                      textTransform: 'uppercase',
                      color: mode === m ? 'var(--white)' : 'var(--grey-2)',
                      cursor: 'pointer',
                      transition: 'color 0.2s, border-color 0.2s',
                      marginBottom: '-1px',
                    }}
                  >
                    {m === 'login' ? 'Sign In' : 'Register'}
                  </button>
                ))}
              </div>

              {/* Form */}
              <form onSubmit={submit} style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
                <Input
                  label="Email"
                  type="email"
                  value={email}
                  onChange={setEmail}
                  required
                  placeholder="you@example.com"
                />

                {mode === 'register' && (
                  <Input
                    label="Username"
                    type="text"
                    value={username}
                    onChange={setUsername}
                    required
                    placeholder="3-50 chars, alphanumeric"
                  />
                )}

                {mode === 'register' && (
                  <Input
                    label="Full Name (optional)"
                    type="text"
                    value={fullName}
                    onChange={setFullName}
                    placeholder="Your name"
                  />
                )}

                <Input
                  label="Password"
                  type="password"
                  value={password}
                  onChange={setPassword}
                  required
                  placeholder={mode === 'register' ? 'Min 8, 1 uppercase, 1 digit' : '••••••••'}
                />

                {/* Error */}
                <AnimatePresence>
                  {error && (
                    <motion.div
                      initial={{ opacity: 0, y: -4 }}
                      animate={{ opacity: 1, y: 0 }}
                      exit={{ opacity: 0 }}
                      style={{
                        padding: '10px 0',
                        borderTop: '1px solid rgba(255,32,32,0.3)',
                        fontFamily: 'JetBrains Mono',
                        fontSize: 11,
                        color: '#ff2020',
                        letterSpacing: '0.04em',
                      }}
                    >
                      ✕ {error}
                    </motion.div>
                  )}
                </AnimatePresence>

                {/* Submit */}
                <button
                  type="submit"
                  disabled={loading}
                  style={{
                    marginTop: 4,
                    padding: '14px',
                    background: loading ? 'var(--grey-3)' : 'var(--white)',
                    color: 'var(--black)',
                    border: 'none',
                    fontFamily: 'Syne',
                    fontWeight: 800,
                    fontSize: 12,
                    letterSpacing: '0.12em',
                    textTransform: 'uppercase',
                    cursor: loading ? 'wait' : 'pointer',
                    transition: 'background 0.2s',
                  }}
                >
                  {loading
                    ? 'Processing…'
                    : mode === 'login'
                    ? 'Authenticate'
                    : 'Create Account'}
                </button>
              </form>

              {/* Close */}
              <button
                onClick={close}
                style={{
                  marginTop: 16,
                  width: '100%',
                  background: 'none',
                  border: 'none',
                  fontFamily: 'JetBrains Mono',
                  fontSize: 10,
                  color: 'var(--grey-2)',
                  letterSpacing: '0.1em',
                  textTransform: 'uppercase',
                  cursor: 'pointer',
                  transition: 'color 0.2s',
                }}
              >
                ← Cancel
              </button>
            </div>
          </motion.div>
        </motion.div>
      )}
    </AnimatePresence>
  );
}

// ── Reusable input ─────────────────────────────────────────────────────────

function Input({
  label,
  type,
  value,
  onChange,
  required,
  placeholder,
}: {
  label: string;
  type: string;
  value: string;
  onChange: (v: string) => void;
  required?: boolean;
  placeholder?: string;
}) {
  return (
    <div>
      <label style={{
        display: 'block',
        fontFamily: 'JetBrains Mono',
        fontSize: 9,
        letterSpacing: '0.14em',
        color: 'var(--grey-2)',
        textTransform: 'uppercase',
        marginBottom: 6,
      }}>
        {label}
      </label>
      <input
        type={type}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        required={required}
        placeholder={placeholder}
        style={{
          width: '100%',
          padding: '12px 14px',
          background: 'var(--black)',
          border: '1px solid var(--border-2)',
          color: 'var(--white)',
          fontFamily: 'Inter',
          fontSize: 13,
          outline: 'none',
          transition: 'border-color 0.2s',
        }}
        onFocus={(e) => { e.target.style.borderColor = 'var(--border-3)'; }}
        onBlur={(e) => { e.target.style.borderColor = 'var(--border-2)'; }}
      />
    </div>
  );
}
