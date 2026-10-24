/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        brand: {
          50:  "#f0f4ff",
          100: "#e0e9ff",
          200: "#c7d7fe",
          300: "#a5b8fc",
          400: "#818cf8",
          500: "#6366f1",
          600: "#4f46e5",
          700: "#4338ca",
          800: "#3730a3",
          900: "#312e81",
        },
        neon: {
          cyan:    "#00f5ff",
          purple:  "#bf00ff",
          green:   "#00ff88",
          red:     "#ff003c",
          orange:  "#ff6b00",
          blue:    "#0080ff",
          pink:    "#ff0080",
        },
        glass: {
          white:  "rgba(255,255,255,0.05)",
          border: "rgba(255,255,255,0.08)",
          heavy:  "rgba(255,255,255,0.10)",
        },
        dark: {
          950: "#030712",
          900: "#0a0f1e",
          800: "#0d1424",
          700: "#111827",
          600: "#1a2236",
        },
      },
      fontFamily: {
        sans: ["Inter", "system-ui", "sans-serif"],
        mono: ["JetBrains Mono", "monospace"],
      },
      backgroundImage: {
        "grid-pattern":    "linear-gradient(rgba(99,102,241,0.05) 1px, transparent 1px), linear-gradient(90deg, rgba(99,102,241,0.05) 1px, transparent 1px)",
        "radial-glow":     "radial-gradient(ellipse at center, rgba(99,102,241,0.15) 0%, transparent 70%)",
        "glow-cyan":       "radial-gradient(ellipse at center, rgba(0,245,255,0.2) 0%, transparent 70%)",
        "glow-red":        "radial-gradient(ellipse at center, rgba(255,0,60,0.2) 0%, transparent 70%)",
        "glow-green":      "radial-gradient(ellipse at center, rgba(0,255,136,0.2) 0%, transparent 70%)",
        "gradient-brand":  "linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #2563eb 100%)",
        "gradient-fake":   "linear-gradient(135deg, #dc2626 0%, #9f1239 100%)",
        "gradient-real":   "linear-gradient(135deg, #16a34a 0%, #059669 100%)",
        "gradient-dark":   "linear-gradient(180deg, #0a0f1e 0%, #030712 100%)",
      },
      backgroundSize: {
        "grid": "40px 40px",
      },
      boxShadow: {
        "neon-brand":  "0 0 20px rgba(99,102,241,0.4), 0 0 60px rgba(99,102,241,0.2)",
        "neon-cyan":   "0 0 20px rgba(0,245,255,0.4), 0 0 60px rgba(0,245,255,0.15)",
        "neon-red":    "0 0 20px rgba(255,0,60,0.5), 0 0 60px rgba(255,0,60,0.2)",
        "neon-green":  "0 0 20px rgba(0,255,136,0.4), 0 0 60px rgba(0,255,136,0.15)",
        "neon-purple": "0 0 20px rgba(191,0,255,0.4), 0 0 60px rgba(191,0,255,0.15)",
        "glass":       "0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.05)",
        "glass-lg":    "0 24px 64px rgba(0,0,0,0.6), inset 0 1px 0 rgba(255,255,255,0.08)",
        "glow-sm":     "0 0 8px rgba(99,102,241,0.6)",
        "inner-glow":  "inset 0 0 30px rgba(99,102,241,0.1)",
      },
      animation: {
        // Basics
        "pulse-slow":     "pulse 3s cubic-bezier(0.4, 0, 0.6, 1) infinite",
        "spin-slow":      "spin 8s linear infinite",
        "spin-reverse":   "spin-reverse 6s linear infinite",
        "bounce-slow":    "bounce 2s infinite",
        // Glow effects
        "glow-pulse":     "glow-pulse 2s ease-in-out infinite alternate",
        "neon-flicker":   "neon-flicker 3s linear infinite",
        // Scan animations
        "scan-down":      "scan-down 2s ease-in-out infinite",
        "scan-horizontal":"scan-horizontal 1.5s ease-in-out infinite",
        // Float
        "float":          "float 6s ease-in-out infinite",
        "float-delayed":  "float 6s ease-in-out 2s infinite",
        // Orb
        "orb-rotate":     "orb-rotate 8s linear infinite",
        "orb-breathe":    "orb-breathe 3s ease-in-out infinite",
        // Data stream
        "data-stream":    "data-stream 1s linear infinite",
        // Reveal
        "slide-up":       "slide-up 0.6s cubic-bezier(0.16, 1, 0.3, 1) forwards",
        "fade-in":        "fade-in 0.4s ease forwards",
        "scale-in":       "scale-in 0.5s cubic-bezier(0.34, 1.56, 0.64, 1) forwards",
        // Counter
        "count-up":       "count-up 1s ease-out forwards",
        // Bars
        "bar-fill":       "bar-fill 1s cubic-bezier(0.16, 1, 0.3, 1) forwards",
        // Ripple
        "ripple":         "ripple 1.5s linear infinite",
        // Pipeline
        "pipeline-flow":  "pipeline-flow 2s linear infinite",
        // Waveform
        "wave-1":         "wave-anim 0.8s ease-in-out infinite alternate",
        "wave-2":         "wave-anim 0.9s ease-in-out 0.1s infinite alternate",
        "wave-3":         "wave-anim 0.7s ease-in-out 0.2s infinite alternate",
      },
      keyframes: {
        "spin-reverse": {
          from: { transform: "rotate(360deg)" },
          to:   { transform: "rotate(0deg)" },
        },
        "glow-pulse": {
          "0%":   { boxShadow: "0 0 10px rgba(99,102,241,0.3), 0 0 30px rgba(99,102,241,0.1)" },
          "100%": { boxShadow: "0 0 30px rgba(99,102,241,0.6), 0 0 80px rgba(99,102,241,0.3)" },
        },
        "neon-flicker": {
          "0%, 95%, 100%": { opacity: "1" },
          "96%":           { opacity: "0.8" },
          "97%":           { opacity: "1" },
          "98%":           { opacity: "0.7" },
          "99%":           { opacity: "1" },
        },
        "scan-down": {
          "0%":   { transform: "translateY(-100%)", opacity: "0" },
          "10%":  { opacity: "1" },
          "90%":  { opacity: "1" },
          "100%": { transform: "translateY(100%)", opacity: "0" },
        },
        "scan-horizontal": {
          "0%":   { transform: "translateX(-100%)" },
          "100%": { transform: "translateX(100%)" },
        },
        "float": {
          "0%, 100%": { transform: "translateY(0px)" },
          "50%":      { transform: "translateY(-20px)" },
        },
        "orb-rotate": {
          "0%":   { transform: "rotate(0deg)" },
          "100%": { transform: "rotate(360deg)" },
        },
        "orb-breathe": {
          "0%, 100%": { transform: "scale(1)", opacity: "0.8" },
          "50%":      { transform: "scale(1.08)", opacity: "1" },
        },
        "data-stream": {
          "0%":   { backgroundPosition: "0% 0%" },
          "100%": { backgroundPosition: "0% 100%" },
        },
        "slide-up": {
          "0%":   { transform: "translateY(30px)", opacity: "0" },
          "100%": { transform: "translateY(0px)", opacity: "1" },
        },
        "fade-in": {
          "0%":   { opacity: "0" },
          "100%": { opacity: "1" },
        },
        "scale-in": {
          "0%":   { transform: "scale(0.8)", opacity: "0" },
          "100%": { transform: "scale(1)", opacity: "1" },
        },
        "bar-fill": {
          "0%":   { width: "0%" },
          "100%": { width: "var(--bar-width)" },
        },
        "ripple": {
          "0%":   { transform: "scale(1)", opacity: "0.6" },
          "100%": { transform: "scale(2.5)", opacity: "0" },
        },
        "pipeline-flow": {
          "0%":   { backgroundPosition: "0% 50%" },
          "100%": { backgroundPosition: "100% 50%" },
        },
        "wave-anim": {
          "0%":   { transform: "scaleY(0.3)" },
          "100%": { transform: "scaleY(1)" },
        },
        "count-up": {
          "0%":   { opacity: "0", transform: "translateY(10px)" },
          "100%": { opacity: "1", transform: "translateY(0)" },
        },
      },
      transitionTimingFunction: {
        "spring":      "cubic-bezier(0.34, 1.56, 0.64, 1)",
        "smooth":      "cubic-bezier(0.16, 1, 0.3, 1)",
        "anticipate":  "cubic-bezier(0.36, 0, 0.66, -0.56)",
      },
      backdropBlur: {
        xs: "2px",
      },
    },
  },
  plugins: [],
};
