import { useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import * as THREE from 'three';

// Mouse state shared across Three.js components
const mouse = { x: 0, y: 0, vx: 0, vy: 0 };

function MouseTracker() {
  useEffect(() => {
    let px = 0, py = 0;
    const fn = (e: MouseEvent) => {
      const nx = (e.clientX / window.innerWidth) * 2 - 1;
      const ny = -(e.clientY / window.innerHeight) * 2 + 1;
      mouse.vx = nx - px;
      mouse.vy = ny - py;
      mouse.x = nx;
      mouse.y = ny;
      px = nx; py = ny;
    };
    window.addEventListener('mousemove', fn);
    return () => window.removeEventListener('mousemove', fn);
  }, []);
  return null;
}

function ParticleField() {
  const pts = useRef<THREE.Points>(null);

  const N = 220;
  const CONNECT_DIST = 2.6;

  const { positions, colors, origPositions, phases, speeds } = useMemo(() => {
    const positions = new Float32Array(N * 3);
    const colors = new Float32Array(N * 3);
    const origPositions = new Float32Array(N * 3);
    const phases = new Float32Array(N);
    const speeds = new Float32Array(N);

    const palette: [number, number, number][] = [
      [0.0, 0.898, 1.0],    // cyan
      [0.655, 0.545, 0.98], // violet
      [0.98, 0.443, 0.514], // rose
      [0.063, 0.722, 0.506],// emerald
      [0.965, 0.620, 0.043],// amber
    ];

    for (let i = 0; i < N; i++) {
      // Spherical shell distribution — denser toward edges
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      const r = 2.5 + Math.pow(Math.random(), 0.7) * 8.5;

      const x = r * Math.sin(phi) * Math.cos(theta);
      const y = r * Math.sin(phi) * Math.sin(theta) * 0.45;
      const z = r * Math.cos(phi) * 0.55 - 1;

      positions[i * 3]     = x;
      positions[i * 3 + 1] = y;
      positions[i * 3 + 2] = z;
      origPositions[i * 3]     = x;
      origPositions[i * 3 + 1] = y;
      origPositions[i * 3 + 2] = z;

      const c = palette[Math.floor(Math.random() * palette.length)];
      colors[i * 3]     = c[0];
      colors[i * 3 + 1] = c[1];
      colors[i * 3 + 2] = c[2];

      phases[i] = Math.random() * Math.PI * 2;
      speeds[i] = 0.3 + Math.random() * 0.7;
    }

    return { positions, colors, origPositions, phases, speeds };
  }, []);

  // Precompute edges
  const linePositions = useMemo(() => {
    const nodes: THREE.Vector3[] = [];
    for (let i = 0; i < N; i++) {
      nodes.push(new THREE.Vector3(origPositions[i*3], origPositions[i*3+1], origPositions[i*3+2]));
    }
    const lv: number[] = [];
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        if (nodes[i].distanceTo(nodes[j]) < CONNECT_DIST) {
          lv.push(nodes[i].x, nodes[i].y, nodes[i].z,
                  nodes[j].x, nodes[j].y, nodes[j].z);
        }
      }
    }
    return new Float32Array(lv);
  }, [origPositions]);

  const linesRef = useRef<THREE.LineSegments>(null);
  const groupRef = useRef<THREE.Group>(null);

  useFrame(({ clock }) => {
    const t = clock.getElapsedTime();
    const g = groupRef.current;
    const p = pts.current;
    if (!g || !p) return;

    // Mouse-driven rotation
    const targetRotY = mouse.x * 0.25;
    const targetRotX = mouse.y * -0.12;
    g.rotation.y += (targetRotY - g.rotation.y) * 0.04;
    g.rotation.x += (targetRotX - g.rotation.x) * 0.03;
    // Slow ambient drift
    g.rotation.y += 0.0018;

    // Per-particle oscillation + mouse repulsion
    const pos = p.geometry.attributes.position.array as Float32Array;
    const mx3d = mouse.x * 5;
    const my3d = mouse.y * 3;

    for (let i = 0; i < N; i++) {
      const ox = origPositions[i*3];
      const oy = origPositions[i*3+1];
      const oz = origPositions[i*3+2];

      // Gentle float
      const dy = Math.sin(t * speeds[i] * 0.7 + phases[i]) * 0.12;
      const dx = Math.cos(t * speeds[i] * 0.4 + phases[i]) * 0.06;

      // Mouse repulsion (in camera-projected space)
      const rpx = ox - mx3d;
      const rpy = oy - my3d;
      const dist = Math.sqrt(rpx*rpx + rpy*rpy);
      const repel = dist < 2.5 ? (1 - dist / 2.5) * 0.6 : 0;

      pos[i*3]     = ox + dx + (repel > 0 ? (rpx / dist) * repel : 0);
      pos[i*3+1]   = oy + dy + (repel > 0 ? (rpy / dist) * repel : 0);
      pos[i*3+2]   = oz;
    }
    p.geometry.attributes.position.needsUpdate = true;

    if (linesRef.current) {
      linesRef.current.rotation.copy(g.rotation);
    }
  });

  return (
    <group ref={groupRef}>
      <points ref={pts}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[positions, 3]} />
          <bufferAttribute attach="attributes-color"    args={[colors, 3]} />
        </bufferGeometry>
        <pointsMaterial
          size={0.055}
          vertexColors
          transparent
          opacity={0.85}
          sizeAttenuation
          depthWrite={false}
        />
      </points>
      <lineSegments ref={linesRef}>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" args={[linePositions, 3]} />
        </bufferGeometry>
        <lineBasicMaterial color="#7c3aed" transparent opacity={0.055} depthWrite={false} />
      </lineSegments>
    </group>
  );
}

function OrbitalRings() {
  const rings = useRef<THREE.Group>(null);
  useFrame(({ clock }) => {
    if (!rings.current) return;
    const t = clock.getElapsedTime();
    const children = rings.current.children;
    // Individual ring speeds
    const speeds = [0.055, 0.032, 0.018, 0.012];
    const tilts  = [0.5, 1.3, 2.1, 0.9];
    children.forEach((child, i) => {
      child.rotation.z = t * speeds[i];
      child.rotation.x = tilts[i] + Math.sin(t * 0.1 + i) * 0.05;
      child.rotation.y = Math.sin(t * 0.07 + i * 1.3) * 0.15;
    });
    // Add mouse influence to whole group
    rings.current.rotation.y += (mouse.x * 0.08 - rings.current.rotation.y) * 0.02;
  });

  const ringDefs = [
    { r: 4.8,  color: '#a78bfa', opacity: 0.09 },
    { r: 7.0,  color: '#00e5ff', opacity: 0.07 },
    { r: 9.2,  color: '#fb7185', opacity: 0.055 },
    { r: 11.5, color: '#10b981', opacity: 0.04 },
  ];

  return (
    <group ref={rings}>
      {ringDefs.map((def, i) => (
        <mesh key={i}>
          <torusGeometry args={[def.r, 0.006 + i * 0.002, 6, 160]} />
          <meshBasicMaterial color={def.color} transparent opacity={def.opacity} depthWrite={false} />
        </mesh>
      ))}
    </group>
  );
}

function DepthFog() {
  // A subtle pulsing glow sphere in the center
  const ref = useRef<THREE.Mesh>(null);
  useFrame(({ clock }) => {
    if (!ref.current) return;
    const mat = ref.current.material as THREE.MeshBasicMaterial;
    mat.opacity = 0.025 + Math.sin(clock.getElapsedTime() * 0.4) * 0.01;
    ref.current.rotation.y += 0.002;
  });
  return (
    <mesh ref={ref}>
      <sphereGeometry args={[2.2, 32, 32]} />
      <meshBasicMaterial color="#a78bfa" transparent opacity={0.025} side={THREE.BackSide} depthWrite={false} />
    </mesh>
  );
}

export default function NeuralBackground() {
  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        zIndex: -10,
        background: 'var(--bg)',
      }}
    >
      {/* Radial gradient overlays */}
      <div style={{
        position: 'absolute', inset: 0,
        background:
          'radial-gradient(ellipse 70% 55% at 15% 15%, rgba(167,139,250,0.07) 0%, transparent 65%),' +
          'radial-gradient(ellipse 55% 45% at 85% 85%, rgba(0,229,255,0.06) 0%, transparent 65%),' +
          'radial-gradient(ellipse 40% 35% at 50% 50%, rgba(251,113,133,0.03) 0%, transparent 60%)',
        pointerEvents: 'none',
      }} />

      <Canvas
        camera={{ position: [0, 0, 11], fov: 52 }}
        dpr={[1, 1.5]}
        gl={{ antialias: false, alpha: true, powerPreference: 'low-power' }}
      >
        <MouseTracker />
        <ParticleField />
        <OrbitalRings />
        <DepthFog />
      </Canvas>

      {/* Bottom fade */}
      <div style={{
        position: 'absolute', bottom: 0, left: 0, right: 0, height: 200,
        background: 'linear-gradient(to bottom, transparent, var(--bg))',
        pointerEvents: 'none',
      }} />
    </div>
  );
}
