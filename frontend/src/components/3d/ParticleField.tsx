import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import { Points, PointMaterial } from "@react-three/drei";
import * as THREE from "three";

// ── Particle cluster ─────────────────────────────────────────────────────
function Particles({ count = 2000 }: { count?: number }) {
  const ref = useRef<THREE.Points>(null!);

  const positions = useMemo(() => {
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const theta = Math.random() * Math.PI * 2;
      const phi   = Math.acos(2 * Math.random() - 1);
      const r     = 2 + Math.random() * 4;
      pos[i * 3]     = r * Math.sin(phi) * Math.cos(theta);
      pos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      pos[i * 3 + 2] = r * Math.cos(phi);
    }
    return pos;
  }, [count]);

  useFrame((state) => {
    if (!ref.current) return;
    ref.current.rotation.y = state.clock.getElapsedTime() * 0.03;
    ref.current.rotation.x = Math.sin(state.clock.getElapsedTime() * 0.01) * 0.1;
  });

  return (
    <Points ref={ref} positions={positions} frustumCulled={false}>
      <PointMaterial
        transparent
        color="#6366f1"
        size={0.015}
        sizeAttenuation
        depthWrite={false}
        blending={THREE.AdditiveBlending}
      />
    </Points>
  );
}

// ── Neural network connection lines ─────────────────────────────────────
function NeuralConnections({ count = 60 }: { count?: number }) {
  const ref = useRef<THREE.LineSegments>(null!);

  const geometry = useMemo(() => {
    const nodes: THREE.Vector3[] = [];
    for (let i = 0; i < count; i++) {
      nodes.push(
        new THREE.Vector3(
          (Math.random() - 0.5) * 10,
          (Math.random() - 0.5) * 8,
          (Math.random() - 0.5) * 6,
        ),
      );
    }
    const points: number[] = [];
    for (let i = 0; i < nodes.length; i++) {
      for (let j = i + 1; j < nodes.length; j++) {
        const dist = nodes[i].distanceTo(nodes[j]);
        if (dist < 2.5) {
          points.push(
            nodes[i].x, nodes[i].y, nodes[i].z,
            nodes[j].x, nodes[j].y, nodes[j].z,
          );
        }
      }
    }
    const geo = new THREE.BufferGeometry();
    geo.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
    return geo;
  }, [count]);

  useFrame((state) => {
    if (!ref.current) return;
    ref.current.rotation.y = state.clock.getElapsedTime() * 0.04;
    ref.current.rotation.z = state.clock.getElapsedTime() * 0.015;
  });

  return (
    <lineSegments ref={ref} geometry={geometry}>
      <lineBasicMaterial
        color="#4f46e5"
        transparent
        opacity={0.15}
        blending={THREE.AdditiveBlending}
      />
    </lineSegments>
  );
}

// ── Floating node spheres ────────────────────────────────────────────────
function FloatingNodes() {
  const groupRef = useRef<THREE.Group>(null!);
  const nodes = useMemo(
    () =>
      Array.from({ length: 8 }, (_, i) => ({
        pos: new THREE.Vector3(
          Math.cos((i / 8) * Math.PI * 2) * 3,
          Math.sin((i / 8) * Math.PI * 2) * 2,
          (Math.random() - 0.5) * 2,
        ),
        scale: 0.04 + Math.random() * 0.06,
        color: i % 3 === 0 ? "#6366f1" : i % 3 === 1 ? "#00f5ff" : "#bf00ff",
      })),
    [],
  );

  useFrame((state) => {
    if (!groupRef.current) return;
    groupRef.current.rotation.y = state.clock.getElapsedTime() * 0.05;
    groupRef.current.children.forEach((child, i) => {
      child.position.y += Math.sin(state.clock.getElapsedTime() + i) * 0.002;
    });
  });

  return (
    <group ref={groupRef}>
      {nodes.map((n, i) => (
        <mesh key={i} position={n.pos}>
          <sphereGeometry args={[n.scale, 8, 8]} />
          <meshStandardMaterial
            color={n.color}
            emissive={n.color}
            emissiveIntensity={2}
            transparent
            opacity={0.8}
          />
        </mesh>
      ))}
    </group>
  );
}

// ── Scene ────────────────────────────────────────────────────────────────
function Scene() {
  return (
    <>
      <ambientLight intensity={0.1} />
      <pointLight position={[5, 5, 5]} color="#6366f1" intensity={2} />
      <pointLight position={[-5, -5, 5]} color="#00f5ff" intensity={1} />
      <Particles count={2500} />
      <NeuralConnections count={80} />
      <FloatingNodes />
    </>
  );
}

// ── Exported component ───────────────────────────────────────────────────
interface ParticleFieldProps {
  className?: string;
}

export function ParticleField({ className = "" }: ParticleFieldProps) {
  return (
    <div className={`pointer-events-none absolute inset-0 ${className}`}>
      <Canvas
        camera={{ position: [0, 0, 6], fov: 60 }}
        gl={{ antialias: false, alpha: true }}
        dpr={[1, 1.5]}
      >
        <Scene />
      </Canvas>
    </div>
  );
}
