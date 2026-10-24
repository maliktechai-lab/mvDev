import { useRef, useMemo } from "react";
import { Canvas, useFrame } from "@react-three/fiber";
import * as THREE from "three";

// ── Inner pulsing core ───────────────────────────────────────────────────
function OrbCore() {
  const meshRef = useRef<THREE.Mesh>(null!);

  useFrame((state) => {
    const t = state.clock.getElapsedTime();
    if (!meshRef.current) return;
    const scale = 1 + Math.sin(t * 2.5) * 0.08;
    meshRef.current.scale.setScalar(scale);
    (meshRef.current.material as THREE.MeshStandardMaterial).emissiveIntensity =
      1.5 + Math.sin(t * 2) * 0.5;
  });

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[0.4, 32, 32]} />
      <meshStandardMaterial
        color="#6366f1"
        emissive="#6366f1"
        emissiveIntensity={2}
        transparent
        opacity={0.9}
      />
    </mesh>
  );
}

// ── Orbiting ring ────────────────────────────────────────────────────────
function OrbRing({
  radius,
  color,
  speed,
  tilt,
}: {
  radius: number;
  color: string;
  speed: number;
  tilt: number;
}) {
  const ref = useRef<THREE.Group>(null!);

  useFrame((state) => {
    if (!ref.current) return;
    ref.current.rotation.z = state.clock.getElapsedTime() * speed;
  });

  return (
    <group ref={ref} rotation={[tilt, 0, 0]}>
      <mesh>
        <torusGeometry args={[radius, 0.008, 8, 64]} />
        <meshStandardMaterial
          color={color}
          emissive={color}
          emissiveIntensity={1.5}
          transparent
          opacity={0.7}
        />
      </mesh>
    </group>
  );
}

// ── Orbiting particle on ring ────────────────────────────────────────────
function OrbitingParticle({
  radius,
  speed,
  color,
  tilt,
  offset,
}: {
  radius: number;
  speed: number;
  color: string;
  tilt: number;
  offset: number;
}) {
  const meshRef = useRef<THREE.Mesh>(null!);

  useFrame((state) => {
    const t = state.clock.getElapsedTime() * speed + offset;
    const x = Math.cos(t) * radius;
    const y = Math.sin(t) * radius;
    if (meshRef.current) {
      meshRef.current.position.set(x, y * Math.cos(tilt), y * Math.sin(tilt));
    }
  });

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[0.04, 8, 8]} />
      <meshStandardMaterial
        color={color}
        emissive={color}
        emissiveIntensity={3}
        transparent
        opacity={0.95}
      />
    </mesh>
  );
}

// ── Energy tendrils ──────────────────────────────────────────────────────
function EnergyTendrils() {
  const ref = useRef<THREE.Points>(null!);

  const positions = useMemo(() => {
    const count = 300;
    const pos = new Float32Array(count * 3);
    for (let i = 0; i < count; i++) {
      const t     = i / count;
      const angle = t * Math.PI * 8;
      const r     = 0.5 + t * 0.8;
      pos[i * 3]     = Math.cos(angle) * r;
      pos[i * 3 + 1] = Math.sin(angle) * r;
      pos[i * 3 + 2] = (t - 0.5) * 2;
    }
    return pos;
  }, []);

  useFrame((state) => {
    if (!ref.current) return;
    ref.current.rotation.z = state.clock.getElapsedTime() * 0.5;
    ref.current.rotation.y = state.clock.getElapsedTime() * 0.3;
  });

  return (
    <points ref={ref}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          array={positions}
          count={positions.length / 3}
          itemSize={3}
        />
      </bufferGeometry>
      <pointsMaterial
        color="#00f5ff"
        size={0.025}
        transparent
        opacity={0.6}
        blending={THREE.AdditiveBlending}
        depthWrite={false}
      />
    </points>
  );
}

// ── Full scene ───────────────────────────────────────────────────────────
function OrbScene() {
  return (
    <>
      <ambientLight intensity={0.3} />
      <pointLight position={[2, 2, 2]} color="#6366f1" intensity={3} />
      <pointLight position={[-2, -2, 2]} color="#00f5ff" intensity={2} />
      <pointLight position={[0, 0, -2]} color="#bf00ff" intensity={1.5} />

      <OrbCore />
      <OrbRing radius={0.75} color="#6366f1" speed={1.2}  tilt={0.3} />
      <OrbRing radius={0.95} color="#00f5ff" speed={-0.8} tilt={1.0} />
      <OrbRing radius={1.15} color="#bf00ff" speed={0.5}  tilt={1.8} />

      <OrbitingParticle radius={0.75} speed={1.2}  color="#ffffff" tilt={0.3} offset={0} />
      <OrbitingParticle radius={0.95} speed={-0.8} color="#00f5ff" tilt={1.0} offset={1} />
      <OrbitingParticle radius={1.15} speed={0.5}  color="#bf00ff" tilt={1.8} offset={2} />

      <EnergyTendrils />
    </>
  );
}

// ── Exported component ───────────────────────────────────────────────────
interface NeuralNetworkOrbProps {
  size?: number;
  className?: string;
}

export function NeuralNetworkOrb({ size = 200, className = "" }: NeuralNetworkOrbProps) {
  return (
    <div
      className={className}
      style={{ width: size, height: size }}
    >
      <Canvas
        camera={{ position: [0, 0, 2.5], fov: 50 }}
        gl={{ antialias: true, alpha: true }}
        dpr={[1, 2]}
      >
        <OrbScene />
      </Canvas>
    </div>
  );
}
