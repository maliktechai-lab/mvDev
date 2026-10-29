import type { AnalysisResult, MediaType } from '../types';

const delay = (ms: number) => new Promise(resolve => setTimeout(resolve, ms));

export async function analyzeMedia(
  file: File,
  onProgress?: (stage: string, progress: number) => void
): Promise<AnalysisResult> {
  const mediaType: MediaType = file.type.startsWith('video/') ? 'video' : 'image';

  // Simulate upload
  onProgress?.('Uploading media...', 10);
  await delay(800);

  // Simulate preprocessing
  onProgress?.('Preprocessing & decoding...', 25);
  await delay(600);

  // Simulate spatial branch
  onProgress?.('Running spatial branch (EfficientNet-B4)...', 50);
  await delay(1000);

  // Simulate frequency branch
  onProgress?.('Running frequency branch (FFT)...', 70);
  await delay(800);

  // Simulate fusion
  onProgress?.('Fusing branch predictions...', 85);
  await delay(500);

  // Simulate Grad-CAM
  onProgress?.('Generating Grad-CAM++ heatmap...', 95);
  await delay(600);

  onProgress?.('Complete', 100);
  await delay(200);

  // Random result with bias toward fake
  const isFake = Math.random() > 0.4;
  const spatialScore = isFake ? 0.65 + Math.random() * 0.3 : 0.1 + Math.random() * 0.3;
  const frequencyScore = isFake ? 0.60 + Math.random() * 0.35 : 0.08 + Math.random() * 0.28;
  const fusedScore = (spatialScore * 0.6 + frequencyScore * 0.4);

  const frameResults = mediaType === 'video' ? Array.from({ length: 8 }, (_, i) => ({
    frameIndex: i,
    timestamp: i * 0.5,
    spatialScore: Math.max(0, Math.min(1, spatialScore + (Math.random() - 0.5) * 0.2)),
    frequencyScore: Math.max(0, Math.min(1, frequencyScore + (Math.random() - 0.5) * 0.2)),
    fusedScore: Math.max(0, Math.min(1, fusedScore + (Math.random() - 0.5) * 0.15)),
    label: (fusedScore > 0.5 ? 'FAKE' : 'REAL') as 'FAKE' | 'REAL',
  })) : undefined;

  return {
    id: `analysis-${Date.now()}`,
    mediaType,
    label: isFake ? 'FAKE' : 'REAL',
    confidence: fusedScore,
    branchScores: { spatial: spatialScore, frequency: frequencyScore, fused: fusedScore },
    frameResults,
    heatmapUrl: undefined,
    processingTime: 4500,
    warnings: [
      'Results are model-based estimates, not forensic proof.',
      ...(mediaType === 'video' ? ['Frame sampling may miss brief manipulations.'] : []),
    ],
    metadata: {
      filename: file.name,
      fileSize: file.size,
      dimensions: { width: 1920, height: 1080 },
      ...(mediaType === 'video' ? { duration: 4.0, framesAnalyzed: 8 } : {}),
    },
  };
}
