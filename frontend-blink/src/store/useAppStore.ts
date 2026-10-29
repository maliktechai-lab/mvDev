import { create } from 'zustand';
import type { AppState, AnalysisStatus, AnalysisResult } from '../types';

export const useAppStore = create<AppState>((set, get) => ({
  status:        'idle',
  uploadedFile:  null,
  previewUrl:    null,
  result:        null,
  error:         null,
  progressStage: '',
  progressPct:   0,
  pollingJobId:  null,

  setFile: (file: File) => {
    const prev = get().previewUrl;
    if (prev) URL.revokeObjectURL(prev);
    const previewUrl = URL.createObjectURL(file);
    set({
      uploadedFile: file,
      previewUrl,
      status:       'idle',
      result:       null,
      error:        null,
      progressStage: '',
      progressPct:  0,
      pollingJobId: null,
    });
  },

  setStatus: (status: AnalysisStatus) => set({ status }),

  setResult: (result: AnalysisResult) =>
    set({ result, status: 'complete', progressPct: 100 }),

  setError: (error: string) =>
    set({ error, status: 'error', progressStage: '', progressPct: 0 }),

  setProgress: (stage: string, pct: number) =>
    set({ progressStage: stage, progressPct: pct }),

  reset: () => {
    const prev = get().previewUrl;
    if (prev) URL.revokeObjectURL(prev);
    set({
      status:        'idle',
      uploadedFile:  null,
      previewUrl:    null,
      result:        null,
      error:         null,
      progressStage: '',
      progressPct:   0,
      pollingJobId:  null,
    });
  },
}));
