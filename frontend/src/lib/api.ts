import axios, { AxiosError, InternalAxiosRequestConfig } from "axios";
import toast from "react-hot-toast";

const BASE_URL = "/api/v1";

export const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 300_000, // 5 min for large file uploads
  headers: { "Content-Type": "application/json" },
});

// ── Request interceptor — attach access token ──────────────────────────────

apiClient.interceptors.request.use(
  (config: InternalAxiosRequestConfig) => {
    const token = localStorage.getItem("access_token");
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// ── Response interceptor — handle 401, token refresh ──────────────────────

let isRefreshing = false;
let failedQueue: Array<{ resolve: (v: string) => void; reject: (e: unknown) => void }> = [];

const processQueue = (error: unknown, token: string | null = null) => {
  failedQueue.forEach((prom) => {
    if (error) prom.reject(error);
    else prom.resolve(token!);
  });
  failedQueue = [];
};

apiClient.interceptors.response.use(
  (response) => response,
  async (error: AxiosError) => {
    const originalRequest = error.config as InternalAxiosRequestConfig & { _retry?: boolean };

    if (error.response?.status === 401 && !originalRequest._retry) {
      if (isRefreshing) {
        return new Promise<string>((resolve, reject) => {
          failedQueue.push({ resolve, reject });
        })
          .then((token) => {
            originalRequest.headers.Authorization = `Bearer ${token}`;
            return apiClient(originalRequest);
          })
          .catch((err) => Promise.reject(err));
      }

      originalRequest._retry = true;
      isRefreshing = true;

      try {
        const refreshToken = localStorage.getItem("refresh_token");
        if (!refreshToken) throw new Error("No refresh token");

        const response = await axios.post(`${BASE_URL}/auth/refresh`, {
          refresh_token: refreshToken,
        });

        const { access_token, refresh_token: newRefresh } = response.data;
        localStorage.setItem("access_token", access_token);
        localStorage.setItem("refresh_token", newRefresh);

        processQueue(null, access_token);
        originalRequest.headers.Authorization = `Bearer ${access_token}`;
        return apiClient(originalRequest);
      } catch (refreshError) {
        processQueue(refreshError, null);
        localStorage.removeItem("access_token");
        localStorage.removeItem("refresh_token");
        window.location.href = "/login";
        return Promise.reject(refreshError);
      } finally {
        isRefreshing = false;
      }
    }

    // Surface error message to user
    const detail = (error.response?.data as { detail?: string })?.detail;
    if (detail && error.response?.status !== 401) {
      toast.error(detail);
    }

    return Promise.reject(error);
  }
);

// ── API helpers ────────────────────────────────────────────────────────────

export const authApi = {
  register: (data: object) => apiClient.post("/auth/register", data),
  login: (data: object) => apiClient.post("/auth/login", data),
  refresh: (token: string) => apiClient.post("/auth/refresh", { refresh_token: token }),
  logout: (token: string) => apiClient.post("/auth/logout", { refresh_token: token }),
  me: () => apiClient.get("/auth/me"),
};

export const jobsApi = {
  list: (params?: object) => apiClient.get("/jobs", { params }),
  get: (id: string) => apiClient.get(`/jobs/${id}`),
  delete: (id: string) => apiClient.delete(`/jobs/${id}`),
};

export const detectApi = {
  image: (
    file: File,
    generateHeatmap: boolean = true,
    onProgress?: (pct: number) => void,
  ) => {
    const formData = new FormData();
    formData.append("file", file);
    return apiClient.post(
      `/detect/image?generate_heatmap=${generateHeatmap}`,
      formData,
      {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (e) => {
          if (onProgress && e.total) {
            onProgress(Math.round((e.loaded / e.total) * 100));
          }
        },
      },
    );
  },

  video: (
    file: File,
    onProgress?: (pct: number) => void,
  ) => {
    const formData = new FormData();
    formData.append("file", file);
    return apiClient.post(
      `/detect/video`,
      formData,
      {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (e) => {
          if (onProgress && e.total) {
            onProgress(Math.round((e.loaded / e.total) * 100));
          }
        },
      },
    );
  },

  audio: (
    file: File,
    onProgress?: (pct: number) => void,
  ) => {
    const formData = new FormData();
    formData.append("file", file);
    return apiClient.post(
      `/detect/audio`,
      formData,
      {
        headers: { "Content-Type": "multipart/form-data" },
        onUploadProgress: (e) => {
          if (onProgress && e.total) {
            onProgress(Math.round((e.loaded / e.total) * 100));
          }
        },
      },
    );
  },
};
