/**
 * Auth store with Zustand persist.
 * Ported from frontend/src/store/authStore.ts to work with frontend-blink's realApi.
 *
 * Changes from the original:
 * - Uses `dg_access_token` / `dg_refresh_token` localStorage keys (matching realApi.ts)
 * - login() calls realApi.login() which stores tokens
 * - fetchMe() uses the apiClient from realApi.ts
 * - logout() calls realApi.logout() then clears state
 */

import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import { login as apiLogin, register as apiRegister, logout as apiLogout, isLoggedIn } from '../api/realApi';
import { apiClient } from '../api/realApi';

export interface User {
  id: string;
  email: string;
  username: string;
  full_name: string | null;
  is_active: boolean;
  is_admin: boolean;
  created_at: string;
}

interface AuthState {
  user: User | null;
  isAuthenticated: boolean;
  isLoading: boolean;

  login: (email: string, password: string) => Promise<void>;
  register: (data: { email: string; username: string; password: string; full_name?: string }) => Promise<void>;
  logout: () => Promise<void>;
  fetchMe: () => Promise<void>;
}

export const useAuthStore = create<AuthState>()(
  persist(
    (set, get) => ({
      user: null,
      isAuthenticated: false,
      isLoading: false,

      login: async (email, password) => {
        set({ isLoading: true });
        try {
          await apiLogin(email, password);
          set({ isAuthenticated: true });
          await get().fetchMe();
        } finally {
          set({ isLoading: false });
        }
      },

      register: async (data) => {
        set({ isLoading: true });
        try {
          await apiRegister(data.email, data.username, data.password, data.full_name);
        } finally {
          set({ isLoading: false });
        }
      },

      logout: async () => {
        apiLogout();
        set({ user: null, isAuthenticated: false });
      },

      fetchMe: async () => {
        try {
          const res = await apiClient.get('/auth/me');
          set({ user: res.data as User, isAuthenticated: true });
        } catch {
          set({ user: null, isAuthenticated: false });
        }
      },
    }),
    {
      name: 'deepguard-auth',
      partialize: (state) => ({
        user: state.user,
        isAuthenticated: state.isAuthenticated,
      }),
    },
  ),
);

/** Initialise auth state from localStorage on app load */
export function initAuth(): boolean {
  return isLoggedIn();
}
