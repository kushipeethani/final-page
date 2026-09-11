import axios from 'axios';

const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

export const api = axios.create({
  baseURL: API_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem('token');
  const userStr = localStorage.getItem('user');

  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }

  if (userStr) {
    try {
      const user = JSON.parse(userStr);
      if (user?.id) {
        config.headers['X-User-Id'] = String(user.id);
      }
    } catch {
      // ignore parse error
    }
  }

  return config;
});

export const getApiErrorMessage = (error: unknown, fallback: string) => {
  if (axios.isAxiosError(error)) {
    const detail = error.response?.data?.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail) && detail.length > 0) {
      return detail.map((d: any) => d.msg || JSON.stringify(d)).join(', ');
    }
    const message = error.response?.data?.message;
    if (typeof message === 'string') return message;

    if (error.code === 'ERR_NETWORK' || !error.response) {
      return `Unable to connect to backend server (${API_URL}). Please ensure the backend is running.`;
    }
  } else if (error instanceof Error) {
    return error.message;
  }
  return fallback;
};

