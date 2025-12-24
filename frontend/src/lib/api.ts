/**
 * AudioGhost AI - API Configuration
 * 
 * API URL is configurable via environment variable for Docker deployment.
 * In development: defaults to http://localhost:8000
 * In Docker: set NEXT_PUBLIC_API_URL to http://api:8000 (or use browser-accessible URL)
 */

// For client-side (browser) requests, we need a URL accessible from the browser
// For server-side requests, we could use internal Docker network
// Since all our API calls are from the browser (client components), we use the public URL
export const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

/**
 * API Endpoints
 */
export const API = {
    auth: {
        status: () => `${API_URL}/api/auth/status`,
        login: () => `${API_URL}/api/auth/login`,
        logout: () => `${API_URL}/api/auth/logout`,
    },
    separate: {
        create: () => `${API_URL}/api/separate/`,
        batch: () => `${API_URL}/api/separate/batch`,
    },
    tasks: {
        status: (taskId: string) => `${API_URL}/api/tasks/${taskId}`,
        download: (taskId: string, fileType: string) => `${API_URL}/api/tasks/${taskId}/download/${fileType}`,
        cancel: (taskId: string) => `${API_URL}/api/tasks/${taskId}`,
        list: () => `${API_URL}/api/tasks/`,
    },
} as const;

