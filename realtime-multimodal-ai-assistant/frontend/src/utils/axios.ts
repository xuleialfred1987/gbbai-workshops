import axios from 'axios';
import type { AxiosRequestConfig } from 'axios';

/**
 * API-specific HTTP instance
 */
const apiHttp = (() => {
  const instance = axios.create();

  instance.interceptors.response.use(
    (response) => response,
    (error) => {
      const errorData = error.response?.data || 'An unexpected error occurred';
      return Promise.reject(errorData);
    }
  );

  return instance;
})();

/**
 * Data fetching utilities
 */
const httpUtils = {
  /**
   * Standard data fetcher
   */
  async fetch(args: string | [string, AxiosRequestConfig]) {
    const [url, config] = Array.isArray(args) ? args : [args, {}];
    const response = await apiHttp.get(url, config);
    return response.data;
  },

  /**
   * Binary data fetcher with blob processing
   */
  async fetchBlob(url: string): Promise<string> {
    const response = await apiHttp.get(url, { responseType: 'blob' });

    if (response.status !== 200) {
      throw new Error('Failed to fetch binary data');
    }

    const blob = new Blob([response.data], { type: 'image/jpeg' });

    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onloadend = () => resolve(reader.result as string);
      reader.onerror = reject;
      reader.readAsDataURL(blob);
    });
  },

  /**
   * Fetch with cache refresh key
   */
  async fetchWithRefreshKey(args: [string, number] | [string, number, AxiosRequestConfig]) {
    const [url, refreshKey, config = {}] = args;
    const response = await apiHttp.get(`${url}?refreshKey=${refreshKey}`, config);
    return response.data;
  },

  async fetchWithRefreshKey2(url: string, refreshKey: number, config: AxiosRequestConfig = {}) {
    // Make sure this signature takes url and refreshKey separately
    const response = await apiHttp.get(`${url}?refreshKey=${refreshKey}`, config);
    return response.data;
  },
};

/**
 * API endpoint definitions
 */
const apiEndpoints = {
  app: {
    list: '/apps',
    create: '/apps/create',
    delete: '/apps/delete',
    customGpt: {
      list: '/apps/custom-gpt',
      create: '/apps/custom-gpt/create',
      delete: '/apps/custom-gpt/delete',
      sas: '/apps/custom-gpt/sas',
    },
  },
  agentFleet: {
    list: '/agent-fleet/agents',
    execute: '/agent-fleet/execute',
    agents: {
      list: '/agent-fleet/agents',
      create: '/agent-fleet/agents/create',
      delete: '/agent-fleet/agents/delete',
    },
    conversations: {
      list: '/agent-fleet/conversations',
      create: '/agent-fleet/conversations/create',
      delete: '/agent-fleet/conversations/delete',
      update: '/agent-fleet/conversations/update',
    },
  },
  chat: '/api/chat',
  embedding: { text: '/embedding/text' },
  image: {
    list: (userId: string) => `/image/list/${encodeURIComponent(userId)}`,
    generate: '/image/generate-image',
    edit: '/image/edit-image',
  },
  kmm: {
    root: '/kmm',
    list: '/kmm/list',
    kb: '/kmm/kb',
    create: '/kmm/kb/create',
    sources: '/kmm/sources',
  },
  tool: {
    list: '/tools',
    create: '/tools/create',
    delete: '/tools/delete',
    deploy: '/tools/deploy',
    sas: '/tools/sas',
  },
  documentation: {
    root: '/documentation',
    faqs: '/documentation/faqs',
    contents: '/documentation/contents',
  },
  video: {
    list: '/video/list',
    details: '/video/details',
    latest: '/video/latest',
    search: '/video/search',
  },
  document_analyzer: {
    projects: {
      list: '/doc_analyzer/projects',
      create: '/doc_analyzer/projects/create',
      delete: '/doc_analyzer/projects/delete',
    },
    files: {
      get: (id: string) => `/doc_analyzer/files/${encodeURIComponent(id)}`,
    },
  },
  voice_ai: {
    projects: {
      list: '/voice_ai/projects',
      create: '/voice_ai/projects/create',
      delete: '/voice_ai/projects/delete',
    },
    audios: {
      list: '/voice_ai/audios',
      create: '/voice_ai/audios/create',
      delete: '/voice_ai/audios/delete',
    },
  },
  workshops: {
    root: '/ips',
    list: '/ips/list',
    details: '/ips',
    search: '/ips/search',
  },
  research: {
    root: '/api/research',
    list: '/api/research/list',
    create: '/api/research/create',
    job: '/api/research/job',
  },
  keyframes: {
    root: '/api/keyframes',
    assets: {
      list: '/api/keyframes/list',
      create: '/api/keyframes/create',
      details: (id: string) => `/api/keyframes/${encodeURIComponent(id)}`,
      update: (id: string) => `/api/keyframes/${encodeURIComponent(id)}`,
      delete: (id: string) => `/api/keyframes/${encodeURIComponent(id)}`,
      search: '/api/keyframes/search',
      byProject: (projectId: string) => `/api/keyframes/project/${encodeURIComponent(projectId)}`,
      generateImage: '/api/keyframes/generate-image',
      favorite: (id: string) => `/api/keyframes/${encodeURIComponent(id)}/favorite`,
    },
  },
  scripts: {
    root: '/api/scripts',
    list: '/api/scripts',
    create: '/api/scripts',
    details: (id: string) => `/api/scripts/${encodeURIComponent(id)}`,
    update: (id: string) => `/api/scripts/${encodeURIComponent(id)}`,
    delete: (id: string) => `/api/scripts/${encodeURIComponent(id)}`,
    search: '/api/scripts/search',
    storyboards: {
      add: (scriptId: string) => `/api/scripts/${encodeURIComponent(scriptId)}/storyboards`,
      update: (scriptId: string, storyboardId: string) =>
        `/api/scripts/${encodeURIComponent(scriptId)}/storyboards/${encodeURIComponent(storyboardId)}`,
      delete: (scriptId: string, storyboardId: string) =>
        `/api/scripts/${encodeURIComponent(scriptId)}/storyboards/${encodeURIComponent(storyboardId)}`,
    },
  },
  modeling: {
    root: '/api/modelings',
    assets: {
      list: '/api/modelings/list',
      create: '/api/modelings/create',
      details: (id: string) => `/api/modelings/${encodeURIComponent(id)}`,
      update: (id: string) => `/api/modelings/${encodeURIComponent(id)}`,
      delete: (id: string) => `/api/modelings/${encodeURIComponent(id)}`,
      search: '/api/modelings/search',
      byProject: (projectId: string) => `/api/modelings/project/${encodeURIComponent(projectId)}`,
      upload: '/api/modelings/upload',
      sasUrl: (id: string) => `/api/modelings/${encodeURIComponent(id)}/sas-url`,
      generateImage: '/api/modelings/generate-image',
      favorite: (id: string) => `/api/modelings/${encodeURIComponent(id)}/favorite`,
    },
  },
  pcbCopilot: {
    root: '/api/pcb-copilot',
    health: '/api/pcb-copilot/health',
    check: '/api/pcb-copilot/check',
    jobs: '/api/pcb-copilot/jobs',
    rules: '/api/pcb-copilot/rules',
    analyze: '/api/pcb-copilot/analyze',
  },
};

// Export everything
export default apiHttp;
export const fetcher = httpUtils.fetch;
export const fetcherBlob = httpUtils.fetchBlob;
export const fetcherWithRefreshKey = httpUtils.fetchWithRefreshKey;
export const fetcherWithRefreshKey2 = httpUtils.fetchWithRefreshKey2;
export const endpoints = apiEndpoints;
