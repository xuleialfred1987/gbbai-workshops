export type ToolCallState = {
  funcName: string;
  callId?: string;
  previousItemId?: string;
  status: 'running' | 'completed' | 'error';
  results?: any;
};

export const BACKEND_HEALTH_CHECK_INTERVAL_MS = 30000;

export const parseToolPayload = (payload: any) => {
  if (typeof payload === 'string') {
    try {
      return JSON.parse(payload);
    } catch {
      return payload;
    }
  }

  return payload;
};