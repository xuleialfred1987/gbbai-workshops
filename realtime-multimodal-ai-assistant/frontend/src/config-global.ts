import { paths } from 'src/routes/paths';

const readEnv = (value: string | undefined, fallback = ''): string => value || fallback;

export const REALTIME_API = readEnv(
  import.meta.env.VITE_REALTIME_API,
  'http://127.0.0.1:8766/realtime'
);

// ROOT PATH AFTER LOGIN SUCCESSFUL
export const DEFAULT_PATH = paths.singleApp.app('realtime-assistant');