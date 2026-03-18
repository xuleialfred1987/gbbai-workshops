/// <reference types="vite/client" />

interface ImportMetaEnv {
	readonly VITE_REALTIME_API?: string;
}

interface ImportMeta {
	readonly env: ImportMetaEnv;
}
