import { resolve } from 'path';
import { defineConfig } from 'vite';
import reactSwc from '@vitejs/plugin-react-swc';
import typeChecker from 'vite-plugin-checker';

const PORT = 8080;
const PATHS = {
  nodeModules: resolve(process.cwd(), 'node_modules'),
  src: resolve(process.cwd(), 'src'),
};

const aliasConfig = [
  { find: /^~(.+)/, replacement: `${PATHS.nodeModules}/$1` },
  { find: /^src(.+)/, replacement: `${PATHS.src}/$1` },
];

const devTools = [
  reactSwc(),
  typeChecker({
    typescript: true,
    eslint: {
      lintCommand: 'eslint "./src/**/*.{js,jsx,ts,tsx}"',
    },
  }),
];

export default defineConfig({
  plugins: devTools,
  resolve: {
    alias: aliasConfig,
  },
  server: {
    host: true,
    port: PORT,
  },
  preview: {
    host: true,
    port: PORT,
  },
});
