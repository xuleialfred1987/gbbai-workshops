import hljs from 'highlight.js/lib/core';
// ---- selective language imports (keeps bundle small) ---------------
import css from 'highlight.js/lib/languages/css';
import xml from 'highlight.js/lib/languages/xml'; // covers html / xml
import json from 'highlight.js/lib/languages/json';
import python from 'highlight.js/lib/languages/python';
import 'highlight.js/styles/base16/tomorrow-night.css';
import markdown from 'highlight.js/lib/languages/markdown';
import javascript from 'highlight.js/lib/languages/javascript';
import typescript from 'highlight.js/lib/languages/typescript';

// ----------------------------------------------------------------------

// declare global {
//   interface Window {
//     HighlightJS: any;
//   }
// }

// HighlightJS.configure({
//   languages: ['javascript', 'typescript', 'markdown', 'python', 'html', 'css', 'json'],
// });

// if (typeof window !== 'undefined') {
//   window.HighlightJS = HighlightJS;
// }

// Prevent duplicate registration during hot‑reloads
type HLJSGlobal = typeof globalThis & { __HLJS__?: typeof hljs };
const g = globalThis as HLJSGlobal;

if (!g.__HLJS__) {
  // register languages once
  hljs.registerLanguage('javascript', javascript);
  hljs.registerLanguage('js', javascript);

  hljs.registerLanguage('typescript', typescript);
  hljs.registerLanguage('ts', typescript);

  hljs.registerLanguage('json', json);

  hljs.registerLanguage('markdown', markdown);
  hljs.registerLanguage('md', markdown);

  hljs.registerLanguage('python', python);
  hljs.registerLanguage('py', python);

  hljs.registerLanguage('html', xml);
  hljs.registerLanguage('xml', xml);

  hljs.registerLanguage('css', css);

  hljs.configure({ ignoreUnescapedHTML: true });

  // Optional: expose for debugging
  if (typeof window !== 'undefined') {
    (window as any).hljs = hljs;
  }

  g.__HLJS__ = hljs; // cache instance
}

export default g.__HLJS__!;
