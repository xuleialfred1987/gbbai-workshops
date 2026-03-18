/* eslint-disable react/no-danger */
import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import { lazy, Suspense } from 'react';
import ReactMarkdown from 'react-markdown';
import type { PluggableList } from 'unified';

// mui
import Box from '@mui/material/Box';
import Link from '@mui/material/Link';
import { useTheme } from '@mui/material/styles';
import CircularProgress from '@mui/material/CircularProgress';

// project imports
import { NavigationLink } from 'src/routes/components';

// import 'src/utils/highlight';
import hljs from 'src/utils/highlight';

import StyledMarkdown from './styles';
import { MarkdownProps } from './types';
// import Image from '../img-wrap';

// ----------------------------------------------------------------------

const REHYPE_PLUGINS: PluggableList = [rehypeRaw];
const REMARK_PLUGINS: PluggableList = [[remarkGfm, { singleTilde: false }]];

// Lazy load the GraphVizRenderer
const GraphVizRenderer = lazy(() => import('./graphviz-renderer'));

// ----------------------------------------------------------------------

/**
 * Wrapper component for lazy loaded GraphvizRenderer
 */
const GraphVizWrapper = ({ dot }: { dot: string }) => (
  <Suspense
    fallback={
      <Box
        sx={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: 320,
          my: 2,
        }}
      >
        <CircularProgress />
      </Box>
    }
  >
    <GraphVizRenderer dot={dot} />
  </Suspense>
);

/**
 * Custom renderer for markdown images
 */
const ImageRenderer = (props: any) => {
  const { alt = '', ...restProps } = props;
  return (
    <img
      alt={alt}
      {...restProps}
      style={{ borderRadius: 6, marginTop: 8, marginBottom: 8, ...restProps.style }}
    />
  );
};

/**
 * Custom renderer for markdown links
 */
const LinkRenderer = (props: any) => {
  const { href, children } = props;
  const isExternal = href?.includes('http') || false;

  if (isExternal) {
    return <Link target="_blank" rel="noopener" {...props} />;
  }

  return (
    <Link component={NavigationLink} path={href} {...props}>
      {children}
    </Link>
  );
};

const CodeRenderer = ({ className = '', children, ...props }: any) => {
  const theme = useTheme();
  const isLightMode = theme.palette.mode === 'light';

  const languageMatch = className.match(/language-(\w+)/);
  const isBlock = Boolean(languageMatch);
  const language = languageMatch ? languageMatch[1] : 'plaintext';
  const code = String(children).replace(/^\n+|\n+$/g, '');

  // Check if this is a GraphViz DOT language block
  if (isBlock && (language === 'dot' || language === 'graphviz')) {
    return <GraphVizWrapper dot={code} />;
  }

  /* ---------- inline <code> ---------- */
  if (!isBlock) {
    return (
      <code
        style={{
          display: 'inline',
          fontSize: 14,
          fontWeight: 500,
          padding: '2px 4px',
          borderRadius: 4,
          color: theme.palette.warning[isLightMode ? 'darker' : 'lighter'],
          backgroundColor: theme.palette.warning[isLightMode ? 'lighter' : 'darker'],
        }}
        {...props}
      >
        {code}
      </code>
    );
  }

  /* ---------- fenced / block code ---------- */
  const highlighted = hljs.getLanguage(language)
    ? hljs.highlight(code, { language }).value
    : hljs.highlightAuto(code).value;

  return (
    <pre {...props}>
      <code
        className={`hljs language-${language}`}
        dangerouslySetInnerHTML={{ __html: highlighted.trim() }}
      />
    </pre>
  );
};

/**
 * Markdown component with custom renderers
 */
const MarkdownRenderer = ({ sx, ...props }: MarkdownProps) => (
  <StyledMarkdown sx={sx}>
    <ReactMarkdown
      components={{
        img: ImageRenderer,
        a: LinkRenderer,
        code: CodeRenderer,
        // Override pre to remove background for graphviz blocks
        pre: ({ children, ...preProps }: any) => {
          // Check if this is a graphviz code block by looking at the child code element
          const codeElement = children?.props;
          const className = codeElement?.className || '';
          const languageMatch = className.match(/language-(\w+)/);
          const language = languageMatch ? languageMatch[1] : '';

          // If it's a graphviz block, return the children directly without pre wrapper
          if (language === 'dot' || language === 'graphviz') {
            return <>{children}</>;
          }

          // Otherwise, render the normal pre block
          return <pre {...preProps}>{children}</pre>;
        },
      }}
      rehypePlugins={REHYPE_PLUGINS}
      remarkPlugins={REMARK_PLUGINS}
      {...props}
    />
  </StyledMarkdown>
);

export default MarkdownRenderer;
