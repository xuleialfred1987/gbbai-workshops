import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';

// mui
import Link from '@mui/material/Link';

// project imports
import { NavigationLink } from 'src/routes/components';

import 'src/utils/highlight';

// import Image from '../img-wrap';
import StyledMarkdown from './styles';
import { MarkdownProps } from './types';

// ----------------------------------------------------------------------

/**
 * Custom renderer for markdown images
 */
const ImageRenderer = (props: any) => {
  // const { display, ...otherProps } = props;
  // const { display } = props;
  // console.log('ImageRenderer', props);

  // const isHtmlImage = display === 'inline-block';
  // if (!isHtmlImage) {
  //   const { src, alt } = otherProps;
  //   const sanitizedSrc = src?.replace(/amp;/g, '') || '';

  //   return <Image alt={alt} ratio="16/9" src={sanitizedSrc} sx={{ borderRadius: 1, my: 1 }} />;
  // }

  // For HTML img tags, render a standard img element
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

/**
 * Markdown component with custom renderers
 */
const MarkdownRenderer = ({ sx, ...props }: MarkdownProps) => (
  <StyledMarkdown sx={sx}>
    <ReactMarkdown
      components={{
        img: ImageRenderer,
        a: LinkRenderer,
      }}
      rehypePlugins={[rehypeRaw, rehypeHighlight, [remarkGfm, { singleTilde: false }]]}
      {...props}
    />
  </StyledMarkdown>
);

export default MarkdownRenderer;
