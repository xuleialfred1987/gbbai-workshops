import remarkGfm from 'remark-gfm';
import rehypeRaw from 'rehype-raw';
import ReactMarkdown from 'react-markdown';
import rehypeHighlight from 'rehype-highlight';

// mui
import Link from '@mui/material/Link';

// project imports
import { NavigationLink } from 'src/routes/components';

import 'src/utils/highlight';

import Image from '../img-wrap';
import StyledMarkdown from './styles';
import { MarkdownProps } from './types';

// ----------------------------------------------------------------------

/**
 * Custom renderer for markdown images
 */
const ImageRenderer = (props: any) => {
  const { src, alt } = props;
  const sanitizedSrc = src?.replace(/amp;/g, '') || '';

  return (
    <Image
      alt={alt}
      ratio="1/1"
      src={sanitizedSrc}
      sx={{
        borderRadius: 1,
        my: 1,
      }}
      {...props}
    />
  );
};

/**
 * Custom renderer for markdown links
 */
const LinkRenderer = (props: any) => {
  const { href, children, ...rest } = props;
  const isExternal = href?.includes('http');

  if (isExternal) {
    return <Link target="_blank" rel="noopener" href={href} {...rest} />;
  }

  return (
    <Link component={NavigationLink} path={href} {...rest}>
      {children}
    </Link>
  );
};

/**
 * Markdown component with custom renderers
 */
const MarkdownImgRenderer = ({ sx, ...props }: MarkdownProps) => (
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

export default MarkdownImgRenderer;
