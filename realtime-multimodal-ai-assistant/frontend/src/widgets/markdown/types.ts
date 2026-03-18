import { Options as ReactMarkdownOptions } from 'react-markdown';

// Material UI imports
import type { Theme, SxProps } from '@mui/material/styles';

// ----------------------------------------------------------------------

/**
 * Configuration interface for Markdown component
 * Extends the base ReactMarkdown options and adds styling capabilities
 */
export type MarkdownProps = ReactMarkdownOptions & {
  /**
   * Material UI system styles object for component styling
   */
  sx?: SxProps<Theme>;
};
