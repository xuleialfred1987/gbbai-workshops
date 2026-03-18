import { Theme, styled } from '@mui/material/styles';

// ----------------------------------------------------------------------

const createMarkdownStyles = (theme: Theme) => {
  const isLightMode = theme.palette.mode === 'light';

  // Define reusable style patterns
  const spacing = (top: number, right?: number, bottom?: number, left?: number) =>
    theme.spacing(top, right ?? top, bottom ?? top, left ?? right ?? top);

  const textStyles = {
    headings: {
      h1: { ...theme.typography.h1, margin: spacing(1, 0), lineHeight: 1.5 },
      h2: { ...theme.typography.h2, margin: spacing(1, 0), lineHeight: 1.5 },
      h3: { ...theme.typography.h3, margin: spacing(1, 0), lineHeight: 1.5 },
      h4: { ...theme.typography.h4, margin: spacing(1, 0), lineHeight: 1.5 },
      h5: { ...theme.typography.h5, margin: 0 },
      h6: { ...theme.typography.h6, margin: 0 },
    },
    paragraph: {
      ...theme.typography.body1,
      margin: spacing(1.25, 0),
      lineHeight: 1.5,
    },
    lineBreak: {
      display: 'grid',
      content: '""',
      marginTop: '0.75em',
    },
  };

  const codeBlockStyles = {
    pre: {
      fontSize: 14.5,
      overflowX: 'auto',
      whiteSpace: 'pre',
      padding: spacing(1.5),
      color: theme.palette.common.white,
      borderRadius: theme.shape.borderRadius * 0.75,
      backgroundColor: isLightMode ? theme.palette.grey[800] : theme.palette.grey[900],
    },
    inlineCode: {
      fontSize: 14,
      borderRadius: 4,
      whiteSpace: 'pre',
      margin: spacing(0, 0.25),
      padding: spacing(0.25, 0.5),
      color: theme.palette.warning[isLightMode ? 'darker' : 'lighter'],
      backgroundColor: theme.palette.warning[isLightMode ? 'lighter' : 'darker'],
      '&.hljs': {
        padding: 0,
        backgroundColor: 'transparent',
      },
    },
  };

  const blockElements = {
    divider: {
      margin: 0,
      marginTop: 12,
      flexShrink: 0,
      borderWidth: 0,
      msFlexNegative: 0,
      WebkitFlexShrink: 0,
      borderStyle: 'solid',
      borderBottomWidth: 'thin',
      borderColor: theme.palette.divider,
    },
    blockquote: {
      lineHeight: 1.5,
      fontSize: '1.5em',
      margin: '40px auto',
      position: 'relative',
      fontFamily: 'Georgia, serif',
      padding: spacing(3, 3, 3, 8),
      color: theme.palette.text.secondary,
      borderRadius: theme.shape.borderRadius * 2,
      backgroundColor: theme.palette.background.neutral,
      [theme.breakpoints.up('md')]: {
        width: '80%',
      },
      '& p, & span': {
        marginBottom: 0,
        fontSize: 'inherit',
        fontFamily: 'inherit',
      },
      '&:before': {
        left: 16,
        top: -8,
        display: 'block',
        fontSize: '3em',
        content: '"\\201C"',
        position: 'absolute',
        color: theme.palette.text.disabled,
      },
    },
  };

  const listStyles = {
    lists: {
      margin: 0,
      '& li': {
        lineHeight: 1.75,
        marginBottom: '8px',
      },
    },
  };

  const tableStyles = {
    table: {
      width: '100%',
      margin: spacing(1.5, 0),
      borderCollapse: 'collapse',
      border: `1px solid ${theme.palette.divider}`,
      'th, td': {
        padding: spacing(1),
        border: `1px solid ${theme.palette.divider}`,
      },
      'tbody tr:nth-of-type(odd)': {
        backgroundColor: theme.palette.background.neutral,
      },
    },
  };

  const formElements = {
    checkbox: {
      position: 'relative',
      cursor: 'pointer',
      '&:before': {
        content: '""',
        top: -2,
        left: -2,
        width: 17,
        height: 17,
        borderRadius: 3,
        position: 'absolute',
        backgroundColor: theme.palette.grey[isLightMode ? 300 : 700],
      },
      '&:checked': {
        '&:before': {
          backgroundColor: theme.palette.primary.main,
        },
        '&:after': {
          content: '""',
          top: 1,
          left: 5,
          width: 4,
          height: 9,
          position: 'absolute',
          transform: 'rotate(45deg)',
          msTransform: 'rotate(45deg)',
          WebkitTransform: 'rotate(45deg)',
          border: `solid ${theme.palette.common.white}`,
          borderWidth: '0 2px 2px 0',
        },
      },
    },
  };

  // Combine all styles into a single style object
  return {
    // Text elements
    h1: textStyles.headings.h1,
    h2: textStyles.headings.h2,
    h3: textStyles.headings.h3,
    h4: textStyles.headings.h4,
    h5: textStyles.headings.h5,
    h6: textStyles.headings.h6,
    p: textStyles.paragraph,
    br: textStyles.lineBreak,

    // Block elements
    hr: blockElements.divider,
    '& blockquote': blockElements.blockquote,

    // List elements
    '& ul, & ol': listStyles.lists,

    // Code elements
    '& pre, & pre > code': codeBlockStyles.pre,
    '& code': codeBlockStyles.inlineCode,

    // Table elements
    table: tableStyles.table,

    // Form elements
    input: {
      '&[type=checkbox]': formElements.checkbox,
    },
  };
};

// Create the styled component
const StyledMarkdown = styled('div')(({ theme }) => ({
  ...createMarkdownStyles(theme),
}));

export default StyledMarkdown;
