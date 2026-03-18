import type { Theme } from '@mui/material/styles';

// ----------------------------------------------------------------------

export function breadcrumbs(theme: Theme) {
  const spacingValues = {
    separator: {
      x: theme.spacing(2),
    },
    listItem: {
      y: theme.spacing(0.25),
    },
  };

  return {
    MuiBreadcrumbs: {
      styleOverrides: {
        separator: {
          marginLeft: spacingValues.separator.x,
          marginRight: spacingValues.separator.x,
        },
        li: {
          display: 'inline-flex',
          margin: `${spacingValues.listItem.y} 0`,
          '& > *': {
            fontFamily: theme.typography.body2.fontFamily,
            fontSize: theme.typography.body2.fontSize,
            fontWeight: theme.typography.body2.fontWeight,
            lineHeight: theme.typography.body2.lineHeight,
            letterSpacing: theme.typography.body2.letterSpacing,
          },
        },
      },
    },
  };
}
