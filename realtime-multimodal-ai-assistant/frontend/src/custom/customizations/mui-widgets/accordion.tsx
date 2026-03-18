import { Theme } from '@mui/material/styles';
import { accordionClasses } from '@mui/material/Accordion';
import { typographyClasses } from '@mui/material/Typography';
import { accordionSummaryClasses } from '@mui/material/AccordionSummary';

// ----------------------------------------------------------------------

export function accordion(theme: Theme) {
  const createStyles = (component: string, styles: object) => ({
    [component]: { styleOverrides: styles },
  });

  const accordionStyles = createStyles('MuiAccordion', {
    root: {
      background: 'transparent',
      [`&.${accordionClasses.expanded}`]: {
        borderRadius: theme.shape.borderRadius,
        boxShadow: theme.customShadows.z2,
        background: theme.palette.background.paper,
      },
      [`&.${accordionClasses.disabled}`]: {
        background: 'transparent',
      },
    },
  });

  const summaryStyles = createStyles('MuiAccordionSummary', {
    root: {
      paddingRight: theme.spacing(1),
      paddingLeft: theme.spacing(1),
      [`&.${accordionClasses.expanded}`]: {
        paddingRight: theme.spacing(0),
        paddingLeft: theme.spacing(2),
      },
      [`&.${accordionSummaryClasses.disabled}`]: {
        opacity: 1,
        color: theme.palette.action.disabled,
        [`& .${typographyClasses.root}`]: {
          color: 'inherit',
        },
      },
    },
    expandIconWrapper: {
      color: 'inherit',
      marginLeft: theme.spacing(2),
      [`&.${accordionClasses.expanded}`]: {
        marginRight: theme.spacing(1),
        marginLeft: theme.spacing(2),
      },
    },
  });

  return {
    ...accordionStyles,
    ...summaryStyles,
  };
}
