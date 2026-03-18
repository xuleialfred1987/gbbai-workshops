import { DialogProps } from '@mui/material/Dialog';

// ----------------------------------------------------------------------

export type ConfirmPopupProps = Omit<DialogProps, 'title' | 'content'> & {
  title: React.ReactNode;
  content?: React.ReactNode;
  action: React.ReactNode;
  onClose: VoidFunction;
};
