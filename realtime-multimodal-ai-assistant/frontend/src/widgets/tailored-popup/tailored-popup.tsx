import React from 'react';

// mui
import {
  Button,
  Dialog,
  DialogProps,
  DialogTitle,
  DialogContent,
  DialogActions,
} from '@mui/material';

// ----------------------------------------------------------------------

export interface PopupProps extends Omit<DialogProps, 'title' | 'content'> {
  title: React.ReactNode;
  content?: React.ReactNode;
  action: React.ReactNode;
  onClose: (event?: object, reason?: 'backdropClick' | 'escapeKeyDown') => void;
  cancelText?: string;
  strictModal?: boolean;
}

export const ConfirmPopup: React.FC<PopupProps> = ({
  title,
  content,
  action,
  open,
  onClose,
  cancelText,
  strictModal,
  ...rest
}) => {
  const handleDialogClose = (event: object, reason: 'backdropClick' | 'escapeKeyDown') => {
    if (strictModal && (reason === 'backdropClick' || reason === 'escapeKeyDown')) {
      // If strict modal, do not close on these reasons
      return;
    }
    // Otherwise, call the provided onClose handler
    if (onClose) {
      onClose(event, reason);
    }
  };

  return (
    <Dialog
      open={open}
      onClose={handleDialogClose} // Use the wrapper
      disableEscapeKeyDown={!!strictModal} // Disable escape if strict modal
      fullWidth
      maxWidth="xs"
      {...rest}
    >
      <DialogTitle sx={{ paddingBottom: 2 }}>{title}</DialogTitle>
      {content && <DialogContent sx={{ typography: 'body2' }}>{content}</DialogContent>}
      <DialogActions>
        {action}
        <Button variant="outlined" color="inherit" onClick={() => onClose()}>
          {cancelText || 'Cancel'}
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default ConfirmPopup;
