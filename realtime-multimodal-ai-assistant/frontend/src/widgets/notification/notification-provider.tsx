import { useRef } from 'react';
import type { VariantType, SnackbarOrigin } from 'notistack';
import { closeSnackbar, SnackbarProvider as NotistackProvider } from 'notistack';

// mui
import { Collapse, IconButton } from '@mui/material';

// project import
import Iconify from '../iconify';
import { StyledIcon, StyledNotistack } from './styles';

// ----------------------------------------------------------------------

interface SnackbarProviderProps {
  children: React.ReactNode;
}

const iconConfig = {
  info: { icon: 'eva:info-fill', color: 'info' },
  success: { icon: 'eva:checkmark-circle-2-fill', color: 'success' },
  warning: { icon: 'eva:alert-triangle-fill', color: 'warning' },
  error: { icon: 'solar:danger-bold', color: 'error' },
} as const;

const anchorOrigin: SnackbarOrigin = {
  vertical: 'top',
  horizontal: 'right',
};

export default function NotificationProvider({ children }: SnackbarProviderProps) {
  const notistackRef = useRef<NotistackProvider>(null);

  const generateIconVariant = () => {
    const variants = {} as Record<VariantType, JSX.Element>;

    Object.entries(iconConfig).forEach(([key, value]) => {
      variants[key as VariantType] = (
        <StyledIcon color={value.color}>
          <Iconify icon={value.icon} width={24} />
        </StyledIcon>
      );
    });

    return variants;
  };

  const handleClose = (snackbarId: string | number) => () => {
    closeSnackbar(snackbarId);
  };

  const action = (snackbarId: string | number) => (
    <IconButton size="small" onClick={handleClose(snackbarId)} sx={{ p: 0.5 }}>
      <Iconify width={16} icon="mingcute:close-line" />
    </IconButton>
  );

  return (
    <NotistackProvider
      ref={notistackRef}
      maxSnack={3}
      preventDuplicate
      autoHideDuration={4000}
      TransitionComponent={Collapse}
      variant="success"
      anchorOrigin={anchorOrigin}
      iconVariant={generateIconVariant()}
      Components={{
        default: StyledNotistack,
        info: StyledNotistack,
        success: StyledNotistack,
        warning: StyledNotistack,
        error: StyledNotistack,
      }}
      action={action}
    >
      {children}
    </NotistackProvider>
  );
}
