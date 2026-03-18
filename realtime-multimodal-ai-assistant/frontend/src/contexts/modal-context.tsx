import React, {
  useMemo,
  useState,
  ReactNode,
  useEffect,
  useContext,
  useCallback,
  createContext,
} from 'react';

// mui
import { Button } from '@mui/material';

// project imports
import { PopupProps, ConfirmPopup } from 'src/widgets/tailored-popup';

// ----------------------------------------------------------------------

type GlobalConfirmProps = {
  title: string;
  content: ReactNode;
  actionConfig: {
    text: string;
    onClick: () => void;
    color?: 'inherit' | 'primary' | 'secondary' | 'success' | 'error' | 'info' | 'warning';
  };
  onPopupClose?: (event?: object, reason?: 'backdropClick' | 'escapeKeyDown') => void; // Updated signature
  strictModal?: boolean;
};

type ModalContextType = {
  showGlobalConfirm: (props: GlobalConfirmProps) => void;
  hideGlobalConfirm: (event?: object, reason?: 'backdropClick' | 'escapeKeyDown') => void; // Updated signature
};

const ModalContext = createContext<ModalContextType | undefined>(undefined);

export const ModalProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [confirmDialogProps, setConfirmDialogProps] = useState<
    (Omit<PopupProps, 'open' | 'onClose'> & { strictModal?: boolean }) | null
  >(null);
  const [isConfirmOpen, setIsConfirmOpen] = useState(false);
  const [externalOnCloseHandler, setExternalOnCloseHandler] = useState<
    GlobalConfirmProps['onPopupClose'] | undefined
  >(undefined);

  const hideGlobalConfirm = useCallback(
    (event?: object, reason?: 'backdropClick' | 'escapeKeyDown') => {
      setIsConfirmOpen(false);
      if (externalOnCloseHandler) {
        externalOnCloseHandler(event, reason); // Pass along event and reason
      }
      setTimeout(() => {
        setConfirmDialogProps(null);
        setExternalOnCloseHandler(undefined);
      }, 300);
    },
    [externalOnCloseHandler]
  );

  const showGlobalConfirm = useCallback(
    ({ title, content, actionConfig, onPopupClose, strictModal = true }: GlobalConfirmProps) => {
      setConfirmDialogProps({
        title,
        content,
        action: (
          <Button
            variant="contained"
            color={actionConfig.color || 'primary'}
            onClick={() => {
              actionConfig.onClick();
              hideGlobalConfirm(); // Close popup after action
            }}
          >
            {actionConfig.text}
          </Button>
        ),
        strictModal, // Pass strictModal
      });
      setExternalOnCloseHandler(() => onPopupClose);
      setIsConfirmOpen(true);
    },
    [hideGlobalConfirm]
  );

  useEffect(() => {
    if (typeof window !== 'undefined') {
      (window as any).showGlobalConfirm = showGlobalConfirm;
      (window as any).hideGlobalConfirm = hideGlobalConfirm;
    }
    return () => {
      if (typeof window !== 'undefined') {
        delete (window as any).showGlobalConfirm;
        delete (window as any).hideGlobalConfirm;
      }
    };
  }, [showGlobalConfirm, hideGlobalConfirm]);

  const contextValue = useMemo(
    () => ({
      showGlobalConfirm,
      hideGlobalConfirm,
    }),
    [showGlobalConfirm, hideGlobalConfirm]
  );

  return (
    <ModalContext.Provider value={contextValue}>
      {children}
      {confirmDialogProps && (
        <ConfirmPopup
          open={isConfirmOpen}
          onClose={hideGlobalConfirm} // This is (event?, reason?) => void
          title={confirmDialogProps.title}
          content={confirmDialogProps.content}
          action={confirmDialogProps.action}
          strictModal={confirmDialogProps.strictModal} // Pass strictModal to ConfirmPopup
        />
      )}
    </ModalContext.Provider>
  );
};

export const useModal = () => {
  const context = useContext(ModalContext);
  if (context === undefined) {
    throw new Error('useModal must be used within a ModalProvider');
  }
  return context;
};
