import { useRef, useCallback } from 'react';

import { styled } from '@mui/material/styles';
import {
  Stack,
  Portal,
  Dialog,
  Button,
  Divider,
  CardHeader,
  DialogActions,
  ClickAwayListener,
} from '@mui/material';

import Scroller from 'src/widgets/scroller';

import { IRtConfiguration } from 'src/types/chat';

import ChatTab from './tab-chat';

// ----------------------------------------------------------------------

const RootStyle = styled('div')(({ theme }) => ({
  // position: 'fixed',
  width: '100%',
  outline: 'none',
  display: 'flex',
  flexDirection: 'row',
  margin: 0,
  boxShadow: theme.customShadows.z20,
  borderRadius: theme.shape.borderRadius,
  backgroundColor: theme.palette.background.paper,

  height: 'auto',
  maxHeight: '80vh',
  overflowY: 'auto',
  minHeight: 440,

  [theme.breakpoints.down('sm')]: {
    // maxHeight: 'none',
    minHeight: 'auto',
    maxHeight: '90vh',
  },
}));

// ----------------------------------------------------------------------

type Props = {
  open: boolean;
  callerId: string;
  onClose: VoidFunction;
  configurations: IRtConfiguration;
  onUpdate: (config: IRtConfiguration) => void;
};

export default function GptConfigEntry({
  open,
  callerId,
  onClose,
  configurations,
  onUpdate,
}: Props) {
  const formRef = useRef<HTMLFormElement>(null);

  const selectedMode = 'rt';

  const handleButtonClick = () => {
    if (formRef.current) {
      formRef.current.submit();
    }
    onClose();
  };

  const handleChangeMode = useCallback((newMode: string) => {}, []);

  return (
    <Portal>
      <ClickAwayListener onClickAway={() => {}}>
        <Dialog
          open={open}
          onClose={onClose}
          fullWidth
          maxWidth="md"
        >
          <RootStyle>
            <Stack sx={{ width: '100%' }}>
              <CardHeader title="Configuration" sx={{ mb: 1.5 }} />

              <Scroller sx={{ flexDirection: 'row' }}>
                <ChatTab
                  onClose={() => {}}
                  configs={configurations}
                  ref={formRef}
                  onUpdate={onUpdate}
                  modes={[]}
                  selectedMode={selectedMode}
                  onChangeMode={handleChangeMode}
                />
              </Scroller>

              <Divider />

              <DialogActions>
                <Button type="submit" variant="contained" onClick={handleButtonClick}>
                  Confirm
                </Button>
                <Button variant="outlined" color="inherit" onClick={onClose}>
                  Cancel
                </Button>
              </DialogActions>
            </Stack>
          </RootStyle>
        </Dialog>
      </ClickAwayListener>
    </Portal>
  );
}
