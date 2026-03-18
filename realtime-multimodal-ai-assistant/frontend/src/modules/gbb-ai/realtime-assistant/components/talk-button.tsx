import { useState, useEffect } from 'react';

import Stack from '@mui/material/Stack';
import IconButton from '@mui/material/IconButton';

import Iconify from 'src/widgets/iconify';

// ---------------------------------------------------------------------

type Props = {
  onToggleRT?: (flag: boolean) => void | Promise<void>;
  isActive?: boolean;
};

export default function ChatMessageInputTalkBtn({ onToggleRT, isActive }: Props) {
  const [isRecording, setIsRecording] = useState(false);

  useEffect(() => {
    if (typeof isActive === 'boolean') {
      setIsRecording(isActive);
    }
  }, [isActive]);

  const onStartListening = async () => {
    await onToggleRT?.(true);
    setIsRecording(true);
  };

  const onStopListening = async () => {
    await onToggleRT?.(false);
    setIsRecording(false);
  };

  return (
    <Stack direction="row" spacing={1} alignItems="center">
      {isRecording && (
        <IconButton
          size="small"
          aria-label="microphone"
          onClick={onStopListening}
          sx={{
            width: 36,
            height: 36,
            background: '#BA000C',
            color: 'white',
            '&:hover': {
              background: '#BA000C',
              boxShadow: '0px 4px 20px rgba(186, 0, 12, 0.2)',
            },
          }}
        >
          <Iconify icon="eva:close-fill" width={22} />
        </IconButton>
      )}
      {!isRecording && (
        <IconButton
          size="small"
          aria-label="microphone"
          onClick={onStartListening}
          sx={{
            width: 36,
            height: 36,
          }}
        >
          <Iconify
            icon="proicons:microphone"
            width={23}
            sx={{ color: '#0ea5e9' }}
          />
        </IconButton>
      )}
    </Stack>
  );
}
