import Fab from '@mui/material/Fab';
import Stack from '@mui/material/Stack';
import Tooltip from '@mui/material/Tooltip';
import { Theme, SxProps } from '@mui/material/styles';

import { useBoolean } from 'src/hooks/boolean';

import Iconify from 'src/widgets/iconify';

import { Message } from 'src/types/chat';

import ChatMessageAvatarComp from './ChatMessageAvatarComp';

// ----------------------------------------------------------------------

type Props = {
  avatarMsg: Message | undefined;
  status: 'idle' | 'running';
  onSetInputText?: (text: string) => void;
  recognizedCallback?: (text: string) => void;
  setSpeakingMode?: (isSpeaking: boolean) => void;
  onSetUserSpeaking?: (flag: boolean) => void;
  sx?: SxProps<Theme>;
};

export default function ChatMessageInputAvatarComp({
  avatarMsg,
  status,
  onSetInputText,
  recognizedCallback,
  setSpeakingMode,
  onSetUserSpeaking,
  sx,
}: Props) {
  const openAvatar = useBoolean();

  const speechText = avatarMsg && avatarMsg.senderId !== 'user' ? avatarMsg.body : '';
  
  return (
    <Stack direction="row" alignItems="center" sx={{ position: 'relative' }}>
      {openAvatar.value && (
        <ChatMessageAvatarComp
          speechText={speechText}
          status={status}
          isOpen={openAvatar.value}
          onCloseAvatar={openAvatar.onFalse}
          onSetUserSpeaking={onSetUserSpeaking}
          recognizedCallback={recognizedCallback}
          onSetInputText={onSetInputText}
        />
      )}

      {openAvatar.value && (
        <Fab
          size="small"
          aria-label="microphone"
          onClick={openAvatar.onFalse}
          sx={{
            minWidth: 32,
            maxWidth: 32,
            minHeight: 32,
            maxHeight: 32,
            boxShadow: 'None',
            background: '#BA000C',
            '&:hover': {
              cursor: 'pointer',
              background: '#BA000C',
              boxShadow: '0px 4px 20px rgba(186, 0, 12, 0.2)',
            },
            ...sx,
          }}
        >
          <Iconify icon="fa6-solid:circle-stop" width={18} />
        </Fab>
      )}

      {!openAvatar.value && (
        <Tooltip title="Open Avatar">
          <Fab
            size="small"
            aria-label="avatar"
            onClick={openAvatar.onTrue}
            sx={{
              minWidth: 32,
              maxWidth: 32,
              minHeight: 32,
              maxHeight: 32,
              boxShadow: 'None',
              background: 'linear-gradient(135deg, #CF46E0 0%, #297BE7 74%)',
              '&:hover': {
                background: 'linear-gradient(135deg, #297BE7, #CF46E0)',
              },
              ...sx,
            }}
          >
            <Iconify icon="ep:avatar" width={18} />
          </Fab>
        </Tooltip>
      )}
    </Stack>
  );
}
