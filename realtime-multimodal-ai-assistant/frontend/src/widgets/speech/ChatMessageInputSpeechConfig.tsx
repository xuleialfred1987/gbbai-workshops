import { useSnackbar } from 'notistack';

// mui
import Button from '@mui/material/Button';
import { Theme, SxProps } from '@mui/material/styles';

// project imports
import { useBoolean } from 'src/hooks/boolean';

import VoiceConfigDialog from './components/voice-config-dialog';
import { useSpeechConfig } from './context/speech-config-context';
import { SpeechOption, SOURCE_LANG_COUNTRIES } from './components/langs';

// ----------------------------------------------------------------------

interface ChatMessageInputSpeechConfigProps {
  sx?: SxProps<Theme>;
}

export default function ChatMessageInputSpeechConfig({
  sx,
}: ChatMessageInputSpeechConfigProps = {}) {
  const openConfig = useBoolean();
  const { enqueueSnackbar } = useSnackbar();

  const { config, setConfig } = useSpeechConfig();

  const speechLang = config?.lang || 'en-us';

  const sourceLangName = SOURCE_LANG_COUNTRIES.find(
    (_country) => _country.value === speechLang
  )?.displayName;

  const handleSetSpeechOption = (newConfig: SpeechOption) => {
    try {
      setConfig(newConfig);
      enqueueSnackbar('Speech configuration set successfully');
    } catch (error) {
      console.error(error);
      enqueueSnackbar(`Failed to set speech configuration`, { variant: 'error' });
    }
  };

  return (
    <>
      <Button
        size="small"
        variant="soft"
        onClick={openConfig.onTrue}
        sx={{
          width: '34px',
          maxWidth: '34px',
          minWidth: '34px',
          height: '28px',
          minHeight: '28px',
          maxHeight: '28px',
          justifyContent: 'center',
          alignItems: 'center',
          lineHeight: 1,
          color: 'text.secondary',
          ...(sx && sx),
        }}
      >
        {sourceLangName}
      </Button>

      <VoiceConfigDialog
        open={openConfig.value}
        onClose={openConfig.onFalse}
        storedConfig={config}
        onSetSpeechOption={handleSetSpeechOption}
      />
    </>
  );
}
