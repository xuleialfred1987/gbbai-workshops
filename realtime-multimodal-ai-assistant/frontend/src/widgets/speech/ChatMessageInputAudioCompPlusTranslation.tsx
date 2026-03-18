import { useSnackbar } from 'notistack';
import micFill from '@iconify/icons-ri/mic-fill';
import { useRef, useState, useEffect } from 'react';
import {
  AudioConfig,
  OutputFormat,
  ResultReason,
  TranslationRecognizer,
  SpeechTranslationConfig,
} from 'microsoft-cognitiveservices-speech-sdk';

import Fab from '@mui/material/Fab';
import Stack from '@mui/material/Stack';
import Button from '@mui/material/Button';
import MenuItem from '@mui/material/MenuItem';
import CircularProgress from '@mui/material/CircularProgress';

// import { azureAISpeechKey, azureAISpeechRegion } from 'src/config-global';
import { getStorage, AZURE_SPEECH_KEY } from 'src/hooks/local-storage';

import Iconify from 'src/widgets/iconify';
import StyledPopover, { usePopover } from 'src/widgets/styled-popover';

import { SOURCE_LANG_COUNTRIES } from './components/langs';

// ----------------------------------------------------------------------

type Props = {
  chatMode?: string;
  onSetInputText?: React.Dispatch<React.SetStateAction<string>>;
};

export default function ChatMessageInputAudioComp({ chatMode, onSetInputText }: Props) {
  const sourceLangPopover = usePopover();
  const { enqueueSnackbar } = useSnackbar();
  const resource = getStorage(AZURE_SPEECH_KEY);

  const azureAISpeechKey = resource?.speechKey || '';
  const azureAISpeechRegion = resource?.speechRegion || '';

  const [isRecording, setIsRecording] = useState(false);
  const [isPreparing, setIsPreparing] = useState<boolean>(false);
  const [speechLang, setSpeechLang] = useState<string>('en-us');
  const [translateLang] = useState<string>('en');
  const [displaySpeech, setDisplaySpeech] = useState<string>('');
  const [displaySpeechChunks, setDisplaySpeechChunks] = useState<string[]>([]);
  // const [displayTranslation, setDisplayTranslation] = useState<string>('');
  // const [displayTranslationChunks, setDisplayTranslationChunks] = useState<string[]>([]);

  const recognizerRef = useRef<TranslationRecognizer | null>(null);
  const partialTextLengthRef = useRef<number>(0);
  const partialTranslationLengthRef = useRef<number>(0);

  const sourceLangName = SOURCE_LANG_COUNTRIES.find(
    (_country) => _country.value === speechLang
  )?.displayName;

  useEffect(() => {
    if (!azureAISpeechKey || !azureAISpeechRegion) {
      return;
    }

    const speechConfig = SpeechTranslationConfig.fromSubscription(
      azureAISpeechKey,
      azureAISpeechRegion
    );
    speechConfig.speechRecognitionLanguage = speechLang;
    speechConfig.addTargetLanguage(translateLang);
    speechConfig.outputFormat = OutputFormat.Simple;

    const audioConfig = AudioConfig.fromDefaultMicrophoneInput();

    // Initialize recognizer
    const recognizer = new TranslationRecognizer(speechConfig, audioConfig);
    recognizerRef.current = recognizer;

    // Event handlers
    recognizer.recognizing = (s, e) => {
      setDisplaySpeech((prev) => {
        const prevTextLength = partialTextLengthRef.current;
        const updatedDisplayText = prev.slice(0, -prevTextLength) + e.result.text;
        return updatedDisplayText;
      });
      // const translation = e.result.translations.get(translateLang) || '';
      // setDisplayTranslation((prev) => {
      //   const prevTextLength = partialTranslationLengthRef.current;
      //   const updatedDisplayText = prev.slice(0, -prevTextLength) + translation;
      //   return updatedDisplayText;
      // });
    };

    recognizer.recognized = async (s, e) => {
      if (
        e.result.reason === ResultReason.RecognizedSpeech ||
        e.result.reason === ResultReason.TranslatedSpeech
      ) {
        setDisplaySpeech('');
        partialTextLengthRef.current = 0;
        setDisplaySpeechChunks((prev) => [...prev, e.result.text]);
      }
    };

    recognizer.canceled = (s, e) => {
      handleStopRecognition();
    };

    recognizer.sessionStopped = (s, e) => {
      handleStopRecognition();
    };
  }, [speechLang, translateLang, azureAISpeechKey, azureAISpeechRegion]);

  useEffect(() => {
    onSetInputText?.(displaySpeechChunks.join('\n') + displaySpeech);
  }, [displaySpeechChunks, displaySpeech, onSetInputText]);

  const handleStartRecognition = async () => {
    if (!azureAISpeechKey || !azureAISpeechRegion) {
      enqueueSnackbar(
        <>
          Speech key missing.
          <br />
          Please configure it in User/Account.
        </>,
        {
          variant: 'info',
        }
      );
      return;
    }

    setDisplaySpeech('');
    setDisplaySpeechChunks([]);
    setIsRecording(!isRecording);
    partialTextLengthRef.current = 0;
    // setDisplayTranslation('');
    // setDisplayTranslationChunks([]);
    partialTranslationLengthRef.current = 0;

    setIsPreparing(true);

    recognizerRef.current?.startContinuousRecognitionAsync(
      () => {
        setIsPreparing(false);
      },
      (err) => console.error('Recognizer error:', err)
    );
  };

  const handleStopRecognition = () => {
    recognizerRef.current?.stopContinuousRecognitionAsync(
      () => {},
      (err) => console.error('Error stopping recognizer:', err)
    );
    setIsPreparing(false);
  };

  const handleUpdateSourceLang = (newValue: string) => {
    sourceLangPopover.onClose();
    setSpeechLang(newValue);
  };

  // const handleUpdateTargetLang = (newValue: string) => {
  //   targetLangPopover.onClose();
  //   setTranslateLang(newValue);
  // };

  const onToggleListening = async () => {
    if (isRecording) {
      handleStopRecognition();
    } else {
      handleStartRecognition();
    }
  };

  return (
    <Stack direction="row" spacing={1} alignItems="center" sx={{ mx: 0.5 }}>
      <Button
        size="small"
        variant="soft"
        onClick={sourceLangPopover.onOpen}
        sx={{ width: '34px', maxWidth: '34px', minWidth: '34px', px: 0, color: 'text.secondary' }}
      >
        {sourceLangName}
      </Button>
      {isPreparing && (
        <CircularProgress
          color="primary"
          sx={{ ml: 1, mr: 0.5, minWidth: 32, maxWidth: 32, minHeight: 32, maxHeight: 32 }}
        />
      )}
      {!isPreparing && isRecording && (
        <Fab
          size="small"
          aria-label="microphone"
          onClick={onToggleListening}
          sx={{
            mx: 0.5,
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
          }}
        >
          <Iconify icon="fa6-solid:circle-stop" width={18} />
        </Fab>
      )}
      {!isPreparing && !isRecording && (
        <Fab
          size="small"
          aria-label="microphone"
          onClick={onToggleListening}
          sx={{
            mx: 0.5,
            minWidth: 32,
            maxWidth: 32,
            minHeight: 32,
            maxHeight: 32,
            boxShadow: 'None',
            background: 'linear-gradient(135deg, #0693E4, #C6A166)',
            '&:hover': {
              background: 'linear-gradient(135deg, #C6A166, #0693E4)',
            },
          }}
        >
          <Iconify icon={micFill} width={18} />
        </Fab>
      )}
      <StyledPopover
        arrow="bottom-center"
        open={sourceLangPopover.open}
        onClose={sourceLangPopover.onClose}
        sx={{ width: 148 }}
      >
        {SOURCE_LANG_COUNTRIES.map((_country) => (
          <MenuItem
            key={_country.name}
            selected={_country.name === speechLang}
            onClick={() => handleUpdateSourceLang(_country.value)}
          >
            {_country.name}
          </MenuItem>
        ))}
      </StyledPopover>
    </Stack>
  );
}
