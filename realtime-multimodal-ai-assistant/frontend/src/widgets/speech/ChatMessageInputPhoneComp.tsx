import { useSnackbar } from 'notistack';
import { useRef, useState, useEffect } from 'react';
import {
  AudioConfig,
  OutputFormat,
  ResultReason,
  SpeechConfig,
  SpeechRecognizer,
} from 'microsoft-cognitiveservices-speech-sdk';

import Fab from '@mui/material/Fab';
import Stack from '@mui/material/Stack';
import Tooltip from '@mui/material/Tooltip';
import { Theme, SxProps } from '@mui/material/styles';
import CircularProgress from '@mui/material/CircularProgress';

import { getStorage, AZURE_SPEECH_KEY } from 'src/hooks/local-storage';

import Iconify from 'src/widgets/iconify';

import { useSpeechConfig } from './context/speech-config-context';

// ----------------------------------------------------------------------

type Props = {
  onSetInputText?: (text: string) => void;
  recognizedCallback?: (text: string) => void;
  setSpeakingMode?: (isSpeaking: boolean) => void;
  onSetUserSpeaking?: (flag: boolean) => void;
  sx?: SxProps<Theme>;
};

export default function ChatMessageInputPhoneComp({
  onSetInputText,
  recognizedCallback,
  setSpeakingMode,
  onSetUserSpeaking,
  sx,
}: Props) {
  const { enqueueSnackbar } = useSnackbar();

  const { config } = useSpeechConfig();
  const resource = getStorage(AZURE_SPEECH_KEY);

  const speechLang = config?.lang || 'en-US';

  const azureAISpeechKey = resource?.speechKey || '';
  const azureAISpeechRegion = resource?.speechRegion || '';

  const [isRecording, setIsRecording] = useState(false);
  const [isPreparing, setIsPreparing] = useState<boolean>(false);
  const [displaySpeech, setDisplaySpeech] = useState<string>('');
  const [displaySpeechChunks, setDisplaySpeechChunks] = useState<string[]>([]);

  const recognizerRef = useRef<SpeechRecognizer | null>(null);
  const partialTextLengthRef = useRef<number>(0);

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

    setIsPreparing(true);

    if (!azureAISpeechKey || !azureAISpeechRegion) {
      return;
    }

    const speechConfig = SpeechConfig.fromSubscription(azureAISpeechKey, azureAISpeechRegion);
    speechConfig.speechRecognitionLanguage = speechLang;
    speechConfig.outputFormat = OutputFormat.Simple;

    const audioConfig = AudioConfig.fromDefaultMicrophoneInput();

    // Initialize recognizer
    const recognizer = new SpeechRecognizer(speechConfig, audioConfig);
    recognizerRef.current = recognizer;

    recognizer.recognizing = (s, e) => {
      onSetUserSpeaking?.(true);
      setDisplaySpeech((prev) => {
        const prevTextLength = partialTextLengthRef.current;
        const updatedDisplayText = prev.slice(0, -prevTextLength) + e.result.text;
        return updatedDisplayText;
      });
    };

    recognizer.recognized = async (s, e) => {
      if (e.result.reason === ResultReason.RecognizedSpeech) {
        setDisplaySpeech('');
        partialTextLengthRef.current = 0;
        setDisplaySpeechChunks([]);
        onSetUserSpeaking?.(false);
        if (recognizedCallback) recognizedCallback(e.result.text);
      }
    };

    recognizer.canceled = (s, e) => {
      setIsPreparing(false);
      setIsRecording(false);
    };

    recognizer.sessionStopped = (s, e) => {
      setIsPreparing(false);
      setIsRecording(false);
    };

    recognizerRef.current?.startContinuousRecognitionAsync(
      () => {
        setIsPreparing(false);
        if (setSpeakingMode) setSpeakingMode(true);
      },
      (err) => console.error('Recognizer error:', err)
    );
  };

  const handleStopRecognition = () => {
    recognizerRef.current?.stopContinuousRecognitionAsync(
      () => {
        setIsRecording(false);
        if (setSpeakingMode) setSpeakingMode(false);
      },
      (err) => console.error('Error stopping recognizer:', err)
    );
    setIsPreparing(false);
  };

  const onToggleListening = async () => {
    if (isRecording) {
      handleStopRecognition();
    } else {
      handleStartRecognition();
    }
  };

  return (
    <Stack direction="row" spacing={1} alignItems="center">
      {isPreparing && (
        <CircularProgress
          color="primary"
          sx={{ minWidth: 32, maxWidth: 32, minHeight: 32, maxHeight: 32 }}
        />
      )}
      {!isPreparing && isRecording && (
        <Fab
          size="small"
          aria-label="microphone"
          onClick={onToggleListening}
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
      {!isPreparing && !isRecording && (
        <Tooltip title="Call GPT">
          <Fab
            size="small"
            aria-label="microphone"
            onClick={onToggleListening}
            sx={{
              minWidth: 32,
              maxWidth: 32,
              minHeight: 32,
              maxHeight: 32,
              boxShadow: 'None',
              background: 'linear-gradient(135deg, #0693E4, #C6A166)',
              '&:hover': {
                background: 'linear-gradient(135deg, #C6A166, #0693E4)',
              },
              ...sx,
            }}
          >
            <Iconify icon="fluent:call-24-filled" width={20} />
          </Fab>
        </Tooltip>
      )}
    </Stack>
  );
}
