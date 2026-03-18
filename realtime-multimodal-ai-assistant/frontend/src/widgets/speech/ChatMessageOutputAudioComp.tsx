import { useSnackbar } from 'notistack';
import { useRef, useState, useEffect } from 'react';
import {
  AudioConfig,
  ResultReason,
  SpeechConfig,
  SpeechSynthesizer,
  CancellationDetails,
  SpeakerAudioDestination,
} from 'microsoft-cognitiveservices-speech-sdk';

import Stack from '@mui/material/Stack';
import IconButton from '@mui/material/IconButton';

import { getStorage, AZURE_SPEECH_KEY } from 'src/hooks/local-storage';

import { cleanTextForAudio } from 'src/utils/string-processor';

import Iconify from 'src/widgets/iconify';

import { useSpeechConfig } from './context/speech-config-context';

// ----------------------------------------------------------------------

type Props = {
  speechText: string;
  status?: 'idle' | 'running';
  isLastMessage?: boolean;
  isSpeakingMode?: boolean;
  isUserSpeaking?: boolean;
};

export default function ChatMessageOutputAudioComp({
  speechText,
  status,
  isLastMessage,
  isSpeakingMode,
  isUserSpeaking,
}: Props) {
  const { enqueueSnackbar } = useSnackbar();
  const resource = getStorage(AZURE_SPEECH_KEY);
  const { config } = useSpeechConfig();

  const azureAISpeechKey = resource?.speechKey || '';
  const azureAISpeechRegion = resource?.speechRegion || '';

  const [isSpeaking, setSpeaking] = useState(false);
  const [isPaused, setIsPaused] = useState(false);

  // Use refs to persist synthesizer and player
  const synthesizerRef = useRef<SpeechSynthesizer | null>(null);
  const playerRef = useRef<SpeakerAudioDestination | null>(null);

  // Create synthesizer only when config or credentials change
  useEffect(() => {
    if (!azureAISpeechKey || !azureAISpeechRegion) return () => {};

    const speechConfig = SpeechConfig.fromSubscription(azureAISpeechKey, azureAISpeechRegion);
    const player = new SpeakerAudioDestination();

    player.onAudioStart = () => {
      setSpeaking(true);
      setIsPaused(false);
    };
    player.onAudioEnd = () => {
      setSpeaking(false);
      setIsPaused(false);
    };

    playerRef.current = player;
    const audioConfig = AudioConfig.fromSpeakerOutput(player);

    synthesizerRef.current = new SpeechSynthesizer(speechConfig, audioConfig);

    return () => {
      synthesizerRef.current?.close();
      synthesizerRef.current = null;
      playerRef.current = null;
    };
    // Only recreate if credentials or config change
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [azureAISpeechKey, azureAISpeechRegion, config]);

  useEffect(() => {
    if (isSpeakingMode && isLastMessage && status === 'idle') {
      textToSpeech();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isSpeakingMode, isLastMessage, status]);

  useEffect(() => {
    if ((!isLastMessage && isSpeaking) || isUserSpeaking || !isSpeakingMode) {
      handlePause();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isLastMessage, isUserSpeaking, isSpeaking, isSpeakingMode]);

  const textToSpeech = async () => {
    if (!azureAISpeechKey || !azureAISpeechRegion) {
      enqueueSnackbar(
        <>
          Speech key missing.
          <br />
          Please configure it in User/Account.
        </>,
        { variant: 'info' }
      );
      return;
    }
    const synth = synthesizerRef.current;
    if (!synth) return;
    // speechConfig.speechSynthesisVoiceName = 'zh-CN-Xiaochen:DragonHDLatestNeural';
    // speechConfig.speechSynthesisVoiceName = 'zh-CN-XiaoxiaoNeural';

    if (config) {
      synth.speakSsmlAsync(
        `<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis' xml:lang='${config.lang}'>
          <voice name='${config.voice}' style='${config.style}'>
            ${cleanTextForAudio(speechText)}
          </voice>
        </speak>`,
        (result) => {
          if (result.reason === ResultReason.Canceled) {
            const cancellation = CancellationDetails.fromResult(result);
            console.error(`Synthesis canceled: ${cancellation.errorDetails}`);
          }
          // Don't close synthesizer here, reuse it
        },
        (err) => {
          console.error(`Error synthesizing: ${err}`);
        }
      );
    } else {
      synth.speakTextAsync(
        cleanTextForAudio(speechText),
        (result) => {
          if (result.reason === ResultReason.Canceled) {
            const cancellation = CancellationDetails.fromResult(result);
            console.error(`Synthesis canceled: ${cancellation.errorDetails}`);
          }
        },
        (err) => {
          console.error(`Error synthesizing: ${err}`);
        }
      );
    }
  };

  async function handlePause() {
    playerRef.current?.pause();
    setIsPaused(true);
  }

  async function handleMute() {
    if (!playerRef.current) return;
    if (!isPaused) {
      playerRef.current.pause();
      setIsPaused(true);
    } else {
      playerRef.current.resume();
      setIsPaused(false);
    }
  }

  const handleClick = async () => {
    if (isSpeaking) {
      handleMute();
    } else {
      textToSpeech();
    }
  };

  const handleRefresh = () => {
    playerRef.current?.pause();
    setSpeaking(false);
    setIsPaused(false);
  };

  let icon = 'fluent:speaker-2-16-filled';
  if (isSpeaking && !isPaused) icon = 'fluent:speaker-off-16-filled';
  else if (isPaused) icon = 'fluent:speaker-2-16-filled';

  return (
    <Stack
      direction="row"
      className="message-actions"
      spacing={0.25}
      sx={{
        opacity: 0,
        transform: 'translateY(-2px)',
        transition: (theme) =>
          theme.transitions.create(['opacity'], {
            duration: theme.transitions.duration.shorter,
          }),
      }}
    >
      <IconButton size="small" onClick={handleRefresh}>
        <Iconify icon="ic:round-refresh" width={16} />
      </IconButton>
      <IconButton size="small" onClick={handleClick}>
        <Iconify icon={icon} width={16} />
      </IconButton>
    </Stack>
  );
}
