import { useSnackbar } from 'notistack';
import { useRef, useState, useEffect, useCallback } from 'react';
import {
  Coordinate,
  AudioConfig,
  OutputFormat,
  AvatarConfig,
  ResultReason,
  SpeechConfig,
  SpeechRecognizer,
  AvatarSynthesizer,
  AvatarVideoFormat,
} from 'microsoft-cognitiveservices-speech-sdk';

import Paper from '@mui/material/Paper';
import { keyframes } from '@mui/material';
import IconButton from '@mui/material/IconButton';

import { getStorage, AZURE_SPEECH_KEY } from 'src/hooks/local-storage';

import { cleanTextForAudio } from 'src/utils/string-processor';

import Iconify from 'src/widgets/iconify';

import { useSpeechConfig } from './context/speech-config-context';

// ----------------------------------------------------------------------

const ZINDEX = 1998;

const slideAnimation = keyframes`
  0% {
    opacity: 0;
    transform: translateY(50%);
  }
  60% {
    opacity: 1;
    transform: translateY(10px);
  }
  80% {
    transform: translateY(-5px);
  }
  100% {
    transform: translateY(0);
  }
`;

type Props = {
  speechText: string;
  status: 'idle' | 'running';
  isLastMessage?: boolean;
  isSpeakingMode?: boolean;
  isUserSpeaking?: boolean;
  isOpen?: boolean;
  onCloseAvatar: VoidFunction;
  onSetUserSpeaking?: (flag: boolean) => void;
  recognizedCallback?: (text: string) => void;
  onSetInputText?: (text: string) => void;
};

export default function ChatMessageAvatarComp({
  speechText,
  status,
  isLastMessage,
  isSpeakingMode,
  isUserSpeaking,
  isOpen,
  onCloseAvatar,
  onSetInputText,
  onSetUserSpeaking,
  recognizedCallback,
}: Props) {
  const { enqueueSnackbar } = useSnackbar();

  const { config } = useSpeechConfig();
  const resource = getStorage(AZURE_SPEECH_KEY);

  const azureAISpeechKey = resource?.speechKey || '';
  const azureAISpeechRegion = resource?.speechRegion || '';

  // const [logging, setLogging] = useState<string>('');
  const [spokenText, setSpokenText] = useState<string>(speechText);
  const [isSpeaking, setIsSpeaking] = useState<boolean>(false);
  // const [ttsVoice, setTtsVoice] = useState<string>('en-US-AriaNeural');
  const [isSessionStarted, setIsSessionStarted] = useState<boolean>(false);
  const [peerConnection, setPeerConnection] = useState<RTCPeerConnection | null>(null);
  const [avatarSynthesizer, setAvatarSynthesizer] = useState<AvatarSynthesizer | null>(null);
  // const [personalVoiceSpeakerProfileID, setPersonalVoiceSpeakerProfileID] = useState<string>('');

  const isSpeakingRef = useRef<boolean>(false);
  const audioRef = useRef<HTMLAudioElement>(null);
  const connectionRetryCountRef = useRef<number>(0);
  const remoteVideoRef = useRef<HTMLVideoElement>(null);
  const avatarSynthesizerRef = useRef<AvatarSynthesizer | null>(null);

  const log = (msg: string) => {
    console.log(msg); // Log to the browser console
    // setLogging((prev) => `${prev}${msg}\n`);
  };

  const setIsSpeakingWithRef = (value: boolean) => {
    setIsSpeaking(value);
    isSpeakingRef.current = value;
  };

  // Add a retry function
  const retryConnection = useCallback(() => {
    if (connectionRetryCountRef.current < 3) {
      connectionRetryCountRef.current += 1;
      log(`Connection attempt ${connectionRetryCountRef.current}/3...`);

      setTimeout(() => {
        startSession();
      }, 2000); // Wait 2 seconds before retrying
      return true;
    }

    log('Maximum connection attempts reached (3). Please try again later.');
    enqueueSnackbar('Failed to connect after multiple attempts', { variant: 'error' });
    return false;
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // const htmlEncode = (text: string) => {
  //   const entityMap: Record<string, string> = {
  //     '&': '&amp;',
  //     '<': '&lt;',
  //     '>': '&gt;',
  //     '"': '&quot;',
  //     "'": '&#39;',
  //     '/': '&#x2F;',
  //   };
  //   return String(text).replace(/[&<>"'\/]/g, (match) => entityMap[match]);
  // };

  const speechLang = config?.lang || 'en-US';

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
    try {
      setDisplaySpeech('');
      setDisplaySpeechChunks([]);
      partialTextLengthRef.current = 0;

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

      recognizer.recognizing = async (s, e) => {
        try {
          if (isSpeakingRef.current && avatarSynthesizerRef.current !== null) {
            await avatarSynthesizerRef.current.stopSpeakingAsync();
            console.log('Forced avatar to stop speaking on user speech');
            setIsSpeakingWithRef(false);
          }
        } catch (error) {
          console.error('Failed to stop avatar speaking:', error);
        }
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
          // onSetUserSpeaking?.(false);
          if (recognizedCallback) recognizedCallback(e.result.text);
        }
      };

      recognizer.canceled = (s, e) => {};

      recognizer.sessionStopped = (s, e) => {};

      recognizerRef.current?.startContinuousRecognitionAsync(
        () => {
          // if (setSpeakingMode) setSpeakingMode(true);
        },
        (err) => console.error('Recognizer error:', err)
      );
    } catch (error) {
      console.error('Error initializing recognizer:', error);
      enqueueSnackbar('Failed to start speech recognition', { variant: 'error' });
      // setIsPreparing(false);
    }
  };

  const handleStopRecognition = useCallback(() => {
    if (recognizerRef.current) {
      console.log('Stopping speech recognition...');

      // Set a cleanup timeout as a fallback
      const forceCleanupTimeout = setTimeout(() => {
        console.warn('Force cleanup triggered - recognition stop timed out');
        if (recognizerRef.current) {
          try {
            recognizerRef.current.close();
          } catch (e) {
            console.error('Error in force cleanup:', e);
          }
          recognizerRef.current = null;
        }
      }, 3000); // Give it 3 seconds before force cleanup

      try {
        // First attempt to properly stop recognition
        recognizerRef.current.stopContinuousRecognitionAsync(
          () => {
            console.log('Speech recognition stopped successfully');
            clearTimeout(forceCleanupTimeout);

            // Even after successful stop, ensure close is called
            if (recognizerRef.current) {
              try {
                recognizerRef.current.close();
              } catch (e) {
                console.error('Error closing recognizer after stop:', e);
              } finally {
                recognizerRef.current = null;
              }
            }
          },
          (err) => {
            console.error('Error stopping recognizer:', err);
            clearTimeout(forceCleanupTimeout);

            if (recognizerRef.current) {
              try {
                recognizerRef.current.close();
              } catch (e) {
                console.error('Error closing recognizer:', e);
              } finally {
                recognizerRef.current = null;
              }
            }
          }
        );
      } catch (error) {
        console.error('Exception during recognition cleanup:', error);
        clearTimeout(forceCleanupTimeout);
        recognizerRef.current = null;
      }
    }
  }, []);

  const setupWebRTC = async (
    iceServerUrl: string,
    iceServerUsername: string,
    iceServerCredential: string,
    synthesizer: AvatarSynthesizer
  ) => {
    if (!iceServerUrl || !iceServerUsername || !iceServerCredential) {
      log('Invalid ICE server credentials.');
      return;
    }

    log('Initializing RTCPeerConnection...');
    const pc = new RTCPeerConnection({
      iceServers: [
        {
          urls: [
            iceServerUrl.includes(':443?transport=tcp')
              ? iceServerUrl
              : iceServerUrl.replace(':3478', ':443?transport=tcp'),
          ],
          username: iceServerUsername,
          credential: iceServerCredential,
        },
      ],
      iceTransportPolicy: iceServerUrl.includes('transport=tcp') ? 'relay' : 'all',
    });

    setPeerConnection(pc);
    log('RTCPeerConnection initialized.');

    pc.onicecandidate = (event) => {
      if (event.candidate) {
        log(`ICE Candidate: ${event.candidate.candidate}`);
      }
    };

    pc.onconnectionstatechange = () => {
      log(`Connection state changed to: ${pc.connectionState}`);
      if (pc.connectionState === 'connected') {
        connectionRetryCountRef.current = 0;
        setIsSessionStarted(true);

        setTimeout(() => {
          console.log('Starting speech recognition after connection established');
          handleStartRecognition();
        }, 500);
      } else if (pc.connectionState === 'disconnected' || pc.connectionState === 'failed') {
        setIsSessionStarted(false);
        setIsSpeakingWithRef(false);
        if (pc.connectionState === 'failed') {
          log('Connection failed, attempting to retry...');
          retryConnection();
        }
      }
    };

    pc.ontrack = (event) => {
      const stream = event.streams[0];
      const videoElement = remoteVideoRef.current;
      const audioElement = audioRef.current;

      if (stream) {
        const videoTracks = stream.getVideoTracks();
        const audioTracks = stream.getAudioTracks();

        log(
          `Stream has ${videoTracks.length} video track(s) and ${audioTracks.length} audio track(s).`
        );

        // Handle video tracks
        if (videoTracks.length > 0 && videoElement) {
          if (videoElement.srcObject !== stream) {
            setIsSessionStarted(true);
            log('Setting srcObject for video element.');
            videoElement.srcObject = stream;

            videoElement.onloadedmetadata = () => {
              videoElement
                .play()
                .then(() => {
                  log('Video playback started successfully.');
                })
                .catch((err) => {
                  // log('Error playing video: ' + err);
                });
            };
          } else {
            log('Video srcObject already set.');
          }
        }

        // Handle audio tracks
        if (audioTracks.length > 0 && audioElement) {
          if (audioElement.srcObject !== stream) {
            log('Setting srcObject for audio element.');
            audioElement.srcObject = stream;

            audioElement.onloadedmetadata = () => {
              audioElement
                .play()
                .then(() => {
                  log('Audio playback started successfully.');
                })
                .catch((err) => {
                  // log('Error playing audio: ' + err);
                });
            };
          } else {
            log('Audio srcObject already set.');
          }
        } else {
          log('No audio tracks found in the stream.');
        }
      } else {
        log('No stream found.');
      }
    };

    // Adding transceivers
    pc.addTransceiver('video', { direction: 'sendrecv' });
    pc.addTransceiver('audio', { direction: 'sendrecv' });

    try {
      const result = await synthesizer.startAvatarAsync(pc);
      if (result?.reason === ResultReason.SynthesizingAudioCompleted) {
        log('Avatar started successfully.');
      } else {
        log(`Failed to start avatar: ${result?.reason}`);
      }
    } catch (error: any) {
      log(`Error starting avatar: ${error.message}`);
      retryConnection();
    }
  };

  const startSession = async () => {
    if (connectionRetryCountRef.current === 0) {
      log('Initializing connection...');
    }
    if (!azureAISpeechKey) {
      alert('Please provide the API key.');
      return;
    }
    let speechConfig: SpeechConfig;
    try {
      speechConfig = SpeechConfig.fromSubscription(azureAISpeechKey, azureAISpeechRegion);
      log('SpeechConfig created successfully.');
    } catch (error: any) {
      log(`Error creating SpeechConfig: ${error.message}`);
      return;
    }

    const videoFormat = new AvatarVideoFormat();
    videoFormat.setCropRange(new Coordinate(0, 0), new Coordinate(1920, 1080));

    const avatarConfig = new AvatarConfig('lisa', 'casual-sitting', videoFormat);
    log('AvatarConfig created.');

    let synthesizer: AvatarSynthesizer;
    try {
      synthesizer = new AvatarSynthesizer(speechConfig, avatarConfig);
      log('AvatarSynthesizer created successfully.');
    } catch (error: any) {
      log(`Error creating AvatarSynthesizer: ${error.message}`);
      return;
    }

    synthesizer.avatarEventReceived = (s: any, e: any) => {
      const offsetMessage =
        e.offset !== 0 ? `, offset from session start: ${e.offset / 10000}ms.` : '';
      // console.log(`[${new Date().toISOString()}] Event received: ${e.description}${offsetMessage}`);
      log(`[Event] ${e.description}${offsetMessage}`);
    };

    setAvatarSynthesizer(synthesizer);
    avatarSynthesizerRef.current = synthesizer;

    try {
      const response = await fetch(
        `https://${azureAISpeechRegion}.tts.speech.microsoft.com/cognitiveservices/avatar/relay/token/v1`,
        {
          headers: {
            'Ocp-Apim-Subscription-Key': azureAISpeechKey,
          },
        }
      );
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      const data = await response.json();
      // log('ICE Server Response: ' + JSON.stringify(data));

      const iceServerUrl = data?.Urls?.[0];
      const iceServerUsername = data?.Username;
      const iceServerCredential = data?.Password;

      if (!iceServerUrl || !iceServerUsername || !iceServerCredential) {
        throw new Error('Incomplete ICE server credentials received.');
      }

      await setupWebRTC(iceServerUrl, iceServerUsername, iceServerCredential, synthesizer);

      // setTimeout(() => {
      //   console.log('Starting speech recognition after 1 second delay');
      //   handleStartRecognition();
      // }, 1000);
    } catch (error: any) {
      log(`Error fetching ICE server credentials: ${error.message}`);
    }
  };

  const speak = useCallback(
    async (_text: string) => {
      if (avatarSynthesizer && _text) {
        setIsSpeakingWithRef(true);
        // let ssml = `<speak version='1.0' xmlns='http://www.w3.org/2001/10/synthesis'
        // xmlns:mstts='http://www.w3.org/2001/mstts' xml:lang='en-US'>
        // <voice name='${ttsVoice}'>`;

        // if (personalVoiceSpeakerProfileID.trim() !== '') {
        //   ssml += `<mstts:ttsembedding speakerProfileId='${personalVoiceSpeakerProfileID}'>
        //           ${htmlEncode(_text)}
        //         </mstts:ttsembedding>`;
        // } else {
        //   ssml += `${htmlEncode(_text)}`;
        // }

        // ssml += `</voice></speak>`;

        try {
          // const result = await avatarSynthesizer.speakSsmlAsync(ssml);
          const result = await avatarSynthesizer.speakTextAsync(_text);
          if (result?.reason === ResultReason.SynthesizingAudioCompleted) {
            log('Speech synthesis completed.');
          } else {
            log(`Speech synthesis failed: ${result?.reason}`);
          }

          // Trigger audio playback as part of user interaction
          const videoElement = remoteVideoRef.current;
          if (videoElement) {
            log('Attempting to play video for audio playback.');
            const playPromise = videoElement.play();
            if (playPromise !== undefined) {
              playPromise
                .then(() => {
                  log('Video playback triggered successfully.');
                })
                .catch((err) => {
                  // log('Error triggering video playback: ' + err);
                });
            }
          } else {
            log('Video element not found.');
          }
        } catch (error: any) {
          log(`Error during speech synthesis: ${error.message}`);
        } finally {
          setIsSpeakingWithRef(false);
        }
      }
    },
    [avatarSynthesizer, remoteVideoRef]
  );

  const stopSpeaking = async () => {
    log('Attempting to stop speaking...');
    if (avatarSynthesizer !== null) {
      try {
        await avatarSynthesizer.stopSpeakingAsync();
        setIsSpeakingWithRef(false);
        log('Successfully stopped speaking.');
      } catch (error) {
        console.error('Error stopping speech:', error);
      }
    }
  };

  const stopSession = useCallback(() => {
    handleStopRecognition();

    if (avatarSynthesizer) {
      avatarSynthesizer.close();
      setAvatarSynthesizer(null);
      avatarSynthesizerRef.current = null;
    }

    if (peerConnection) {
      peerConnection.close();
      setPeerConnection(null);
    }

    setIsSessionStarted(false);
  }, [avatarSynthesizer, peerConnection, handleStopRecognition]);

  useEffect(() => {
    if (avatarSynthesizer && speechText && isSessionStarted && status === 'idle') {
      speak(cleanTextForAudio(speechText));
      setSpokenText(speechText);
    }
  }, [speak, status, speechText, avatarSynthesizer, isSessionStarted]);

  useEffect(() => {
    if (isOpen) {
      startSession();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen]);

  useEffect(
    () =>
      // Return the cleanup function directly without block statement
      () => {
        stopSession();

        if (recognizerRef.current) {
          console.log('Component unmounting - forcing recognizer cleanup');
          try {
            recognizerRef.current.close();
          } catch (e) {
            console.error('Error closing recognizer on unmount:', e);
          }
          recognizerRef.current = null;
        }
      },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    []
  );

  const handleRefresh = () => {
    if (isSpeaking) {
      stopSpeaking();
    } else {
      speak(cleanTextForAudio(spokenText));
    }
  };

  let icon = 'fluent:speaker-2-16-filled';
  if (isSpeaking) icon = 'fluent:speaker-off-16-filled';

  return (
    <Paper
      sx={{
        p: 2.5,
        borderRadius: 1,
        position: 'absolute',
        zIndex: ZINDEX + 1,
        overflow: 'hidden',
        flexDirection: 'column',
        right: -88,
        bottom: 26,
        width: `282px`,
        height: `262px`,
        background: 'transparent',
        animation: `${slideAnimation} 0.5s ease-in`, // Apply the animation
      }}
    >
      {isSessionStarted && (
        <IconButton
          size="small"
          onClick={handleRefresh}
          sx={{
            right: 32,
            bottom: 16,
            width: 32,
            height: 32,
            position: 'absolute',
            zIndex: ZINDEX + 2,
          }}
        >
          <Iconify icon={icon} width={18} />
        </IconButton>
      )}

      <div
        style={{
          width: '220px',
          height: '220px',
          borderRadius: '50%',
          overflow: 'hidden',
          margin: '0 auto',
          // border: '3.75px solid transparent',
          background: 'linear-gradient(135deg, #CF46E0 0%, #297BE7 74%)',
          boxShadow: '0 0 16px 4px rgba(40, 40, 40, 0.18)',
        }}
      >
        <video
          ref={remoteVideoRef}
          autoPlay
          playsInline
          // controls
          style={{
            margin: '3.5px',
            width: 'calc(100% - 7px)',
            height: 'calc(100% - 7px)',
            objectFit: 'cover',
            borderRadius: '50%',
            background: 'black',
          }}
        >
          <track kind="captions" />
        </video>
        <audio ref={audioRef} autoPlay controls style={{ display: 'none' }}>
          <track kind="captions" />
        </audio>
      </div>
    </Paper>
  );
}
