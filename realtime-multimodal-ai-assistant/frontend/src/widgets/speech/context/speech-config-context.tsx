import {
  useMemo,
  useState,
  ReactNode,
  useEffect,
  useContext,
  useCallback,
  createContext,
} from 'react';

import { getStorage, setStorage, AZURE_SPEECH_CONFIG } from 'src/hooks/local-storage';

import { SpeechOption } from 'src/widgets/speech/components/langs';

// ----------------------------------------------------------------------

interface SpeechConfigContextType {
  config: SpeechOption | null;
  setConfig: (config: SpeechOption) => void;
}

const SpeechConfigContext = createContext<SpeechConfigContextType | undefined>(undefined);

export const useSpeechConfig = () => {
  const context = useContext(SpeechConfigContext);
  if (!context) throw new Error('useSpeechConfig must be used within a SpeechConfigProvider');
  return context;
};

export function SpeechConfigProvider({ children }: { children: ReactNode }) {
  const [config, setConfigState] = useState<SpeechOption | null>(getStorage(AZURE_SPEECH_CONFIG));

  const setConfig = useCallback((newConfig: SpeechOption) => {
    setConfigState(newConfig);
    setStorage(AZURE_SPEECH_CONFIG, newConfig);
  }, []);

  useEffect(() => {
    setStorage(AZURE_SPEECH_CONFIG, config);
  }, [config]);

  const contextValue = useMemo(() => ({ config, setConfig }), [config, setConfig]);

  return (
    <SpeechConfigContext.Provider value={contextValue}>{children}</SpeechConfigContext.Provider>
  );
}
