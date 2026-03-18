import * as React from 'react';

// project imports
import { useLocalStorage } from 'src/hooks/local-storage';

import type { SettingsValueProps } from '../types';
import { SettingsContext } from './settings-context';

// ----------------------------------------------------------------------

/** Storage key for persisting settings */
const STORAGE_KEY = 'gbbai-settings';

/** Default application settings */
const DEFAULT_SETTINGS: SettingsValueProps = {
  themeMode: 'light',
  themeLayout: 'vertical',
  themeColorPresets: 'default',
  themeStretch: false,
};

/** Component props definition */
interface ProviderProps {
  children: React.ReactNode;
}

// ----------------------------------------------------------------------
// Provider Component
// ----------------------------------------------------------------------

/**
 * Settings context provider component
 * Manages application theme settings with local storage persistence
 */
export const SettingsContextProvider: React.FC<ProviderProps> = ({ children }) => {
  // Load persisted settings from storage
  const { state: persistedSettings, update: persistSettings } = useLocalStorage(
    STORAGE_KEY,
    DEFAULT_SETTINGS
  );

  // Create memoized context value to prevent unnecessary renders
  const settingsContextValue = React.useMemo(
    () => ({
      ...persistedSettings,
      onUpdate: persistSettings,
    }),
    [persistedSettings, persistSettings]
  );

  return (
    <SettingsContext.Provider value={settingsContextValue}>{children}</SettingsContext.Provider>
  );
};
