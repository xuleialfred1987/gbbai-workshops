import React from 'react';

// project imports
import type { SettingsContextProps } from '../types';

// -----------------------------------------------------------------------

const INITIAL_STATE: SettingsContextProps = {} as SettingsContextProps;

/**
 * React context for managing application theme settings
 */
export const SettingsContext = React.createContext<SettingsContextProps>(INITIAL_STATE);

/**
 * Custom hook for accessing settings context values and methods
 * @throws {Error} If used outside of SettingsProvider
 * @returns {SettingsContextProps} Settings context value
 */
export const useSettingsContext = (): SettingsContextProps => {
  // Retrieve settings from context
  const settingsData = React.useContext<SettingsContextProps>(SettingsContext);

  // Validate context availability
  if (settingsData === INITIAL_STATE || !Object.keys(settingsData).length) {
    throw new Error('useSettingsContext must be use inside SettingsProvider');
  }

  return settingsData;
};
