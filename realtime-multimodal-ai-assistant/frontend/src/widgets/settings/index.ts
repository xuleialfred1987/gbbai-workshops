import { SettingsContextProvider } from './context/settings-provider';
import type { SettingsValueProps, SettingsContextProps } from './types';
import { SettingsContext, useSettingsContext } from './context/settings-context';

// ----------------------------------------------------------------------

// Primary exports
export {
  // Context components
  SettingsContext,
  useSettingsContext,
  SettingsContextProvider,
};

// Type exports
export type { SettingsValueProps, SettingsContextProps };

// Singleton instance for easier consumption
const SettingsModule = {
  Context: SettingsContext,
  Provider: SettingsContextProvider,
  useSettings: useSettingsContext,
};

export default SettingsModule;
