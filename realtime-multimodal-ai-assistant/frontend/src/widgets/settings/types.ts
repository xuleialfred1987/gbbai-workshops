type ThemeModeOptions = 'light' | 'dark';
type ThemeLayoutOptions = 'vertical' | 'horizontal' | 'mini';
type ThemeColorOptions = 'default' | 'cyan';

interface BaseSettings {
  themeStretch: boolean;
  themeMode: ThemeModeOptions;
  themeLayout: ThemeLayoutOptions;
  themeColorPresets: ThemeColorOptions;
}

interface SettingsUpdate {
  onUpdate(name: string, value: string | boolean): void;
}

export type SettingsValueProps = BaseSettings;
export type SettingsContextProps = BaseSettings & SettingsUpdate;
