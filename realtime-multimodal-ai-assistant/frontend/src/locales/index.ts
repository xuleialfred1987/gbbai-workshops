// project imports
import LocaleProvider from './locale-provider';
import { allLangs, defaultLang } from './config-lang';
import { useLocales, useTranslate } from './locales-hook';

// ----------------------------------------------------------------------

export {
  // Language configuration exports
  allLangs,
  useLocales, // Hook
  defaultLang,

  // Hook exports
  useTranslate,

  // Locale provider component
  LocaleProvider,
};
