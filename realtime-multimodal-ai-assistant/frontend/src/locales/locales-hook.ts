import { useMemo, useCallback } from 'react';
import { useTranslation } from 'react-i18next';

// project imports
import { retrieveFromLocalStorage } from 'src/utils/local-storage';

import { allLangs, defaultLang } from './config-lang';

// ----------------------------------------------------------------------

export const useLocales = () => {
  // Get language from storage or use default
  const storedLanguage = retrieveFromLocalStorage('i18nextLng');

  // Find the language configuration or fall back to default
  const currentLang = useMemo(
    () => allLangs.find((lang) => lang.value === storedLanguage) || defaultLang,
    [storedLanguage]
  );

  // Return language data
  return useMemo(
    () => ({
      allLangs,
      currentLang,
    }),
    [currentLang]
  );
};

// ----------------------------------------------------------------------

export const useTranslate = () => {
  // Get translation utilities from i18next
  const translation = useTranslation();
  const { t, i18n, ready } = translation;

  // Handler for changing the active language
  const handleLanguageChange = useCallback(
    (newLanguageCode: string) => {
      i18n.changeLanguage(newLanguageCode);
    },
    [i18n]
  );

  // Return translation utilities
  return {
    t,
    i18n,
    ready,
    onChangeLang: handleLanguageChange,
  };
};
