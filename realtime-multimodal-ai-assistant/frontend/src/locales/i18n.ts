import i18next from 'i18next';
import { initReactI18next } from 'react-i18next';
import LanguageDetector from 'i18next-browser-languagedetector';

// project imports
import { retrieveFromLocalStorage } from 'src/utils/local-storage';

import en from './langs/en.json';
import cn from './langs/cn.json';
import jp from './langs/jp.json';
import { defaultLang } from './config-lang';

// ----------------------------------------------------------------------

// Define language resources mapping
const LANGUAGE_RESOURCES = {
  en: { translations: en },
  cn: { translations: cn },
  jp: { translations: jp },
};

// Configuration options for i18next
const CONFIG_OPTIONS = {
  resources: LANGUAGE_RESOURCES,
  lng: retrieveFromLocalStorage('i18nextLng', defaultLang.value),
  fallbackLng: retrieveFromLocalStorage('i18nextLng', defaultLang.value),
  debug: false,
  ns: ['translations'],
  defaultNS: 'translations',
  interpolation: {
    escapeValue: false,
  },
};

// Initialize the i18next instance with plugins
const i18nInstance = i18next.use(LanguageDetector).use(initReactI18next);

// Configure and initialize i18next
i18nInstance.init(CONFIG_OPTIONS);

// Export the configured instance
export default i18nInstance;
