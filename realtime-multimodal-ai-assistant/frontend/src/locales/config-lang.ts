import { merge } from 'lodash';
import {
  ja as japaneseDateAdapter,
  enUS as englishDateAdapter,
  zhCN as chineseDateAdapter,
} from 'date-fns/locale';

// MUI component localization imports
import * as muiCore from '@mui/material/locale';
import * as muiDataGrid from '@mui/x-data-grid';
import * as muiDatePickers from '@mui/x-date-pickers/locales';

// ----------------------------------------------------------------------

// Define supported locales with their configurations
const SUPPORTED_LOCALES = {
  ENGLISH: {
    displayName: 'English',
    code: 'en',
    muiLocale: merge(muiDatePickers.enUS, muiDataGrid.enUS, muiCore.enUS),
    dateAdapter: englishDateAdapter,
    flagIcon: 'flagpack:gb-nir',
  },
  CHINESE: {
    displayName: '中文',
    code: 'cn',
    muiLocale: merge(muiDatePickers.zhCN, muiDataGrid.zhCN, muiCore.zhCN),
    dateAdapter: chineseDateAdapter,
    flagIcon: 'flagpack:cn',
  },
  JAPANESE: {
    displayName: '日本語',
    code: 'jp',
    muiLocale: merge(muiDatePickers.jaJP, muiDataGrid.jaJP, muiCore.jaJP),
    dateAdapter: japaneseDateAdapter,
    flagIcon: 'flagpack:jp',
  },
};

// Convert object to array format
const localesList = Object.values(SUPPORTED_LOCALES).map((locale) => ({
  label: locale.displayName,
  value: locale.code,
  systemValue: locale.muiLocale,
  adapterLocale: locale.dateAdapter,
  icon: locale.flagIcon,
}));

// Set default language to English
const primaryLocale = localesList[0];

// Export with renamed variables
export { localesList as allLangs, primaryLocale as defaultLang };
