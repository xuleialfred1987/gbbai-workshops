import { FC, ReactNode } from 'react';

// mui
import { AdapterDateFns } from '@mui/x-date-pickers/AdapterDateFns';
import { LocalizationProvider as MuiDateLocalizationProvider } from '@mui/x-date-pickers/LocalizationProvider';

// project imports
import { useLocales } from './locales-hook';

// ----------------------------------------------------------------------

// Define component props type
type DateLocalizationProps = {
  children: ReactNode;
};

const DateLocalizationWrapper: FC<DateLocalizationProps> = ({ children }) => {
  // Extract locale settings from the hook
  const { currentLang } = useLocales();
  const { adapterLocale } = currentLang;

  // Configure the date adapter with the current locale
  const localizationConfig = {
    dateAdapter: AdapterDateFns,
    adapterLocale,
  };

  return (
    <MuiDateLocalizationProvider {...localizationConfig}>{children}</MuiDateLocalizationProvider>
  );
};

export default DateLocalizationWrapper;
