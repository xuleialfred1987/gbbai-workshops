// mui
import { Theme } from '@mui/material/styles';
import { buttonClasses } from '@mui/material/Button';

// project imports
import Iconify from 'src/widgets/iconify';

// ----------------------------------------------------------------------

// Icon definitions
const Icons = {
  switchView: () => <Iconify icon="eva:chevron-down-fill" width={24} />,
  leftArrow: () => <Iconify icon="eva:arrow-ios-back-fill" width={24} />,
  rightArrow: () => <Iconify icon="eva:arrow-ios-forward-fill" width={24} />,
  calendar: () => <Iconify icon="solar:calendar-mark-bold-duotone" width={24} />,
  clock: () => <Iconify icon="solar:clock-circle-outline" width={24} />,
};

// Picker type definitions
const pickerTypes = {
  date: [
    'DatePicker',
    'DateTimePicker',
    'StaticDatePicker',
    'DesktopDatePicker',
    'DesktopDateTimePicker',
    'MobileDatePicker',
    'MobileDateTimePicker',
  ],
  time: ['TimePicker', 'MobileTimePicker', 'StaticTimePicker', 'DesktopTimePicker'],
};

// Helper function to generate picker configurations
const generatePickerConfig = (types: string[], iconSet: Record<string, () => JSX.Element>) =>
  types.reduce(
    (acc, type) => {
      acc[`Mui${type}`] = {
        defaultProps: {
          slots: iconSet,
        },
      };
      return acc;
    },
    {} as Record<string, any>
  );

// Picker configurations
const datePickerConfig = generatePickerConfig(pickerTypes.date, {
  openPickerIcon: Icons.calendar,
  leftArrowIcon: Icons.leftArrow,
  rightArrowIcon: Icons.rightArrow,
  switchViewIcon: Icons.switchView,
});

const timePickerConfig = generatePickerConfig(pickerTypes.time, {
  openPickerIcon: Icons.clock,
  rightArrowIcon: Icons.rightArrow,
  switchViewIcon: Icons.switchView,
});

// Main export function
export function datePicker(theme: Theme) {
  const actionBarStyles = {
    backgroundColor: theme.palette.text.primary,
    color: theme.palette.mode === 'light' ? theme.palette.common.white : theme.palette.grey[800],
  };

  return {
    MuiPickersLayout: {
      styleOverrides: {
        root: {
          [`& .MuiPickersLayout-actionBar .${buttonClasses.root}:last-of-type`]: actionBarStyles,
        },
      },
    },
    ...datePickerConfig,
    ...timePickerConfig,
  };
}
