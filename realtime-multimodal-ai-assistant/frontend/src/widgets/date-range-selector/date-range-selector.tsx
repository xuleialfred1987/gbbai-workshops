import React from 'react';

// mui
import { DatePicker, DateCalendar } from '@mui/x-date-pickers';
import {
  Paper,
  Stack,
  Button,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  FormHelperText,
} from '@mui/material';

// project imports
import { useResponsiveUI } from 'src/hooks/responsive-ui';

import { DateRangeSelectorProps } from './types';

// ----------------------------------------------------------------------

const DateRangeDialog: React.FC<DateRangeSelectorProps> = ({
  title = 'Select date range',
  variant = 'input',
  startDate,
  endDate,
  onChangeStartDate,
  onChangeEndDate,
  open,
  onClose,
  error,
}) => {
  const isMediumScreen = useResponsiveUI('up', 'md');
  const useCalendar = variant === 'calendar';

  const dialogPaperStyles = useCalendar ? { sx: { maxWidth: 720 } } : undefined;

  return (
    <Dialog
      open={open}
      onClose={onClose}
      fullWidth
      maxWidth={useCalendar ? false : 'xs'}
      PaperProps={dialogPaperStyles}
    >
      <DialogTitle sx={{ pb: 2 }}>{title}</DialogTitle>
      <DialogContent sx={useCalendar && isMediumScreen ? { overflow: 'unset' } : {}}>
        <Stack
          direction={useCalendar && isMediumScreen ? 'row' : 'column'}
          spacing={useCalendar ? 3 : 2}
          justifyContent="center"
          sx={{ pt: 1 }}
        >
          {useCalendar ? (
            <>
              <Paper
                variant="outlined"
                sx={{
                  borderRadius: 2,
                  borderColor: 'divider',
                  borderStyle: 'dashed',
                }}
              >
                <DateCalendar value={startDate} onChange={onChangeStartDate} />
              </Paper>
              <Paper
                variant="outlined"
                sx={{
                  borderRadius: 2,
                  borderColor: 'divider',
                  borderStyle: 'dashed',
                }}
              >
                <DateCalendar value={endDate} onChange={onChangeEndDate} />
              </Paper>
            </>
          ) : (
            <>
              <DatePicker label="Start date" value={startDate} onChange={onChangeStartDate} />
              <DatePicker label="End date" value={endDate} onChange={onChangeEndDate} />
            </>
          )}
        </Stack>
        {error && (
          <FormHelperText error sx={{ px: 2 }}>
            End date must be later than start date
          </FormHelperText>
        )}
      </DialogContent>
      <DialogActions>
        <Button onClick={onClose} variant="outlined" color="inherit">
          Cancel
        </Button>
        <Button onClick={onClose} variant="contained" disabled={error}>
          Apply
        </Button>
      </DialogActions>
    </Dialog>
  );
};

export default DateRangeDialog;
