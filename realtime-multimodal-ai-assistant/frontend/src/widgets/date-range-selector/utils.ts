import { getYear, isSameDay, isSameMonth } from 'date-fns';

// project imports
import { fDate } from 'src/utils/time-formatter';

// ----------------------------------------------------------------------

export const getShortDateLabel = (start: Date | null, end: Date | null): string => {
  const currentYear = new Date().getFullYear();
  const startYear = start ? getYear(start) : null;
  const endYear = end ? getYear(end) : null;

  const withinSameCurrentYear = startYear === currentYear && endYear === currentYear;
  const singleDay = start && end ? isSameDay(new Date(start), new Date(end)) : false;
  const sameMonthRange = start && end ? isSameMonth(new Date(start), new Date(end)) : false;

  if (withinSameCurrentYear) {
    if (sameMonthRange) {
      return singleDay
        ? fDate(end, 'dd MMM yy')
        : `${fDate(start, 'dd')} - ${fDate(end, 'dd MMM yy')}`;
    }
    return `${fDate(start, 'dd MMM')} - ${fDate(end, 'dd MMM yy')}`;
  }

  return `${fDate(start, 'dd MMM yy')} - ${fDate(end, 'dd MMM yy')}`;
};
