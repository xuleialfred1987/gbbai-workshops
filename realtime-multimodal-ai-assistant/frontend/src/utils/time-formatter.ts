import { format, getTime, formatDistanceToNow } from 'date-fns';

// ----------------------------------------------------------------------

type InputValue = Date | string | number | null | undefined;

export function fDate(date: InputValue, newFormat?: string) {
  const fm = newFormat || 'dd MMM yyyy';

  return date ? format(new Date(date), fm) : '';
}

export function fShortDate(date: InputValue, newFormat?: string) {
  const fm = newFormat || 'yy-MM-dd';

  return date ? format(new Date(date), fm) : '';
}

export function fDateTime(date: InputValue, newFormat?: string) {
  const fm = newFormat || 'dd MMM yyyy p';

  return date ? format(new Date(date), fm) : '';
}

export function fDateTimeYMdHm(date: InputValue) {
  return date ? format(new Date(date), 'yyyy-MM-dd HH:mm') : '';
}

export function fDateTimeYMdHms(date: InputValue) {
  return date ? format(new Date(date), 'yyyy-MM-dd HH:mm:ss') : '';
}

export function fTimestamp(date: InputValue) {
  return date ? getTime(new Date(date)) : '';
}

export function fToNow(date: InputValue) {
  return date
    ? formatDistanceToNow(new Date(date), {
        addSuffix: true,
      })
    : '';
}

export function reformDate(date: InputValue) {
  if (!date) return '';

  const dateString = date.toString();
  const hasTime = dateString.includes(':'); // Check if there is a space (indicating time)
  // console.log(dateString);

  if (hasTime) {
    return fDateTime(date);
  }
  return fDateTime(date);
}

export function fSecondsToHHMMSS(timeStr: string): string {
  // Ensure the input ends with "s" and has a decimal point
  if (!timeStr.endsWith('s')) {
    throw new Error('Time must end with "s"');
  }

  // Remove the trailing "s" and parse the number
  const totalSecondsStr = timeStr.slice(0, -1);
  const totalSeconds = parseFloat(totalSecondsStr);

  // Ensure the parsed value is a valid number
  if (Number.isNaN(totalSeconds)) {
    throw new Error('Invalid number of seconds');
  }

  // Convert total seconds to hours, minutes, and remaining seconds
  const hours = Math.floor(totalSeconds / 3600);
  const minutes = Math.floor((totalSeconds % 3600) / 60);
  const seconds = Math.floor(totalSeconds % 60);

  // Convert hours, minutes, and seconds to strings
  const hoursStr = hours.toString();
  const minutesStr = minutes.toString();
  const secondsStr = seconds.toString();

  // Conditionally format the output based on the presence of hours
  if (hours > 0) {
    return `${hoursStr}:${minutesStr.padStart(2, '0')}:${secondsStr.padStart(2, '0')}`;
  }
  return `${minutesStr}:${secondsStr.padStart(2, '0')}`;
}
