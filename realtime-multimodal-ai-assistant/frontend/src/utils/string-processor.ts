// ----------------------------------------------------------------------

export function escapeSpecialCharacters(str: string | undefined): string {
  // Escapes all non-alphanumeric characters with a backslash
  if (str === undefined) return '';
  if (typeof str !== 'string') return str;
  return str.replace(/(['"\\])/g, '\\$1');
}

export function createQueryString(params: Record<string, any>): string {
  const queryString = new URLSearchParams(params).toString();
  return queryString ? `?${queryString}` : '';
}

export function cleanTextForAudio(originalText: string): string {
  let cleanedText = originalText;
  // console.log('originalText', cleanedText);

  // Remove URLs
  cleanedText = cleanedText.replace(/https?:\/\/\S+/g, ' ');

  // Remove special characters but keep letters, numbers, whitespace, commas, and currency symbols
  // cleanedText = cleanedText.replace(/[^\p{L}\p{N}\s,\p{Sc}]/gu, ' ');
  cleanedText = cleanedText.replace(/[-*#<>™®]/g, ' ');

  // Replace <SYS> with an empty string
  cleanedText = cleanedText.replace(/(SYS)/g, ' ');

  // Replace underscores with spaces
  cleanedText = cleanedText.replace(/_/g, ' ');

  // console.log('cleanedText', cleanedText);
  return cleanedText;
}

export function processRagSourceFile(sourcefile: string) {
  const firstHashIndex = sourcefile.indexOf('#');
  const lastHashIndex = sourcefile.lastIndexOf('#');
  const source =
    lastHashIndex !== -1 ? sourcefile.substring(firstHashIndex + 1, lastHashIndex) : sourcefile;
  return source;
}
