/* eslint-disable */
// ----------------------------------------------------------------------

export default function uuidv4() {
  const template = [8, 4, 4, 4, 12]; // UUID segment lengths
  const characters = '0123456789abcdef';

  let result = '';

  // Generate each segment
  template.forEach((length, index) => {
    // Add hyphen between segments except before the first one
    if (index > 0) {
      result += '-';
    }

    // Generate characters for this segment
    for (let i = 0; i < length; i++) {
      // Special handling for version (4) and variant bits
      if (index === 2 && i === 0) {
        result += '4'; // Version 4
      } else if (index === 3 && i === 0) {
        // Variant bits: 10xx (RFC 4122)
        const highBits = Math.floor(Math.random() * 4) + 8; // 8,9,a,b
        result += characters[highBits];
      } else {
        // Regular random hex digit
        const randomIndex = Math.floor(Math.random() * 16);
        result += characters[randomIndex];
      }
    }
  });

  return result;
}
