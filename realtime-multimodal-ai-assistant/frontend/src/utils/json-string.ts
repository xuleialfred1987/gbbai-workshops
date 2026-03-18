// ----------------------------------------------------------------------

export default function isJsonString(str: string): boolean {
  try {
    const result = JSON.parse(str);
    if (typeof result === 'object' && result !== null) {
      return (
        Array.isArray(result) ||
        (!Array.isArray(result) && !Array.isArray(result) && typeof result === 'object')
      );
    }
    return false;
  } catch (e) {
    return false;
  }
}

export function isJsonObject(item: any): boolean {
  return (
    typeof item === 'object' && // Is an object
    !Array.isArray(item) && // and is not an Array
    item !== null // and is not null
  );
}

export function parseJsonData(data: any) {
  try {
    const keys = Object.keys(data);
    const items = keys.map((key) => {
      const values = Object.values(data[key]);
      if (values.every((value) => typeof value === 'string')) {
        return values as string[];
      }
      if (values.every((value) => typeof value === 'number')) {
        return values as number[];
      }
      throw new Error(`Unexpected data type in key ${key}`);
    }) as any[];
    return { keys, items };
  } catch (e) {
    return undefined;
  }
}

export function checkChartAvailability(data: any) {
  try {
    const keys = Object.keys(data);
    if (keys.length < 2) {
      return false;
    }
    let flag = false;
    keys.map((key) => {
      const values = Object.values(data[key]);
      if (values.every((value) => typeof value === 'string')) {
        return values as string[];
      }
      if (values.every((value) => typeof value === 'number')) {
        flag = true;
        return values as number[];
      }
      throw new Error(`Unexpected data type in key ${key}`);
    }) as any[];
    return flag;
  } catch (e) {
    return false;
  }
}

export function safeJsonParse(str: string) {
  try {
    if (str === null || str === undefined || str === '') {
      return null;
    }
    // Remove unescaped control characters (except for valid \n, \t, etc.)
    // This replaces raw newlines and tabs inside string values with spaces
    // eslint-disable-next-line no-control-regex
    const sanitized = str.replace(/[\u0000-\u001F]+/g, ' ');
    return JSON.parse(sanitized);
  } catch (e) {
    console.error('JSON parse error:', e, str);
    return null;
  }
}
