export const isLocalStorageAccessible = (): boolean => {
  const testKey = '__test_key__';
  try {
    window.localStorage.setItem(testKey, testKey);
    window.localStorage.removeItem(testKey);
    return true;
  } catch (err) {
    return false;
  }
}

export const retrieveFromLocalStorage = (itemKey: string, fallbackValue = ''): string | undefined => {
  const canAccessStorage = isLocalStorageAccessible();
  let retrievedValue;

  if (canAccessStorage) {
    retrievedValue = window.localStorage.getItem(itemKey) || fallbackValue;
  }

  return retrievedValue;
}