import { useState, useEffect, useCallback } from 'react';

// ----------------------------------------------------------------------

// ----------------------------------------------------------------------
// Storage key constants
// ----------------------------------------------------------------------
export const AOAI_CREDENTIAL_KEY = 'gbb-ai-aoai-resources';
export const AOAI_STORAGE_CONFIG = 'gbb-ai-aoai-configurations';
export const AZURE_SPEECH_KEY = 'gbb-ai-speech-key';
export const AZURE_SPEECH_REGION = 'gbb-ai-speech-region';
export const AZURE_SPEECH_CONFIG = 'gbb-ai-speech-config';

// ----------------------------------------------------------------------
// Local storage operations
// ----------------------------------------------------------------------

/**
 * Retrieves data from localStorage with error handling
 */
export const getStorage = (key: string) => {
  try {
    const rawData = window.localStorage.getItem(key);
    return rawData ? JSON.parse(rawData) : null;
  } catch (err) {
    console.error('Error retrieving data from localStorage:', err);
    return null;
  }
};

/**
 * Saves data to localStorage with error handling
 */
export const setStorage = (key: string, value: any): boolean => {
  try {
    const serializedValue = JSON.stringify(value);
    window.localStorage.setItem(key, serializedValue);
    return true;
  } catch (err) {
    console.error('Error saving data to localStorage:', err);
    return false;
  }
};

/**
 * Removes data from localStorage with error handling
 */
export const removeStorage = (key: string): boolean => {
  try {
    window.localStorage.removeItem(key);
    return true;
  } catch (err) {
    console.error('Error removing data from localStorage:', err);
    return false;
  }
};

// ----------------------------------------------------------------------
// Hook for managing localStorage state
// ----------------------------------------------------------------------

/**
 * Custom React hook for local storage integration
 */
export function useLocalStorage(key: string, initialState: any) {
  const [state, setState] = useState(initialState);

  // Load from storage on initial render
  useEffect(() => {
    const storedData = getStorage(key);

    if (storedData) {
      setState((current: any) => ({
        ...current,
        ...storedData,
      }));
    }
  }, [key]);

  // Function to update both state and localStorage
  const updateState = useCallback(
    (newData: any) => {
      setState((current: any) => {
        const updatedData = {
          ...current,
          ...newData,
        };

        setStorage(key, updatedData);
        return updatedData;
      });
    },
    [key]
  );

  // Function to update a specific property
  const update = useCallback(
    (propertyName: string, newValue: any) => {
      updateState({ [propertyName]: newValue });
    },
    [updateState]
  );

  // Function to reset the state to initial and clear storage
  const reset = useCallback(() => {
    removeStorage(key);
    setState(initialState);
  }, [key, initialState]);

  return {
    state,
    update,
    reset,
  };
}
