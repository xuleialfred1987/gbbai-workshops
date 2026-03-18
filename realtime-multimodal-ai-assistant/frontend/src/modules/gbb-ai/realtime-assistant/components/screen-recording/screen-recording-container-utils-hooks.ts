import { useMemo, useEffect } from 'react';

import { SCREEN_FLOATING_ROOT_ID } from './screen-recording-container-types';

// ----------------------------------------------------------------------

export function useMountNode() {
  return useMemo(() => {
    if (typeof window === 'undefined') {
      return null;
    }

    let container = document.getElementById(SCREEN_FLOATING_ROOT_ID);
    if (!container) {
      container = document.createElement('div');
      container.setAttribute('id', SCREEN_FLOATING_ROOT_ID);
      document.body.appendChild(container);
    }

    return container;
  }, []);
}

// ----------------------------------------------------------------------

export function useKeyboardShortcuts(isFloating: boolean, onToggleFloating?: () => void) {
  useEffect(() => {
    if (!isFloating || !onToggleFloating) {
      return undefined;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        onToggleFloating();
      } else if (event.key.toLowerCase() === 'f' && (event.metaKey || event.ctrlKey)) {
        event.preventDefault();
        onToggleFloating();
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => {
      window.removeEventListener('keydown', handleKeyDown);
    };
  }, [isFloating, onToggleFloating]);
}
