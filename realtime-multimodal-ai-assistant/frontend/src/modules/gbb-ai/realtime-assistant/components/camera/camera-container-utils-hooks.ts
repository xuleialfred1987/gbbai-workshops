import { useMemo, useEffect } from 'react';

// ----------------------------------------------------------------------

export function useMountNode() {
  return useMemo(() => {
    if (typeof window === 'undefined') {
      return null;
    }
    let container = document.getElementById('camera-floating-root');
    if (!container) {
      container = document.createElement('div');
      container.setAttribute('id', 'camera-floating-root');
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
