import { useState, useEffect } from 'react';
import { ReadyState } from 'react-use-websocket';

import { REALTIME_API } from 'src/config-global';

import { BACKEND_HEALTH_CHECK_INTERVAL_MS } from '../utils/realtime-tool-utils';

type Props = {
  readyState: ReadyState;
  wsConnecting: boolean;
};

export default function useBackendHealth({ readyState, wsConnecting }: Props) {
  const [isBackendAvailable, setIsBackendAvailable] = useState<boolean | null>(null);

  useEffect(() => {
    let isMounted = true;
    let timeoutId: ReturnType<typeof setTimeout> | null = null;

    const healthUrl = new URL('/test', REALTIME_API).toString();

    const shouldPollBackend =
      readyState !== ReadyState.OPEN &&
      readyState !== ReadyState.CONNECTING &&
      readyState !== ReadyState.CLOSING &&
      !wsConnecting &&
      document.visibilityState === 'visible';

    const checkBackendHealth = async () => {
      try {
        const response = await fetch(healthUrl, {
          method: 'GET',
          cache: 'no-store',
        });

        if (!isMounted) {
          return;
        }

        setIsBackendAvailable(response.ok);
      } catch {
        if (!isMounted) {
          return;
        }

        setIsBackendAvailable(false);
      }
    };

    const scheduleNextCheck = () => {
      if (!isMounted || !shouldPollBackend) {
        return;
      }

      timeoutId = setTimeout(async () => {
        await checkBackendHealth();
        scheduleNextCheck();
      }, BACKEND_HEALTH_CHECK_INTERVAL_MS);
    };

    if (shouldPollBackend) {
      checkBackendHealth();
      scheduleNextCheck();
    }

    return () => {
      isMounted = false;
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, [readyState, wsConnecting]);

  return isBackendAvailable;
}