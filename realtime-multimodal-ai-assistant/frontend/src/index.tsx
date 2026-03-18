import { Suspense } from 'react';
import ReactDOM from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { HelmetProvider } from 'react-helmet-async';

import Router from 'src/routes/modules';

import 'src/index.css';
import 'src/locales/i18n';
import ThemeCustomization from 'src/custom';
import { LocaleProvider } from 'src/locales';
import { ModalProvider } from 'src/contexts/modal-context';

import { SettingsContextProvider } from 'src/widgets/settings';
import { MotionDeferred } from 'src/widgets/motion/motion-deferred';
import NotificationProvider from 'src/widgets/notification/notification-provider';
import { SpeechConfigProvider } from 'src/widgets/speech/context/speech-config-context';

// ----------------------------------------------------------------------

const App = () => (
  <HelmetProvider>
    <BrowserRouter>
      <Suspense fallback={<div>Loading...</div>}>
        <LocaleProvider>
          <SettingsContextProvider>
            <SpeechConfigProvider>
              <ThemeCustomization>
                <ModalProvider>
                  <MotionDeferred>
                    <NotificationProvider>
                      <Router />
                    </NotificationProvider>
                  </MotionDeferred>
                </ModalProvider>
              </ThemeCustomization>
            </SpeechConfigProvider>
          </SettingsContextProvider>
        </LocaleProvider>
      </Suspense>
    </BrowserRouter>
  </HelmetProvider>
);

const root = ReactDOM.createRoot(document.getElementById('root') as HTMLElement);
root.render(
  // <StrictMode>
  <App />
  // </StrictMode>
);
