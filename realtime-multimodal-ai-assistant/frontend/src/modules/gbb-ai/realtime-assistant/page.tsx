import { Helmet } from 'react-helmet-async';

import RealtimeAssistant from './entry';

// ----------------------------------------------------------------------

const REALTIME_ASSISTANT_ID = 'realtime-assistant';

export default function RealtimeAssistantPage() {
  return (
    <>
      <Helmet>
        <title>GBB/AI: Realtime Assistant</title>
      </Helmet>

      <RealtimeAssistant id={REALTIME_ASSISTANT_ID} />
    </>
  );
}