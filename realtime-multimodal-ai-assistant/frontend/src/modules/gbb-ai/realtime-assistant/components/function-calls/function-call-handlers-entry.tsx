import FunctionTeamsHandler from './function-call-handlers-teams';
import FunctionBookCsHandler from './function-call-handlers-book-cs';
import FunctionToolStatusHandler from './function-call-handlers-status';
import FunctionKBSearchHandler from './function-call-handlers-kb-search';
import FunctionGroundingHandler from './function-call-handlers-grounding';
import FunctionAddToCartHandler from './function-call-handlers-add-to-cart';
import FunctionBingSearchHandler from './function-call-handlers-bing-search';
import FunctionPhoneStoreHandler from './function-call-handlers-search-store';
import FunctionIntentSearchHandler from './function-call-handlers-intent-search';
import FunctionLiveAgentTransferHandler from './function-call-handlers-live-agent-transfer';

// ----------------------------------------------------------------------

type Props = {
  function_calls: {
    funcName: string;
    results?: any;
    status?: 'running' | 'completed' | 'error';
    callId?: string;
  }[];
};

type TeamsResultData = {
  status: 'success' | 'error';
  message?: string;
  title?: string;
  has_image?: boolean;
  has_chart?: boolean;
  error_code?: number;
  error_message?: string;
};

// Helper function to safely parse results (handles both string and object)
function parseResults(results: any) {
  if (typeof results === 'string') {
    try {
      return JSON.parse(results);
    } catch (e) {
      console.error('Failed to parse results:', e);
      return null;
    }
  }
  return results;
}

function normalizeTeamsPayload(parsedData: any): {
  result: TeamsResultData;
  originalMessage?: string;
  title?: string;
  imageUrl?: string;
} | null {
  if (!parsedData || typeof parsedData !== 'object') {
    return null;
  }

  const nestedResult =
    parsedData.result && typeof parsedData.result === 'object' ? parsedData.result : undefined;
  const baseResult = nestedResult || parsedData;

  const status: 'success' | 'error' =
    baseResult.status === 'error' || baseResult.error_code || baseResult.error_message
      ? 'error'
      : 'success';

  return {
    result: {
      status,
      message: baseResult.message,
      title: baseResult.title ?? parsedData.title,
      has_image: Boolean(baseResult.has_image),
      has_chart: Boolean(baseResult.has_chart),
      error_code: baseResult.error_code,
      error_message: baseResult.error_message,
    },
    originalMessage: parsedData.original_message ?? parsedData.originalMessage,
    title: parsedData.title ?? baseResult.title,
    imageUrl: parsedData.imageUrl ?? parsedData.image_url ?? baseResult.imageUrl ?? baseResult.image_url,
  };
}

export default function ChatMessageItemFuncHandler({ function_calls }: Props) {
  if (!function_calls || function_calls.length === 0) return null;

  return (
    <>
      {function_calls.map((functionCall, index) => {
        const { funcName, results, status, callId } = functionCall;

        if (status === 'running') {
          return (
            <FunctionToolStatusHandler
              key={callId || `${funcName}-${index}`}
              funcName={funcName}
              status={status}
            />
          );
        }

        if (!results) {
          return null;
        }

        if (funcName === 'intent_search') {
          const data = parseResults(results);
          return data ? (
            <FunctionIntentSearchHandler key={callId || `${funcName}-${index}`} data={data} />
          ) : null;
        }
        if (funcName === 'search_phone_store') {
          const data = parseResults(results);
          return data ? (
            <FunctionPhoneStoreHandler key={callId || `${funcName}-${index}`} data={data} />
          ) : null;
        }
        if (funcName === 'add_to_cart') {
          const data = parseResults(results);
          return data ? (
            <FunctionAddToCartHandler key={callId || `${funcName}-${index}`} data={data} />
          ) : null;
        }
        if (funcName === 'bing_search') {
          const data = parseResults(results);
          return data ? (
            <FunctionBingSearchHandler key={callId || `${funcName}-${index}`} data={data} />
          ) : null;
        }
        if (funcName === 'book_cs_center') {
          const data = parseResults(results);
          return data ? (
            <FunctionBookCsHandler key={callId || `${funcName}-${index}`} data={data} />
          ) : null;
        }
        if (funcName === 'internal_search') {
          const data = parseResults(results);
          return data ? (
            <FunctionKBSearchHandler key={callId || `${funcName}-${index}`} data={data} />
          ) : null;
        }
        if (funcName === 'report_grounding') {
          const data = parseResults(results);
          return data ? (
            <FunctionGroundingHandler key={callId || `${funcName}-${index}`} data={data} />
          ) : null;
        }
        if (funcName === 'send_to_teams') {
          const normalizedPayload = normalizeTeamsPayload(parseResults(results));
          if (!normalizedPayload) return null;

          const { result, originalMessage, title, imageUrl } = normalizedPayload;

          return (
            <FunctionTeamsHandler
              key={callId || `${funcName}-${index}`}
              data={result}
              originalMessage={originalMessage}
              title={title}
              imageUrl={imageUrl}
            />
          );
        }

        if (funcName === 'transfer_to_live_agent') {
          const parsedData = parseResults(results);
          if (!parsedData) return null;

          return (
            <FunctionLiveAgentTransferHandler
              key={callId || `${funcName}-${index}`}
              data={parsedData.result || parsedData}
            />
          );
        }

        return (
          <FunctionToolStatusHandler
            key={callId || `${funcName}-${index}`}
            funcName={funcName}
            status={status || 'completed'}
          />
        );
      })}
    </>
  );
}
