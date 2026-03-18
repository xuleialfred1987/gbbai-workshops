// utils/error-handling.ts
/**
 * Utility functions for handling and parsing API errors
 * 
 * Handles various error formats including:
 * - Azure OpenAI API errors with error codes (e.g., RateLimitReached, QuotaExceeded)
 * - Axios HTTP errors 
 * - JSON embedded in error messages
 * - General JavaScript errors
 * 
 * Example Azure OpenAI error:
 * {
 *   "error": {
 *     "code": "RateLimitReached",
 *     "message": "Rate limit of 1 per 0s exceeded for UserConcurrentRequests. Please wait 0 seconds before retrying."
 *   }
 * }
 */

export interface ParsedError {
  code?: string;
  message: string;
  originalError?: any;
}

/**
 * Extracts error code and message from various error formats
 * Handles Azure OpenAI API errors, Axios errors, and general errors
 */
export function parseApiError(error: any): ParsedError {
  // Handle Axios response errors (Azure OpenAI API)
  if (error?.response?.data?.error) {
    const azureError = error.response.data.error;
    return {
      code: azureError.code,
      message: azureError.message || azureError.details || 'Unknown error',
      originalError: error,
    };
  }

  // Handle errors with embedded JSON in message
  if (error?.message) {
    try {
      const messageMatch = error.message.match(/\{.*\}/);
      if (messageMatch) {
        const errorJson = JSON.parse(messageMatch[0]);
        if (errorJson.error) {
          return {
            code: errorJson.error.code,
            message: errorJson.error.message || errorJson.error.details || error.message,
            originalError: error,
          };
        }
      }
    } catch {
      // If parsing fails, fall through to use original message
    }
    
    return {
      message: error.message,
      originalError: error,
    };
  }

  // Handle string errors
  if (typeof error === 'string') {
    return {
      message: error,
      originalError: error,
    };
  }

  // Fallback for unknown error types
  return {
    message: 'Unknown error occurred',
    originalError: error,
  };
}

/**
 * Formats an error for display to users
 * Includes error code if available
 */
export function formatErrorMessage(error: any): string {
  const parsed = parseApiError(error);
  return parsed.code ? `[${parsed.code}] ${parsed.message}` : parsed.message;
}

/**
 * Formats multiple errors for display
 * Used when handling batch operations with multiple failures
 */
export function formatMultipleErrors(
  errors: any[],
  baseMessage: string,
  maxErrorsToShow: number = 3,
  context?: { models?: string[]; outputNumber?: number }
): string {
  if (errors.length === 0) {
    return baseMessage;
  }

  const errorDetails = errors.map((error, index) => {
    const parsed = parseApiError(error);
    let errorText = parsed.code ? `[${parsed.code}] ${parsed.message}` : parsed.message;
    
    // Add context information if available
    if (context?.models && context?.outputNumber) {
      const modelIndex = Math.floor(index / context.outputNumber);
      const callIndex = index % context.outputNumber;
      const modelName = context.models[modelIndex] || `Model ${modelIndex + 1}`;
      errorText = `${modelName} (${callIndex + 1}): ${errorText}`;
    } else {
      errorText = `${index + 1}. ${errorText}`;
    }
    
    return errorText;
  });

  const errorsToShow = errorDetails.slice(0, maxErrorsToShow);
  const additionalErrors = errorDetails.length - errorsToShow.length;

  let detailedMessage = `${baseMessage}\n\nDetails:\n${errorsToShow.join('\n')}`;

  if (additionalErrors > 0) {
    detailedMessage += `\n... and ${additionalErrors} more error${additionalErrors > 1 ? 's' : ''}`;
  }

  return detailedMessage;
}
