import { useRef, useEffect, useCallback } from 'react';

// project imports
import { Message } from 'src/types/chat';

// ----------------------------------------------------------------------

type ScrollRefType = HTMLDivElement;

export default function useMessagesScroll(messages: Message[]) {
  // Create reference for the messages container
  const containerRef = useRef<ScrollRefType>(null);

  // Function to handle scrolling to bottom
  const scrollToBottom = useCallback(() => {
    const element = containerRef.current;

    // Skip if messages or element are missing
    if (!messages || !element) {
      return;
    }

    // Set scroll position to bottom
    element.scrollTop = element.scrollHeight;
  }, [messages]);

  // Perform scroll effect when messages change
  useEffect(() => {
    // Trigger scroll when messages update
    scrollToBottom();

    // Disable eslint warning for dependency array
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [messages]);

  // Return the reference object for external use
  return {
    messagesEndRef: containerRef,
  };
}
