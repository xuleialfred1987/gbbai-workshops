import { useRef, useState, useEffect, useCallback } from 'react';

export default function useMessagesScroll(messages: any[]) {
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const [isUserScrolling, setIsUserScrolling] = useState(false);
  const lastMessageCountRef = useRef(0);
  const lastMessageContentRef = useRef('');
  const shouldAutoScrollRef = useRef(true);
  const isAutoScrollingRef = useRef(false);
  const animationFrameRef = useRef<number | null>(null);

  // Check if scrollbar is at bottom (with some tolerance)
  const isAtBottom = useCallback(() => {
    const container = scrollContainerRef.current;
    if (!container) return true;

    const tolerance = 50; // pixels from bottom
    return container.scrollHeight - container.scrollTop - container.clientHeight < tolerance;
  }, []);

  // Cancel any ongoing animation
  const cancelAnimation = useCallback(() => {
    if (animationFrameRef.current) {
      cancelAnimationFrame(animationFrameRef.current);
      animationFrameRef.current = null;
    }
  }, []);

  // Smooth scroll animation (only for new messages and manual scroll)
  const smoothScrollTo = useCallback(
    (element: HTMLElement, target: number, duration: number = 300) => {
      cancelAnimation();

      const start = element.scrollTop;
      const change = target - start;
      const startTime = performance.now();

      const animateScroll = (currentTime: number) => {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);

        // Easing function (ease-out-cubic)
        const easeOutCubic = 1 - (1 - progress) ** 3;

        element.scrollTop = start + change * easeOutCubic;

        if (progress < 1) {
          animationFrameRef.current = requestAnimationFrame(animateScroll);
        } else {
          isAutoScrollingRef.current = false;
          animationFrameRef.current = null;
        }
      };

      isAutoScrollingRef.current = true;
      animationFrameRef.current = requestAnimationFrame(animateScroll);
    },
    [cancelAnimation]
  );

  // Instant scroll for streaming
  const instantScrollToBottom = useCallback(() => {
    if (scrollContainerRef.current) {
      const container = scrollContainerRef.current;
      isAutoScrollingRef.current = true;
      container.scrollTop = container.scrollHeight;

      // Use RAF to ensure the flag is reset after the scroll event
      requestAnimationFrame(() => {
        isAutoScrollingRef.current = false;
      });
    }
  }, []);

  // Scroll to bottom function
  const scrollToBottom = useCallback(
    (smooth: boolean = true) => {
      if (smooth && messagesEndRef.current && scrollContainerRef.current) {
        const container = scrollContainerRef.current;
        smoothScrollTo(container, container.scrollHeight, 300);
      } else {
        instantScrollToBottom();
      }
    },
    [smoothScrollTo, instantScrollToBottom]
  );

  // Handle scroll events - detect user scrolling
  const handleScroll = useCallback(() => {
    // Skip if this is auto-scrolling
    if (isAutoScrollingRef.current) return;

    const container = scrollContainerRef.current;
    if (!container) return;

    const atBottom = isAtBottom();

    // Immediately update state when user scrolls
    if (!atBottom) {
      // User scrolled up - immediately disable auto-scroll and cancel animations
      shouldAutoScrollRef.current = false;
      setIsUserScrolling(true);
      cancelAnimation();
    } else {
      // User is at bottom - enable auto-scroll
      shouldAutoScrollRef.current = true;
      setIsUserScrolling(false);
    }
  }, [isAtBottom, cancelAnimation]);

  // Auto-scroll when messages change (new or streaming)
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (!container || messages.length === 0) return;

    // Get the last message content
    const lastMessage = messages[messages.length - 1];
    const currentContent = lastMessage?.body || '';

    // Check if this is a new message or content update
    const isNewMessage = messages.length > lastMessageCountRef.current;
    const isContentUpdate = currentContent !== lastMessageContentRef.current;

    if (isNewMessage) {
      // New message added - enable auto-scroll and scroll to bottom smoothly
      shouldAutoScrollRef.current = true;
      setIsUserScrolling(false);
      // Small delay to ensure DOM is updated
      setTimeout(() => scrollToBottom(true), 50);
    } else if (isContentUpdate && shouldAutoScrollRef.current && !isUserScrolling) {
      // Content streaming - use instant scroll
      instantScrollToBottom();
    }

    // Update refs
    lastMessageCountRef.current = messages.length;
    lastMessageContentRef.current = currentContent;
  }, [messages, scrollToBottom, instantScrollToBottom, isUserScrolling]);

  // Add scroll listener
  useEffect(() => {
    const container = scrollContainerRef.current;
    if (container) {
      container.addEventListener('scroll', handleScroll, { passive: true });

      // Check initial state
      setTimeout(() => {
        handleScroll();
      }, 100);

      return () => {
        container.removeEventListener('scroll', handleScroll);
        cancelAnimation();
      };
    }
    return undefined;
  }, [handleScroll, cancelAnimation]);

  // Manual scroll to bottom function
  const manualScrollToBottom = useCallback(() => {
    shouldAutoScrollRef.current = true;
    setIsUserScrolling(false);
    scrollToBottom(true);
  }, [scrollToBottom]);

  // Cleanup on unmount
  useEffect(() => cancelAnimation, [cancelAnimation]);

  return {
    messagesEndRef,
    scrollContainerRef,
    scrollToBottom: manualScrollToBottom,
    isUserScrolling,
    isAtBottom: isAtBottom(),
  };
}
