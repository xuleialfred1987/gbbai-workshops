import { useRef, RefObject, useEffect, useLayoutEffect } from 'react';

// ----------------------------------------------------------------------

// Use appropriate effect based on environment
const useEnvSafeEffect = typeof window === 'undefined' ? useEffect : useLayoutEffect;

// Overload for Window events
export function useEventListener<K extends keyof WindowEventMap>(
  eventName: K,
  handler: (event: WindowEventMap[K]) => void,
  element?: undefined,
  options?: boolean | AddEventListenerOptions
): void;

// Overload for HTMLElement events
export function useEventListener<
  K extends keyof HTMLElementEventMap,
  T extends HTMLElement = HTMLDivElement,
>(
  eventName: K,
  handler: (event: HTMLElementEventMap[K]) => void,
  element: RefObject<T>,
  options?: boolean | AddEventListenerOptions
): void;

// Overload for Document events
export function useEventListener<K extends keyof DocumentEventMap>(
  eventName: K,
  handler: (event: DocumentEventMap[K]) => void,
  element: RefObject<Document>,
  options?: boolean | AddEventListenerOptions
): void;

// Implementation
export function useEventListener<
  KW extends keyof WindowEventMap,
  KH extends keyof HTMLElementEventMap,
  T extends HTMLElement | void = void,
>(
  eventName: KW | KH,
  handler: (event: WindowEventMap[KW] | HTMLElementEventMap[KH] | Event) => void,
  element?: RefObject<T>,
  options?: boolean | AddEventListenerOptions
) {
  // Store handler in ref to prevent unnecessary re-renders
  const handlerRef = useRef(handler);

  // Update ref when handler changes
  useEnvSafeEffect(() => {
    handlerRef.current = handler;
  }, [handler]);

  // Setup and cleanup event listener
  useEffect(() => {
    // Determine target (element or window)
    const target: T | Window = element?.current ?? window;

    // Early return if target doesn't support event listeners
    if (!target || typeof target.addEventListener !== 'function') return undefined;

    // Wrapper that calls current handler reference
    const listenerCallback = ((e: any) => handlerRef.current(e)) as EventListener;

    // Attach listener
    target.addEventListener(eventName, listenerCallback, options);

    // Cleanup function
    return () => target.removeEventListener(eventName, listenerCallback);
  }, [eventName, element, options]);
}
