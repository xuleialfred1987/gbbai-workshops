import { createPortal } from 'react-dom';
import { useMemo, useCallback } from 'react';

import ScreenRecordingHeader from './screen-recording-container-header';
import ScreenRecordingPreview from './screen-recording-container-preview';
import ScreenRecordingPaperBox from './screen-recording-container-paper-box';
import type { ScreenRecordingContainerProps } from './screen-recording-container-types';
import {
  useMountNode,
  useFloatingLayout,
  useScreenRecording,
  useKeyboardShortcuts,
} from './screen-recording-container-hooks';

// ----------------------------------------------------------------------

export default function ScreenRecordingContainer({
  onClose,
  isFloating = false,
  onToggleFloating,
  messageCount,
  ...props
}: ScreenRecordingContainerProps) {
  const mountNode = useMountNode();

  const { selectedOption, selectedStream, error, isLoadingStream, assignVideoRef, stopStream } =
    useScreenRecording({ ...props, messageCount });

  const {
    isDragging,
    floatingPosition,
    floatingSize,
    resetFloatingLayout,
    handlePointerDown,
    handleHeaderPointerDown,
  } = useFloatingLayout(isFloating);

  useKeyboardShortcuts(isFloating, onToggleFloating);

  const handleClosePanel = useCallback(() => {
    stopStream();
    onClose?.();
  }, [onClose, stopStream]);

  const handleToggleFloating = useCallback(() => {
    onToggleFloating?.();
  }, [onToggleFloating]);

  const clampValue = useCallback((value: number, min: number, max: number) => {
    if (Number.isNaN(value)) {
      return min;
    }
    return Math.min(Math.max(value, min), max);
  }, []);

  const previewMinHeight = useMemo(() => {
    const defaultMinHeight = 220;

    if (!isFloating) {
      return defaultMinHeight;
    }

    const verticalPadding = 32; // stack padding
    const headerReserve = 88;
    const footerReserve = error ? 28 : 0;
    const availableHeight = floatingSize.height - verticalPadding - headerReserve - footerReserve;

    return clampValue(availableHeight, 140, defaultMinHeight);
  }, [clampValue, error, floatingSize.height, isFloating]);

  const hasActiveStream = Boolean(selectedStream);

  const previewContent = useMemo(
    () => (
      <ScreenRecordingPreview
        assignVideoRef={assignVideoRef}
        isLoading={isLoadingStream}
        selectedStream={selectedStream}
        selectedOption={selectedOption}
        error={error}
        minHeight={previewMinHeight}
      />
    ),
    [assignVideoRef, error, isLoadingStream, previewMinHeight, selectedOption, selectedStream]
  );

  const headerContent = useMemo(
    () => (
      <ScreenRecordingHeader
        isFloating={isFloating}
        isDragging={isDragging}
        onPointerDown={handleHeaderPointerDown}
        onDoubleClick={isFloating ? resetFloatingLayout : undefined}
        onToggleFloating={onToggleFloating ? handleToggleFloating : undefined}
        onClose={onClose ? handleClosePanel : undefined}
        selectedOption={selectedOption}
        hasActiveStream={hasActiveStream}
      />
    ),
    [
      handleClosePanel,
      handleHeaderPointerDown,
      handleToggleFloating,
      hasActiveStream,
      isDragging,
      isFloating,
      onClose,
      onToggleFloating,
      resetFloatingLayout,
      selectedOption,
    ]
  );

  const panelContent = useMemo(
    () => (
      <ScreenRecordingPaperBox
        isFloating={isFloating}
        position={floatingPosition}
        size={floatingSize}
        header={headerContent}
        content={previewContent}
        handles={isFloating}
        onPointerDown={handlePointerDown}
      />
    ),
    [floatingPosition, floatingSize, handlePointerDown, headerContent, isFloating, previewContent]
  );

  if (isFloating && mountNode) {
    return createPortal(panelContent, mountNode);
  }

  return panelContent;
}

export { createRecordingOptions } from './screen-recording-container-types';
export { SCREEN_HEADER_TITLE as SCREEN_RECORDER_TITLE } from './screen-recording-container-types';

export type {
  RecordingOption,
  RecordingOptionId,
  ScreenRecordingContainerProps,
} from './screen-recording-container-types';
