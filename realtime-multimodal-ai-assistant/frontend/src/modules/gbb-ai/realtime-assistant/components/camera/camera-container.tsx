import { createPortal } from 'react-dom';
import { useMemo, useState } from 'react';

import FloatingHeader from './camera-container-header';
import PaperLikeBox from './camera-container-paper-box';
import CameraPreviewContent from './camera-container-preview';
import type { CameraContainerProps } from './camera-container-types';
import {
  useMountNode,
  useCameraStream,
  useFloatingLayout,
  useKeyboardShortcuts,
} from './camera-container-hooks';

// ----------------------------------------------------------------------

export default function FloatingCameraContainer({
  videoRef,
  onClose,
  onStreamStarted,
  onStreamStopped,
  isFloating = false,
  onToggleFloating,
  title,
  conversation,
  isListening,
  isSpeaking,
  onSend,
  onCaptureFrame,
  processImageCaption,
  resourceName,
  returnToHomepage,
  onCameraCapture,
  messageCount,
}: CameraContainerProps) {
  const [facingMode, setFacingMode] = useState<'user' | 'environment'>('user');

  const mountNode = useMountNode();

  const { assignVideoRef, isLoading, error } = useCameraStream(
    facingMode,
    videoRef,
    onStreamStarted,
    onStreamStopped,
    onCameraCapture,
    messageCount
  );

  const {
    isDragging,
    floatingPosition,
    floatingSize,
    resetFloatingLayout,
    handlePointerDown,
    handleHeaderPointerDown,
  } = useFloatingLayout(isFloating);

  useKeyboardShortcuts(isFloating, onToggleFloating);

  const previewContent = useMemo(
    () => (
      <CameraPreviewContent
        assignVideoRef={assignVideoRef}
        isLoading={isLoading}
        error={error}
        isFloating={isFloating}
        conversation={conversation}
        isListening={isListening}
        isSpeaking={isSpeaking}
        returnToHomepage={returnToHomepage}
      />
    ),
    [
      assignVideoRef,
      error,
      isLoading,
      conversation,
      isFloating,
      isListening,
      isSpeaking,
      returnToHomepage,
    ]
  );

  const panelContent = useMemo(
    () => (
      <PaperLikeBox
        sx={
          isFloating
            ? { position: 'fixed', zIndex: 1400, ...floatingPosition, ...floatingSize }
            : {}
        }
        isFloating={isFloating}
        size={floatingSize}
        onPointerDown={handlePointerDown}
        header={
          <FloatingHeader
            title={title}
            isFloating={isFloating}
            isDragging={isDragging}
            onPointerDown={handleHeaderPointerDown}
            onDoubleClick={isFloating ? resetFloatingLayout : undefined}
            onClose={onClose}
            onToggleFloating={onToggleFloating}
            onFlipCamera={() => setFacingMode((prev) => (prev === 'user' ? 'environment' : 'user'))}
          />
        }
        content={previewContent}
        handles={isFloating}
      />
    ),
    [
      isFloating,
      floatingPosition,
      floatingSize,
      handlePointerDown,
      title,
      isDragging,
      handleHeaderPointerDown,
      resetFloatingLayout,
      onClose,
      onToggleFloating,
      previewContent,
    ]
  );

  if (isFloating && mountNode) {
    return createPortal(panelContent, mountNode);
  }

  return panelContent;
}

// Re-export types for external use
export type {
  VideoRef,
  ResizeHandle,
  PaperLikeBoxProps,
  FloatingHeaderProps,
  CameraContainerProps,
} from './camera-container-types';
