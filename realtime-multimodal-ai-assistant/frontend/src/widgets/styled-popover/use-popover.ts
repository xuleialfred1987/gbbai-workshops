import { useState } from 'react';
import type { Dispatch, MouseEvent, SetStateAction } from 'react';

// ----------------------------------------------------------------------

/**
 * Interface defining the return type from the popover hook
 */
interface PopoverHandlerResult {
  /** Current anchor element or null when closed */
  open: HTMLElement | null;

  /** Handler to open the popover */
  onOpen: (event: MouseEvent<HTMLElement>) => void;

  /** Handler to close the popover */
  onClose: () => void;

  /** Direct state setter for advanced use cases */
  setOpen: Dispatch<SetStateAction<HTMLElement | null>>;
}

/**
 * Custom hook to manage popover state and handlers
 *
 * @returns Object containing popover state and control functions
 */
export default function usePopover(): PopoverHandlerResult {
  // Initialize anchor element state as null (closed)
  const [anchorEl, setAnchorEl] = useState<HTMLElement | null>(null);

  // Create handlers as inline functions
  const handleOpen = (event: MouseEvent<HTMLElement>): void => {
    setAnchorEl(event.currentTarget);
  };

  const handleClose = (): void => {
    setAnchorEl(null);
  };

  // Return handlers and state
  return {
    open: anchorEl,
    onOpen: handleOpen,
    onClose: handleClose,
    setOpen: setAnchorEl,
  };
}
