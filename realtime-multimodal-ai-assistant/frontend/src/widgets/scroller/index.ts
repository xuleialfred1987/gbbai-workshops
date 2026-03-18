import ScrollerComponent from './scroller';
import type { ScrollerProps } from './types';

// ----------------------------------------------------------------------

// Re-export component types
export type { ScrollerProps };

// Export main component with a single consistent name
const Scroller = ScrollerComponent;

export default Scroller;
