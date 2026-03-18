// project import
import { TranExitType, TranHoverType, TranEnterType } from '../types';

// ----------------------------------------------------------------------

// Default configuration values
const TRANSITIONS = {
  DEFAULT_EASING: [0.43, 0.13, 0.23, 0.96],
  DURATIONS: {
    HOVER: 0.32,
    ENTER: 0.64,
    EXIT: 0.48,
  },
};

/**
 * Transition factory
 * Creates transition configurations for different animation states
 */
class TransitionFactory {
  /**
   * Create a hover transition configuration
   * @param options - Optional hover transition parameters
   * @returns Transition configuration object
   */
  static createHoverTransition(options?: TranHoverType) {
    return {
      duration: options?.duration ?? TRANSITIONS.DURATIONS.HOVER,
      ease: options?.ease ?? TRANSITIONS.DEFAULT_EASING,
    };
  }

  /**
   * Create an enter transition configuration
   * @param options - Optional enter transition parameters
   * @returns Transition configuration object
   */
  static createEnterTransition(options?: TranEnterType) {
    return {
      duration: options?.durationIn ?? TRANSITIONS.DURATIONS.ENTER,
      ease: options?.easeIn ?? TRANSITIONS.DEFAULT_EASING,
    };
  }

  /**
   * Create an exit transition configuration
   * @param options - Optional exit transition parameters
   * @returns Transition configuration object
   */
  static createExitTransition(options?: TranExitType) {
    return {
      duration: options?.durationOut ?? TRANSITIONS.DURATIONS.EXIT,
      ease: options?.easeOut ?? TRANSITIONS.DEFAULT_EASING,
    };
  }
}

// Export transition generators with backward-compatible names
export const varTranHover = TransitionFactory.createHoverTransition;
export const varTranEnter = TransitionFactory.createEnterTransition;
export const varTranExit = TransitionFactory.createExitTransition;

// Export constants for external use
export const TRANSITION_DEFAULTS = { ...TRANSITIONS };
