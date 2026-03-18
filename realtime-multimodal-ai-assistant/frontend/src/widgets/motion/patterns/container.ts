// ----------------------------------------------------------------------

interface SequenceAnimationConfig {
  childrenDelay?: number;
  initialDelay?: number;
  exitDelay?: number;
}

const BASE_TIMING = 0.1;

/**
 * Creates animation variants for container elements
 * that coordinate child animations with staggering effects
 */
function generateSequentialMotion({
  childrenDelay = BASE_TIMING,
  initialDelay = BASE_TIMING,
  exitDelay = BASE_TIMING,
}: SequenceAnimationConfig = {}): Record<string, any> {
  return {
    animate: {
      transition: {
        staggerChildren: childrenDelay,
        delayChildren: initialDelay,
      },
    },
    exit: {
      transition: {
        staggerChildren: exitDelay,
        staggerDirection: -1,
      },
    },
  };
}

export const varSequence = generateSequentialMotion;
export type Props = SequenceAnimationConfig;
