// project import
import { VariantsType } from '../types';
import { varTranExit, varTranEnter } from './motion-presets';

// ----------------------------------------------------------------------

type Direction = 'up' | 'down' | 'left' | 'right' | null;
type FadeMode = 'in' | 'out';

/**
 * Generates fade animation variants with directional options
 */
export const varDissolve = (config?: VariantsType) => {
  // Extract configuration with defaults
  const { distance = 120, durationIn, durationOut, easeIn, easeOut } = config || {};

  // Transition configurations
  const enterTransition = varTranEnter({ durationIn, easeIn });
  const exitTransition = varTranExit({ durationOut, easeOut });

  // Helper to generate directional variants
  const createDirectionalVariant = (mode: FadeMode, direction: Direction = null) => {
    // Basic properties based on animation mode
    const isEntering = mode === 'in';
    const baseOpacity = isEntering ? 0 : 1;
    const targetOpacity = isEntering ? 1 : 0;

    // Initial position offset based on direction
    let initialOffset = {};
    let finalOffset = {};

    if (direction) {
      const axis = direction === 'up' || direction === 'down' ? 'y' : 'x';
      const sign = direction === 'down' || direction === 'right' ? 1 : -1;
      const value = distance * (isEntering ? sign : 0);
      const exitValue = distance * (isEntering ? 0 : sign);

      initialOffset = { [axis]: value };
      finalOffset = { [axis]: exitValue };
    }

    return {
      initial: { opacity: baseOpacity, ...initialOffset },
      animate: {
        opacity: targetOpacity,
        ...(direction ? { [direction === 'up' || direction === 'down' ? 'y' : 'x']: 0 } : {}),
        transition: isEntering ? enterTransition : exitTransition,
      },
      exit: {
        opacity: baseOpacity,
        ...finalOffset,
        transition: isEntering ? exitTransition : enterTransition,
      },
    };
  };

  // Generate all variants using the creator function
  return {
    // IN variants
    in: createDirectionalVariant('in'),
    inUp: createDirectionalVariant('in', 'up'),
    inDown: createDirectionalVariant('in', 'down'),
    inLeft: createDirectionalVariant('in', 'left'),
    inRight: createDirectionalVariant('in', 'right'),

    // OUT variants
    out: createDirectionalVariant('out'),
    outUp: createDirectionalVariant('out', 'up'),
    outDown: createDirectionalVariant('out', 'down'),
    outLeft: createDirectionalVariant('out', 'left'),
    outRight: createDirectionalVariant('out', 'right'),
  };
};
