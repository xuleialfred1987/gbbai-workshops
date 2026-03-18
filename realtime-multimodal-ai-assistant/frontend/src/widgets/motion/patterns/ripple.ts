// project import
import { VariantsType } from '../types';
import { varTranExit, varTranEnter } from './motion-presets';

// ----------------------------------------------------------------------

// Define types for our helper function
interface DirectionalVariantProps {
  axis: 'x' | 'y';
  enterValues: number[];
  exitValues: number[];
  enterScale: {
    axis: 'X' | 'Y';
    values: number[];
  };
  exitScale: {
    axis: 'X' | 'Y';
    values: number[];
  };
}

/**
 * Creates bounce animation variants with customizable properties
 */
const createBounceVariants = (options?: VariantsType) => {
  const {
    durationIn = undefined,
    durationOut = undefined,
    easeIn = undefined,
    easeOut = undefined,
  } = options || {};

  // Shared transition configurations
  const enterTransition = varTranEnter({ durationIn, easeIn });
  const exitTransition = varTranExit({ durationOut, easeOut });

  // Common animation patterns
  const fadeIn = [0, 1, 1, 1, 1, 1];
  const fadeOut = [1, 1, 0];

  // Helper function to generate directional variants with proper typing
  function generateDirectionalVariant({
    axis,
    enterValues,
    exitValues,
    enterScale,
    exitScale,
  }: DirectionalVariantProps) {
    const scaleKey = `scale${enterScale.axis}`;

    return {
      initial: {},
      animate: {
        [axis]: enterValues,
        [scaleKey]: enterScale.values,
        opacity: fadeIn,
        transition: enterTransition,
      },
      exit: {
        [axis]: exitValues,
        [scaleKey]: exitScale.values,
        opacity: fadeOut,
        transition: exitTransition,
      },
    };
  }

  // Entry animations
  const entryVariants = {
    in: {
      initial: {},
      animate: {
        scale: [0.3, 1.1, 0.9, 1.03, 0.97, 1],
        opacity: fadeIn,
        transition: enterTransition,
      },
      exit: {
        scale: [0.9, 1.1, 0.3],
        opacity: fadeOut,
        transition: exitTransition,
      },
    },

    inUp: generateDirectionalVariant({
      axis: 'y',
      enterValues: [720, -24, 12, -4, 0],
      exitValues: [12, -24, 720],
      enterScale: { axis: 'Y', values: [4, 0.9, 0.95, 0.985, 1] },
      exitScale: { axis: 'Y', values: [0.985, 0.9, 3] },
    }),

    inDown: generateDirectionalVariant({
      axis: 'y',
      enterValues: [-720, 24, -12, 4, 0],
      exitValues: [-12, 24, -720],
      enterScale: { axis: 'Y', values: [4, 0.9, 0.95, 0.985, 1] },
      exitScale: { axis: 'Y', values: [0.985, 0.9, 3] },
    }),

    inLeft: generateDirectionalVariant({
      axis: 'x',
      enterValues: [-720, 24, -12, 4, 0],
      exitValues: [0, 24, -720],
      enterScale: { axis: 'X', values: [3, 1, 0.98, 0.995, 1] },
      exitScale: { axis: 'X', values: [1, 0.9, 2] },
    }),

    inRight: generateDirectionalVariant({
      axis: 'x',
      enterValues: [720, -24, 12, -4, 0],
      exitValues: [0, -24, 720],
      enterScale: { axis: 'X', values: [3, 1, 0.98, 0.995, 1] },
      exitScale: { axis: 'X', values: [1, 0.9, 2] },
    }),
  };

  // Exit-only animations
  const exitVariants = {
    out: {
      animate: {
        scale: [0.9, 1.1, 0.3],
        opacity: fadeOut,
      },
    },
    outUp: {
      animate: {
        y: [-12, 24, -720],
        scaleY: [0.985, 0.9, 3],
        opacity: fadeOut,
      },
    },
    outDown: {
      animate: {
        y: [12, -24, 720],
        scaleY: [0.985, 0.9, 3],
        opacity: fadeOut,
      },
    },
    outLeft: {
      animate: {
        x: [0, 24, -720],
        scaleX: [1, 0.9, 2],
        opacity: fadeOut,
      },
    },
    outRight: {
      animate: {
        x: [0, -24, 720],
        scaleX: [1, 0.9, 2],
        opacity: fadeOut,
      },
    },
  };

  return {
    ...entryVariants,
    ...exitVariants,
  };
};

export const varRipple = createBounceVariants;
