import { m, MotionProps } from 'framer-motion';

// mui
import { Box, BoxProps } from '@mui/material';

// project import
import { varSequence } from './patterns';

// ----------------------------------------------------------------------

/**
 * Animated container component using Framer Motion
 * @param {Object} props - Component props
 * @param {boolean} props.animated - Whether animation is enabled
 * @param {boolean} props.isActionMode - Controls animation behavior mode
 * @param {ReactNode} props.children - Child components
 * @returns {JSX.Element} - Animated container component
 */
type AnimatedContainerProps = BoxProps &
  MotionProps & {
    animated?: boolean;
    isActionMode?: boolean;
  };

const AnimatedContainer = ({
  animated = true,
  isActionMode = false,
  children,
  ...restProps
}: AnimatedContainerProps): JSX.Element => {
  const sequenceVariants = varSequence();

  // Action mode implementation
  if (isActionMode) {
    return (
      <Box
        component={m.div}
        initial={false}
        animate={animated ? 'animate' : 'exit'}
        variants={sequenceVariants}
        {...restProps}
      >
        {children}
      </Box>
    );
  }

  // Default implementation
  return (
    <Box
      component={m.div}
      initial="initial"
      animate="animate"
      exit="exit"
      variants={sequenceVariants}
      {...restProps}
    >
      {children}
    </Box>
  );
};

export default AnimatedContainer;
