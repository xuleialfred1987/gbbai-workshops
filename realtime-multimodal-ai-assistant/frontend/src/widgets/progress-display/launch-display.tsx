import { m as motion } from 'framer-motion';
import { useState, useEffect, useCallback } from 'react';

// mui
import Stack from '@mui/material/Stack';
import Container from '@mui/material/Container';
import { Theme, SxProps, useTheme } from '@mui/material/styles';

// project imports
import Logo from '../logo';

// ----------------------------------------------------------------------

const spinnerVariants = {
  initial: { opacity: 0, scale: 0 },
  animate: {
    opacity: 1,
    scale: 1,
    transition: { duration: 0.5 },
  },
  exit: {
    opacity: 0,
    scale: 0,
    transition: { duration: 0.5 },
  },
};

interface LaunchDisplayProps {
  sx?: SxProps<Theme>;
  [key: string]: any;
}

export default function LaunchDisplay({ sx, ...other }: LaunchDisplayProps) {
  const theme = useTheme();
  const [isReady, setIsReady] = useState(false);

  useEffect(() => {
    setIsReady(true);
  }, []);

  const renderContent = useCallback(
    () => (
      <Stack
        direction="column"
        spacing={4}
        alignItems="center"
        justifyContent="center"
        sx={{ height: 1, width: 1 }}
      >
        <motion.div
          animate={{
            scale: [1.15, 1, 1, 1.15, 1.15],
            opacity: [1, 0.48, 0.48, 1, 1],
          }}
          transition={{
            duration: 2,
            ease: 'easeInOut',
            repeatDelay: 1,
            repeat: Infinity,
          }}
        >
          <Logo disabledLink singleMode sx={{ width: 64, height: 64 }} />
        </motion.div>

        <motion.div initial="initial" animate="animate" exit="exit" variants={spinnerVariants}>
          <Stack direction="row" spacing={2}>
            {[...Array(3)].map((_, index) => (
              <motion.div
                key={index}
                animate={{
                  y: [0, -15, 0],
                  backgroundColor: [
                    theme.palette.primary.main,
                    theme.palette.primary.light,
                    theme.palette.primary.main,
                  ],
                }}
                transition={{
                  duration: 1,
                  repeat: Infinity,
                  repeatType: 'loop',
                  delay: index * 0.2,
                }}
                style={{
                  width: 12,
                  height: 12,
                  borderRadius: '50%',
                  backgroundColor: theme.palette.primary.main,
                }}
              />
            ))}
          </Stack>
        </motion.div>
      </Stack>
    ),
    [theme.palette]
  );

  if (!isReady) return null;

  return (
    <Container
      maxWidth={false}
      component={motion.div}
      sx={{
        right: 0,
        bottom: 0,
        width: 1,
        height: 1,
        position: 'fixed',
        zIndex: 9998,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        bgcolor: (_theme) => _theme.palette.background.default,
        ...sx,
      }}
      {...other}
    >
      {renderContent()}
    </Container>
  );
}
