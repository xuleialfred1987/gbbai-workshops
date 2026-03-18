import type { CSSProperties } from 'react';

// mui
import { styled } from '@mui/system';
import { Box, useTheme, CircularProgress } from '@mui/material';

// ----------------------------------------------------------------------

// Create a styled component for the loader container
const LoaderContainer = styled(Box)(({ theme }) => ({
  padding: theme.spacing(0, 5),
  display: 'flex',
  width: '100%',
  height: '100%',
  alignItems: 'center',
  justifyContent: 'center',
  flexDirection: 'column',
  minHeight: '100%',
  flexGrow: 1,
}));

type SpinnerProps = {
  customStyles?: CSSProperties | Record<string, unknown>;
  [key: string]: any;
};

/**
 * Component that displays a loading spinner centered in its container
 */
function LoadingDisplay({ customStyles = {}, ...rest }: SpinnerProps) {
  const theme = useTheme();

  return (
    <LoaderContainer style={customStyles} {...rest}>
      <CircularProgress color="primary" thickness={4} size={theme.spacing(6)} />
    </LoaderContainer>
  );
}

export default LoadingDisplay;
