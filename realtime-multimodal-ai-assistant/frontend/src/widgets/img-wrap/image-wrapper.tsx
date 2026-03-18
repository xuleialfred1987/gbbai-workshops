import { useMemo, forwardRef } from 'react';
import { LazyLoadImage } from 'react-lazy-load-image-component';

// mui
import Box from '@mui/material/Box';
import { SxProps } from '@mui/system';
import { alpha, Theme, useTheme } from '@mui/material/styles';

// project imports
import { calculateRatio } from './utils';
import { ImageBlockProps } from './types';

// ----------------------------------------------------------------------

const DEFAULT_PLACEHOLDER = '/assets/images/placeholder.svg';
const WRAPPER_CLASS = 'component-image-wrapper';
const COMPONENT_CLASS = 'component-image';

const useOverlayStyles = (overlay: ImageBlockProps['overlay']): SxProps<Theme> => {
  const theme = useTheme();

  return overlay
    ? {
        position: 'relative',
        '&:before': {
          content: "''",
          position: 'absolute',
          inset: 0,
          zIndex: 1,
          background: overlay || alpha(theme.palette.grey[900], 0.48),
        },
      }
    : {};
};

const createImageStyles = (ratio: ImageBlockProps['ratio']): SxProps<Theme> => ({
  width: 1,
  height: 1,
  objectFit: 'cover',
  verticalAlign: 'bottom',
  ...(ratio && {
    position: 'absolute',
    top: 0,
    left: 0,
  }),
});

const createWrapperStyles = (ratio: ImageBlockProps['ratio']): SxProps<Theme> => ({
  overflow: 'hidden',
  position: 'relative',
  verticalAlign: 'bottom',
  display: 'inline-block',
  ...(ratio && { width: 1 }),
  [`& span.${WRAPPER_CLASS}`]: {
    width: 1,
    height: 1,
    verticalAlign: 'bottom',
    backgroundSize: 'cover !important',
    ...(ratio && {
      paddingTop: calculateRatio(ratio),
    }),
  },
});

const ImageWrapper = forwardRef<HTMLSpanElement, ImageBlockProps>((props, ref) => {
  const {
    ratio,
    overlay,
    disabledEffect = false,
    alt,
    src,
    effect = 'blur',
    wrapperClassName,
    sx,
    ...lazyLoadProps
  } = props;

  const overlayStyles = useOverlayStyles(overlay);
  const imageStyles = useMemo(() => createImageStyles(ratio), [ratio]);
  const wrapperStyles = useMemo(() => createWrapperStyles(ratio), [ratio]);

  const lazyImageElement = (
    <Box
      component={LazyLoadImage}
      alt={alt}
      src={src}
      effect={disabledEffect ? undefined : effect}
      wrapperClassName={wrapperClassName || WRAPPER_CLASS}
      placeholderSrc={DEFAULT_PLACEHOLDER}
      sx={imageStyles}
      {...lazyLoadProps}
    />
  );

  return (
    <Box
      ref={ref}
      component="span"
      className={COMPONENT_CLASS}
      sx={{
        ...(wrapperStyles as any),
        ...(overlayStyles as any),
        ...(sx as any),
      }}
    >
      {lazyImageElement}
    </Box>
  );
});

export default ImageWrapper;
