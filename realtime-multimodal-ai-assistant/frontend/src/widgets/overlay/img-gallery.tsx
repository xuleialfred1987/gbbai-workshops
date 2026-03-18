import ImageGalleryCore, { useLightboxState } from 'yet-another-react-lightbox';
import {
  Zoom,
  Video,
  Captions,
  Slideshow,
  Fullscreen,
  Thumbnails,
} from 'yet-another-react-lightbox/plugins';

// mui
import { Typography } from '@mui/material';
import Container from '@mui/material/Container';

// project imports
import Icon from '../iconify';
import { ImageGalleryProps } from './types';
import ImageGalleryStyleWrapper from './styles';

// ----------------------------------------------------------------------

const ImageGallery = ({
  slides,
  disabledZoom,
  disabledVideo,
  disabledTotal,
  disabledCaptions,
  disabledSlideshow,
  disabledThumbnails,
  disabledFullscreen,
  onGetCurrentIndex,
  ...restProps
}: ImageGalleryProps) => {
  const imageCount = slides?.length || 0;
  const useFinity = imageCount < 5;

  // Configure which plugins to use based on props
  const activePlugins = configurePlugins({
    disabledZoom,
    disabledVideo,
    disabledCaptions,
    disabledSlideshow,
    disabledThumbnails,
    disabledFullscreen,
  });

  return (
    <>
      <ImageGalleryStyleWrapper />

      <ImageGalleryCore
        slides={slides}
        animation={{ swipe: 240 }}
        carousel={{ finite: useFinity }}
        controller={{ closeOnBackdropClick: true }}
        plugins={activePlugins}
        on={{
          view: ({ index }: { index: number }) => {
            if (onGetCurrentIndex) {
              onGetCurrentIndex(index);
            }
          },
        }}
        toolbar={{
          buttons: [
            <ImageCounter key="image-counter" total={imageCount} disabled={disabledTotal} />,
            'close',
          ],
        }}
        render={{
          iconClose: () => <Icon width={24} icon="carbon:close" />,
          iconZoomIn: () => <Icon width={24} icon="carbon:zoom-in" />,
          iconZoomOut: () => <Icon width={24} icon="carbon:zoom-out" />,
          iconSlideshowPlay: () => <Icon width={24} icon="carbon:play" />,
          iconSlideshowPause: () => <Icon width={24} icon="carbon:pause" />,
          iconPrev: () => <Icon width={32} icon="carbon:chevron-left" />,
          iconNext: () => <Icon width={32} icon="carbon:chevron-right" />,
          iconExitFullscreen: () => <Icon width={24} icon="carbon:center-to-fit" />,
          iconEnterFullscreen: () => <Icon width={24} icon="carbon:fit-to-screen" />,
        }}
        {...restProps}
      />
    </>
  );
};

// ----------------------------------------------------------------------

/**
 * Determines which plugins to include based on provided options
 */
function configurePlugins({
  disabledZoom,
  disabledVideo,
  disabledCaptions,
  disabledSlideshow,
  disabledThumbnails,
  disabledFullscreen,
}: ImageGalleryProps) {
  const allPlugins = [Captions, Fullscreen, Slideshow, Thumbnails, Video, Zoom];

  // Create a filter condition mapping
  const pluginFilters = [
    { condition: disabledThumbnails, plugin: Thumbnails },
    { condition: disabledCaptions, plugin: Captions },
    { condition: disabledFullscreen, plugin: Fullscreen },
    { condition: disabledSlideshow, plugin: Slideshow },
    { condition: disabledZoom, plugin: Zoom },
    { condition: disabledVideo, plugin: Video },
  ];

  // Filter out disabled plugins
  return allPlugins.filter(
    (plugin) => !pluginFilters.some((filter) => filter.condition && filter.plugin === plugin)
  );
}

// ----------------------------------------------------------------------

interface CounterProps {
  total: number;
  disabled?: boolean;
}

/**
 * Displays current image position in gallery
 */
function ImageCounter({ total, disabled }: CounterProps) {
  const { currentIndex } = useLightboxState();

  if (disabled || total <= 0) {
    return null;
  }

  return (
    <Container
      component="span"
      className="yarl__button"
      sx={{
        display: 'inline-flex',
        justifyContent: 'center',
        alignItems: 'center',
      }}
    >
      <Typography variant="body2">
        <strong>{currentIndex + 1}</strong> / {total}
      </Typography>
    </Container>
  );
}

export default ImageGallery;
