import { useState } from 'react';
import { m } from 'framer-motion';
import { Icon } from '@iconify/react';
import roundFullscreen from '@iconify/icons-ic/round-fullscreen';
import roundFullscreenExit from '@iconify/icons-ic/round-fullscreen-exit';

import { useTheme } from '@mui/material/styles';
import IconButton from '@mui/material/IconButton';

import { varHover } from 'src/widgets/motion';

// ----------------------------------------------------------------------

export default function FullscreenButton() {
  const theme = useTheme();
  const [fullscreen, setFullscreen] = useState(false);

  const toggleFullScreen = () => {
    if (!document.fullscreenElement) {
      document.documentElement.requestFullscreen();
      setFullscreen(true);
    } else if (document.exitFullscreen) {
      document.exitFullscreen();
      setFullscreen(false);
    }
  };

  return (
    <IconButton
      component={m.button}
      whileTap="tap"
      whileHover="hover"
      variants={varHover(1.05)}
      onClick={toggleFullScreen}
      // color={fullscreen ? 'primary' : 'inherit'}
      color="default"
      sx={{
        padding: 0,
        width: 40,
        height: 40,
        color: theme.palette.grey[600],
      }}
    >
      <Icon icon={fullscreen ? roundFullscreenExit : roundFullscreen} width={28} height={28} />
    </IconButton>
  );
}
