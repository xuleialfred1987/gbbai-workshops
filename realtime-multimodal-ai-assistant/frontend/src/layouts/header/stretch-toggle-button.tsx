import { m } from 'framer-motion';

import Box from '@mui/material/Box';
import IconButton from '@mui/material/IconButton';

import { varHover } from 'src/widgets/motion';
import { useSettingsContext } from 'src/widgets/settings/context';

// ----------------------------------------------------------------------

const iconUrl = (name: string) => `/assets/icons/setting/${name}.svg`;

export default function StretchToggleButton() {
  // const { themeMode, onToggleMode } = useSettings();

  const settings = useSettingsContext();

  return (
    <IconButton
      component={m.button}
      whileTap="tap"
      whileHover="hover"
      onClick={() => settings.onUpdate('themeStretch', !settings.themeStretch)}
      variants={varHover(1.05)}
      sx={{
        padding: 0,
        width: 40,
        height: 40,
        '& .svg-color': {
          background: (theme) =>
            `linear-gradient(135deg, ${theme.palette.grey[700]} 0%, ${theme.palette.grey[800]} 100%)`,
          // ...(settings.themeLayout === 'vertical' && {
          //   background: (theme) =>
          //     `linear-gradient(135deg, ${theme.palette.primary.light} 0%, ${theme.palette.primary.main} 100%)`,
          // }),
        },
      }}
    >
      <Box
        // color="red"
        component="img"
        src={settings.themeStretch ? iconUrl('ic_stretch_on') : iconUrl('ic_stretch_off')}
        sx={{
          width: 24,
          height: 24,
          color: 'red',
          // flexShrink: 0,
        }}
      />
      {/* {icon('ic_strecth_on')} */}
      {/* <Iconify icon="/assets/icons/setting/ic_strecth_on.svg" width={24} /> */}
      {/* <Icon
        icon={settings.themeStretch ? <Iconify icon="fluent:mail-24-filled" /> : moonFill}
        width={22}
        height={22}
      /> */}
    </IconButton>
  );
}
