import Button from '@mui/material/Button';
import { Theme, SxProps } from '@mui/material/styles';

import { NavigationLink } from 'src/routes/components';

import { DEFAULT_PATH } from 'src/config-global';

// ----------------------------------------------------------------------

type Props = {
  sx?: SxProps<Theme>;
};

export default function LoginButton({ sx }: Props) {
  return (
    <Button
      component={NavigationLink}
      path={DEFAULT_PATH}
      variant="outlined"
      sx={{ mr: 1, ...sx }}
    >
      Login
    </Button>
  );
}
