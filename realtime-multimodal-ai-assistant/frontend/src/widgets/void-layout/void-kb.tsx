import Box from '@mui/material/Box';
import { alpha } from '@mui/material/styles';
import Typography from '@mui/material/Typography';
import Stack, { StackProps } from '@mui/material/Stack';

// ----------------------------------------------------------------------

interface VoidKbProps extends StackProps {
  title?: string;
  imgUrl?: string;
  filled?: boolean;
  description?: string;
  action?: React.ReactNode;
  imgWidth?: number;
  sx?: object;
}

const VoidKb = ({
  title,
  imgUrl,
  action,
  filled,
  description,
  imgWidth,
  sx,
  ...other
}: VoidKbProps) => (
  <Stack
    flexGrow={1}
    alignItems="center"
    justifyContent="center"
    sx={{
      px: 3,
      height: '100%',
      ...(filled && {
        borderRadius: 1,
        bgcolor: (theme) => alpha(theme.palette.grey[500], 0.04),
        border: (theme) => `dashed 1px ${alpha(theme.palette.grey[500], 0.08)}`,
      }),
      ...sx,
    }}
    {...other}
  >
    <Box
      component="img"
      alt="empty content"
      src={imgUrl || '/assets/icons/empty/ic_kb.svg'}
      sx={{ width: '100%', maxWidth: imgWidth || 160 }}
    />

    {title && (
      <Typography
        variant="subtitle1"
        component="span"
        sx={{ mt: 0.5, color: 'text.disabled', textAlign: 'center' }}
      >
        {title}
      </Typography>
    )}

    {description && (
      <Typography variant="body2" sx={{ mt: 1, color: 'text.disabled', textAlign: 'center' }}>
        {description}
      </Typography>
    )}

    {action}
  </Stack>
);

export default VoidKb;
