import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import { alpha, useTheme } from '@mui/material/styles';

import { bgBlur } from 'src/custom/css';

import Iconify from 'src/widgets/iconify';

type Props = {
  data: {
    query: string;
    intent_key?: string;
    text?: string;
    score?: number;
    count?: number;
  };
};

export default function FunctionIntentSearchHandler({ data }: Props) {
  const theme = useTheme();
  const isDarkMode = theme.palette.mode === 'dark';

  return (
    <Stack
      spacing={1.25}
      sx={{
        mt: 1.5,
        p: 1.5,
        borderRadius: 1.5,
        border: `1px solid ${alpha(
          isDarkMode ? theme.palette.grey[700] : theme.palette.grey[300],
          0.5
        )}`,
        bgcolor: isDarkMode
          ? alpha(theme.palette.grey[800], 0.6)
          : alpha(theme.palette.grey[100], 0.82),
        ...bgBlur({
          color: isDarkMode ? theme.palette.grey[900] : theme.palette.common.white,
          opacity: isDarkMode ? 0.56 : 0.84,
        }),
      }}
    >
      <Stack direction="row" justifyContent="space-between" alignItems="center" spacing={1}>
        <Stack direction="row" alignItems="center" spacing={1}>
          <Iconify icon="mdi:arrow-decision-outline" width={18} />
          <Typography variant="body2" sx={{ fontWeight: 600 }}>
            Intent detection
          </Typography>
        </Stack>
        <Chip label={data.intent_key || 'unknown'} color="info" size="small" variant="soft" />
      </Stack>

      <Typography variant="caption" color="text.secondary">
        Query: {data.query}
      </Typography>

      {data.text && (
        <Typography variant="body2" sx={{ lineHeight: 1.55 }}>
          Matched example: {data.text}
        </Typography>
      )}

      <Stack direction="row" spacing={1}>
        {typeof data.score === 'number' && (
          <Chip label={`Score ${data.score.toFixed(3)}`} size="small" variant="outlined" />
        )}
        {typeof data.count === 'number' && (
          <Chip
            label={`${data.count} result${data.count === 1 ? '' : 's'}`}
            size="small"
            variant="outlined"
          />
        )}
      </Stack>
    </Stack>
  );
}
