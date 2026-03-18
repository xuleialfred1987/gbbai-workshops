import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import { alpha, useTheme } from '@mui/material/styles';

import { bgBlur } from 'src/custom/css';

import Iconify from 'src/widgets/iconify';

type Props = {
  funcName: string;
  status?: 'running' | 'completed' | 'error';
  description?: string;
};

const TOOL_LABELS: Record<string, string> = {
  intent_search: 'Intent detection',
  internal_search: 'Knowledge base search',
  report_grounding: 'Source grounding',
  search_phone_store: 'Store search',
  book_cs_center: 'Service booking',
  get_claim_details: 'Claim lookup',
  send_to_teams: 'Teams handoff',
  transfer_to_live_agent: 'Live agent transfer',
};

const STATUS_META = {
  running: { label: 'Running', color: 'warning' as const, icon: 'svg-spinners:3-dots-scale' },
  completed: { label: 'Done', color: 'success' as const, icon: 'solar:check-circle-bold' },
  error: { label: 'Error', color: 'error' as const, icon: 'solar:danger-bold' },
};

export default function FunctionToolStatusHandler({ funcName, status = 'running', description }: Props) {
  const theme = useTheme();
  const isDarkMode = theme.palette.mode === 'dark';
  const meta = STATUS_META[status];

  return (
    <Stack
      direction="row"
      justifyContent="space-between"
      alignItems="center"
      spacing={1.5}
      sx={{
        mt: 1.5,
        p: 1.25,
        px: 1.5,
        borderRadius: 1.5,
        border: `1px solid ${alpha(
          isDarkMode ? theme.palette.grey[700] : theme.palette.grey[300],
          0.5
        )}`,
        bgcolor: isDarkMode ? alpha(theme.palette.grey[800], 0.6) : alpha(theme.palette.grey[100], 0.8),
        ...bgBlur({
          color: isDarkMode ? theme.palette.grey[900] : theme.palette.common.white,
          opacity: isDarkMode ? 0.56 : 0.82,
        }),
      }}
    >
      <Stack direction="row" spacing={1.1} alignItems="center" sx={{ minWidth: 0 }}>
        <Iconify icon={meta.icon} width={18} />
        <Stack spacing={0.25} sx={{ minWidth: 0 }}>
          <Typography variant="body2" sx={{ fontWeight: 600 }}>
            {TOOL_LABELS[funcName] || funcName}
          </Typography>
          {description && (
            <Typography variant="caption" color="text.secondary" sx={{ lineHeight: 1.4 }}>
              {description}
            </Typography>
          )}
        </Stack>
      </Stack>

      <Chip label={meta.label} color={meta.color} size="small" variant="soft" sx={{ height: 22 }} />
    </Stack>
  );
}