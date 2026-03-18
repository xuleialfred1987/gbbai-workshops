import { useState } from 'react';

import Card from '@mui/material/Card';
import Chip from '@mui/material/Chip';
import Alert from '@mui/material/Alert';
import Stack from '@mui/material/Stack';
import Divider from '@mui/material/Divider';
import Collapse from '@mui/material/Collapse';
import ButtonBase from '@mui/material/ButtonBase';
import Typography from '@mui/material/Typography';
import { alpha, useTheme } from '@mui/material/styles';

import { bgBlur } from 'src/custom/css';

import Iconify from 'src/widgets/iconify';

type TranscriptEntry = {
  role: 'user' | 'assistant';
  text: string;
  created_at?: string;
};

type Props = {
  data: {
    status?: 'success' | 'error';
    handoff_id?: string;
    destination?: string;
    reason?: string;
    issue_summary?: string;
    intent_key?: string | null;
    serial_number?: string | null;
    transcript_line_count?: number;
    transcript?: TranscriptEntry[];
    delivery?: {
      mode?: string;
      configured_endpoint?: boolean;
      status_code?: number;
      message?: string;
    };
    error_message?: string;
  };
};

export default function FunctionLiveAgentTransferHandler({ data }: Props) {
  const theme = useTheme();
  const isDarkMode = theme.palette.mode === 'dark';
  const isSuccess = data.status !== 'error';
  const accentColor = isSuccess ? '#0F8CFF' : theme.palette.error.main;
  const transcriptEntries = data.transcript || [];
  const [isTranscriptExpanded, setIsTranscriptExpanded] = useState(false);

  return (
    <Stack spacing={1.25} sx={{ mt: 1.5, width: '100%', maxWidth: 560 }}>
      <Card
        sx={{
          p: 2,
          borderRadius: 2,
          border: `1px solid ${alpha(accentColor, 0.28)}`,
          bgcolor: isDarkMode ? alpha('#081A2B', 0.78) : alpha('#F2F8FF', 0.96),
          color: 'text.primary',
          boxShadow: `0 6px 16px ${alpha(accentColor, 0.05)}`,
          ...bgBlur({
            color: isDarkMode ? '#081A2B' : '#F2F8FF',
            opacity: isDarkMode ? 0.8 : 0.9,
          }),
        }}
      >
        <Stack spacing={1.5}>
          <Stack direction="row" spacing={1.25} alignItems="center" justifyContent="space-between">
            <Stack direction="row" spacing={1.1} alignItems="center">
              <Stack
                alignItems="center"
                justifyContent="center"
                sx={{
                  width: 36,
                  height: 36,
                  borderRadius: '50%',
                  bgcolor: alpha(accentColor, 0.12),
                  color: accentColor,
                }}
              >
                <Iconify icon="solar:headphones-round-bold" width={20} />
              </Stack>
              <Stack spacing={0.25}>
                <Typography variant="subtitle1" sx={{ fontWeight: 700 }}>
                  {isSuccess ? 'Live agent transfer started' : 'Live agent transfer failed'}
                </Typography>
                <Typography variant="caption" color="text.secondary">
                  {data.destination || 'ASUS live agent'}
                </Typography>
              </Stack>
            </Stack>

            <Chip
              label={isSuccess ? 'Escalated' : 'Error'}
              color={isSuccess ? 'info' : 'error'}
              variant="soft"
              size="small"
            />
          </Stack>

          <Stack direction="row" spacing={0.75} useFlexGap flexWrap="wrap">
            {data.handoff_id && (
              <Chip label={`Handoff ${data.handoff_id}`} size="small" variant="outlined" />
            )}
            {data.intent_key && (
              <Chip
                label={`Intent: ${data.intent_key}`}
                size="small"
                variant="outlined"
                color="info"
              />
            )}
            {data.serial_number && (
              <Chip label={`Serial: ${data.serial_number}`} size="small" variant="outlined" />
            )}
            {typeof data.transcript_line_count === 'number' && (
              <Chip
                label={`${data.transcript_line_count} transcript lines`}
                size="small"
                variant="outlined"
              />
            )}
          </Stack>

          {data.issue_summary && (
            <Stack spacing={0.5}>
              <Typography variant="caption" color="text.secondary">
                Case summary
              </Typography>
              <Typography variant="body2" sx={{ fontWeight: 600, lineHeight: 1.6 }}>
                {data.issue_summary}
              </Typography>
            </Stack>
          )}

          {data.reason && (
            <Stack spacing={0.5}>
              <Typography variant="caption" color="text.secondary">
                Transfer reason
              </Typography>
              <Typography variant="body2" color="text.secondary" sx={{ lineHeight: 1.6 }}>
                {data.reason}
              </Typography>
            </Stack>
          )}

          <Divider flexItem sx={{ borderColor: alpha(accentColor, 0.16) }} />

          <Stack spacing={0.9}>
            <ButtonBase
              onClick={() => setIsTranscriptExpanded((prev) => !prev)}
              sx={{
                width: '100%',
                borderRadius: 1.05,
                p: 1,
                justifyContent: 'space-between',
                border: `1px solid ${alpha(accentColor, 0.16)}`,
                bgcolor: alpha(accentColor, isDarkMode ? 0.08 : 0.05),
              }}
            >
              <Typography variant="body2" sx={{ fontWeight: 600 }}>
                {isTranscriptExpanded ? 'Hide transcript snapshot' : 'Show transcript snapshot'}
              </Typography>

              <Stack direction="row" spacing={0.75} alignItems="center">
                {typeof data.transcript_line_count === 'number' && (
                  <Chip
                    label={`${data.transcript_line_count} lines`}
                    size="small"
                    variant="soft"
                    color="info"
                  />
                )}
                <Iconify
                  icon={
                    isTranscriptExpanded ? 'solar:alt-arrow-up-bold' : 'solar:alt-arrow-down-bold'
                  }
                  width={18}
                />
              </Stack>
            </ButtonBase>

            <Collapse in={isTranscriptExpanded} timeout="auto" unmountOnExit>
              <Stack spacing={1} sx={{ pt: 0.9 }}>
                {transcriptEntries.length ? (
                  transcriptEntries.map((entry, index) => {
                    const isUser = entry.role === 'user';

                    return (
                      <Stack
                        key={`${entry.role}-${index}-${entry.text.slice(0, 24)}`}
                        spacing={0.5}
                        sx={{
                          p: 1,
                          borderRadius: 1.25,
                          bgcolor: isUser
                            ? alpha(theme.palette.warning.main, isDarkMode ? 0.12 : 0.08)
                            : alpha(theme.palette.info.main, isDarkMode ? 0.14 : 0.1),
                          border: `1px solid ${alpha(
                            isUser ? theme.palette.warning.main : theme.palette.info.main,
                            0.18
                          )}`,
                        }}
                      >
                        <Stack direction="row" spacing={0.75} alignItems="center">
                          {isUser && (
                            <Iconify
                              icon="solar:user-bold"
                              width={14}
                              sx={{ color: theme.palette.warning.main }}
                            />
                          )}
                          <Typography variant="caption" sx={{ fontWeight: 700 }}>
                            {isUser ? 'Customer' : 'AI assistant'}
                          </Typography>
                        </Stack>
                        <Typography variant="body2" sx={{ fontSize: 13, lineHeight: 1.55 }}>
                          {entry.text}
                        </Typography>
                      </Stack>
                    );
                  })
                ) : (
                  <Typography variant="body2" color="text.secondary">
                    No transcript preview was available for this handoff.
                  </Typography>
                )}
              </Stack>
            </Collapse>
          </Stack>

          {/* {data.delivery?.message && <Alert severity="info">{data.delivery.message}</Alert>} */}
          {data.error_message && <Alert severity="error">{data.error_message}</Alert>}
        </Stack>
      </Card>
    </Stack>
  );
}
