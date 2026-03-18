import Card from '@mui/material/Card';
import Stack from '@mui/material/Stack';
import { Box, Chip } from '@mui/material';
import Typography from '@mui/material/Typography';
import { alpha, useTheme } from '@mui/material/styles';

import MediaPreview from 'src/widgets/media-preview';

// ----------------------------------------------------------------------

type Chunk = {
  id: string;
  content: string;
  score: number;
  reranker_score: number;
  title: string;
};

type Props = {
  data: {
    query: string;
    results_text: string;
    chunks: Chunk[];
    count: number;
  };
};

export default function FunctionKBSearchHandler({ data }: Props) {
  const theme = useTheme();
  const { chunks, count } = data;

  if (!chunks || chunks.length === 0) return null;

  return (
    <Stack spacing={1} sx={{ width: '100%', px: 0.15, mt: 1, pb: 1 }}>
      <Stack direction="row" alignItems="center" spacing={1}>
        <Typography variant="subtitle2" fontSize={13} color="text.secondary">
          Knowledge Base Results
        </Typography>
        <Chip
          variant="soft"
          color="info"
          label={`${count} ${count === 1 ? 'result' : 'results'}`}
          size="small"
          sx={{
            height: 18,
            fontSize: 11,
          }}
        />
      </Stack>
      <Stack
        spacing={1.5}
        sx={{
          width: '100%',
          overflowX: 'auto',
          '&::-webkit-scrollbar': { display: 'none' },
          msOverflowStyle: 'none', // IE and Edge
          scrollbarWidth: 'none', // Firefox
        }}
      >
        <Stack direction="row" spacing={1.5} sx={{ minWidth: 'min-content' }}>
          {chunks.map((chunk, index) => {
            const { id, content, title, reranker_score } = chunk;
            // Extract the main heading from content if it starts with ###
            const lines = content.split('\n');
            const heading = lines[0].startsWith('###') ? lines[0].replace('###', '').trim() : null;
            const bodyContent = heading ? lines.slice(1).join('\n').trim() : content;

            return (
              <Card
                key={id || index}
                sx={{
                  px: 1.5,
                  py: 1.25,
                  minWidth: 280,
                  maxWidth: 320,
                  borderRadius: 1.25,
                  boxShadow: 'none',
                  bgcolor: `${alpha(theme.palette.grey[400], 0.16)}`,
                  color: 'common.white',
                  cursor: 'default',
                  transition: 'all 0.2s ease-in-out',
                  //   '&:hover': {
                  //     bgcolor: `${alpha(theme.palette.grey[300], 0.18)}`,
                  //   },
                }}
              >
                <Stack spacing={1.25} sx={{ width: '100%' }}>
                  {/* Header with title and score */}
                  <Stack
                    direction="row"
                    spacing={1}
                    alignItems="center"
                    justifyContent="space-between"
                  >
                    <Stack
                      direction="row"
                      spacing={0.75}
                      alignItems="center"
                      sx={{ flex: 1, minWidth: 0 }}
                    >
                      <MediaPreview
                        file={title}
                        sx={{
                          width: 14,
                          height: 14,
                          flexShrink: 0,
                        }}
                      />
                      <Typography
                        variant="caption"
                        color="text.primary"
                        fontWeight={600}
                        sx={{
                          fontSize: 11,
                          overflow: 'hidden',
                          textOverflow: 'ellipsis',
                          whiteSpace: 'nowrap',
                          flex: 1,
                        }}
                      >
                        {title}
                      </Typography>
                    </Stack>
                    <Chip
                      label={reranker_score.toFixed(2)}
                      size="small"
                      variant="soft"
                      color="success"
                      sx={{
                        height: 18,
                        fontSize: 10,
                        flexShrink: 0,
                      }}
                    />
                  </Stack>

                  {/* Main heading if exists */}
                  {heading && (
                    <Typography
                      variant="subtitle2"
                      color="text.primary"
                      sx={{
                        fontSize: 13,
                        fontWeight: 600,
                        display: '-webkit-box',
                        overflow: 'hidden',
                        textOverflow: 'ellipsis',
                        WebkitBoxOrient: 'vertical',
                        WebkitLineClamp: 2,
                      }}
                    >
                      {heading}
                    </Typography>
                  )}

                  {/* Content */}
                  <Box
                    sx={{
                      maxHeight: 150,
                      overflowY: 'auto',
                      pr: 0.5,
                    }}
                  >
                    <Typography
                      variant="body2"
                      color="text.secondary"
                      sx={{
                        fontSize: 12,
                        lineHeight: 1.6,
                        whiteSpace: 'pre-line',
                      }}
                    >
                      {bodyContent}
                    </Typography>
                  </Box>
                </Stack>
              </Card>
            );
          })}
        </Stack>
      </Stack>
    </Stack>
  );
}
