import Box from '@mui/material/Box';
import Card from '@mui/material/Card';
import Chip from '@mui/material/Chip';
import Stack from '@mui/material/Stack';
import Typography from '@mui/material/Typography';
import { alpha, useTheme } from '@mui/material/styles';

import { bgBlur } from 'src/custom/css';

import Iconify from 'src/widgets/iconify';
import MediaPreview from 'src/widgets/media-preview';

type Props = {
  data: {
    status?: string;
    count?: number;
    sources_recorded?: string[];
    kb_results?: {
      query?: string;
      chunks?: {
        id: string;
        content: string;
        reranker_score?: number;
        title?: string;
      }[];
    };
  };
};

const normalizeSourceId = (value: string | number | null | undefined) =>
  String(value ?? '')
    .trim()
    .replace(/^[#[]+/, '')
    .replace(/[\]:]+$/g, '')
    .replace(/^source[-_\s]*/i, '')
    .trim()
    .toLowerCase();

const normalizeDigitsOnly = (value: string | number | null | undefined) =>
  normalizeSourceId(value).replace(/[^0-9a-z]/gi, '');

export default function FunctionGroundingHandler({ data }: Props) {
  const theme = useTheme();
  const isDarkMode = theme.palette.mode === 'dark';
  const kbChunks = data.kb_results?.chunks || [];
  const groundedSourceIds = new Set((data.sources_recorded || []).map((sourceId) => normalizeSourceId(sourceId)));
  const selectedChunks = (data.sources_recorded || []).map((sourceId) => {
    const normalizedSourceId = normalizeSourceId(sourceId);
    const normalizedToken = normalizeDigitsOnly(sourceId);
    const matchedChunk =
      kbChunks.find((chunk) => normalizeSourceId(chunk.id) === normalizedSourceId) ||
      kbChunks.find((chunk) => normalizeDigitsOnly(chunk.id) === normalizedToken) ||
      ((data.sources_recorded?.length || 0) === 1 ? kbChunks[0] : undefined);
    const lines = matchedChunk?.content?.split('\n') || [];
    const heading = lines[0]?.startsWith('###') ? lines[0].replace('###', '').trim() : null;
    const excerpt = matchedChunk
      ? (heading ? lines.slice(1).join('\n').trim() : matchedChunk.content).slice(0, 220)
      : null;

    return {
      id: matchedChunk ? normalizeSourceId(matchedChunk.id) : normalizedSourceId,
      label: matchedChunk?.title || heading || `#${normalizedSourceId}`,
      excerpt,
      rerankerScore: matchedChunk?.reranker_score,
    };
  });
  const unselectedKbChunks = kbChunks.filter(
    (chunk) => !groundedSourceIds.has(normalizeSourceId(chunk.id))
  );

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
          <Iconify icon="solar:document-text-bold" width={18} />
          <Typography variant="body2" sx={{ fontWeight: 600 }}>
            Source grounding
          </Typography>
        </Stack>
        <Chip label={data.status || 'ok'} color="success" size="small" variant="soft" />
      </Stack>

      <Typography variant="caption" color="text.secondary">
        {typeof data.count === 'number'
          ? `${data.count} source${data.count === 1 ? '' : 's'} recorded`
          : 'Sources recorded'}
      </Typography>

      {!!selectedChunks.length && (
        <Stack spacing={1}>
          <Typography variant="caption" color="text.secondary">
            Selected knowledge source{selectedChunks.length === 1 ? '' : 's'}
          </Typography>

          <Stack spacing={1}>
            {selectedChunks.map((chunk) => (
              <Card
                key={chunk.id}
                sx={{
                  px: 1.25,
                  py: 1,
                  borderRadius: 1.25,
                  boxShadow: 'none',
                  border: `1px solid ${alpha(theme.palette.success.main, 0.35)}`,
                  bgcolor: alpha(theme.palette.success.main, isDarkMode ? 0.12 : 0.08),
                }}
              >
                <Stack spacing={0.75}>
                  <Stack direction="row" spacing={0.75} alignItems="center" useFlexGap flexWrap="wrap">
                    <Chip
                      label="Selected"
                      size="small"
                      variant="soft"
                      color="success"
                      sx={{ height: 18, fontSize: 10 }}
                    />
                    <Chip
                      label={`#${chunk.id}`}
                      size="small"
                      variant="soft"
                      color="info"
                      sx={{ height: 18, fontSize: 10 }}
                    />
                    {typeof chunk.rerankerScore === 'number' && (
                      <Chip
                        label={chunk.rerankerScore.toFixed(2)}
                        size="small"
                        variant="soft"
                        color="success"
                        sx={{ height: 18, fontSize: 10 }}
                      />
                    )}
                  </Stack>

                  <Typography
                    variant="subtitle2"
                    color="text.primary"
                    sx={{ fontSize: 13, fontWeight: 600 }}
                  >
                    {chunk.label}
                  </Typography>

                  {chunk.excerpt && (
                    <Typography
                      variant="caption"
                      color="text.secondary"
                      sx={{
                        lineHeight: 1.6,
                        whiteSpace: 'pre-line',
                      }}
                    >
                      {chunk.excerpt}
                    </Typography>
                  )}
                </Stack>
              </Card>
            ))}
          </Stack>
        </Stack>
      )}

      {!!unselectedKbChunks.length && (
        <>
          <Typography variant="caption" color="text.secondary">
            Retrieved knowledge source{unselectedKbChunks.length === 1 ? '' : 's'}
          </Typography>

          <Stack
            spacing={1.5}
            sx={{
              width: '100%',
              overflowX: 'auto',
              '&::-webkit-scrollbar': { display: 'none' },
              msOverflowStyle: 'none',
              scrollbarWidth: 'none',
            }}
          >
            <Stack direction="row" spacing={1.5} sx={{ minWidth: 'min-content' }}>
              {unselectedKbChunks.map((chunk, index) => {
                const { id, content, title, reranker_score } = chunk;
                const isGrounded = groundedSourceIds.has(normalizeSourceId(id));
                const previewLabel = title || `source-${id || index}`;
                const lines = content.split('\n');
                const heading = lines[0].startsWith('###')
                  ? lines[0].replace('###', '').trim()
                  : null;
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
                      border: `1px solid ${alpha(
                        isGrounded ? theme.palette.success.main : theme.palette.grey[400],
                        isGrounded ? 0.45 : 0.2
                      )}`,
                      bgcolor: isGrounded
                        ? alpha(theme.palette.success.main, isDarkMode ? 0.12 : 0.08)
                        : alpha(theme.palette.grey[400], 0.16),
                      color: 'common.white',
                    }}
                  >
                    <Stack spacing={1.25} sx={{ width: '100%' }}>
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
                            file={previewLabel}
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
                            {title || previewLabel}
                          </Typography>
                        </Stack>
                        <Stack direction="row" spacing={0.5} alignItems="center">
                          {isGrounded && (
                            <Chip
                              label="Selected"
                              size="small"
                              variant="soft"
                              color="success"
                              sx={{ height: 18, fontSize: 10 }}
                            />
                          )}
                          <Chip
                            label={`#${id}`}
                            size="small"
                            variant="soft"
                            color="info"
                            sx={{ height: 18, fontSize: 10 }}
                          />
                          {typeof reranker_score === 'number' && (
                            <Chip
                              label={reranker_score.toFixed(2)}
                              size="small"
                              variant="soft"
                              color="success"
                              sx={{ height: 18, fontSize: 10 }}
                            />
                          )}
                        </Stack>
                      </Stack>

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

                      <Box sx={{ maxHeight: 150, overflowY: 'auto', pr: 0.5 }}>
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
        </>
      )}
    </Stack>
  );
}
