import React from 'react';
import { Icon } from '@iconify/react';
import ReactPlayer from 'react-player';
import { formatDistanceToNowStrict } from 'date-fns';
import playCircleFilled from '@iconify/icons-ant-design/play-circle-filled';

// mui
import { styled, useTheme } from '@mui/material/styles';
import { Box, Stack, Paper, Avatar, Typography, ListItemText } from '@mui/material';

// project imports
import isJsonString from 'src/utils/json-string';

import { bgBlur } from 'src/custom/css';

import Markdown from 'src/widgets/markdown';

import { Message, Conversation } from 'src/types/chat';

import ChatMessageItemFuncHandler from './function-calls/function-call-handlers-entry';

// ----------------------------------------------------------------------

const InfoStyle = styled(Typography)(({ theme }) => ({
  display: 'flex',
  marginBottom: theme.spacing(0.75),
  color: theme.palette.text.secondary,
}));

const MessageImgStyle = styled('img')(({ theme }) => ({
  width: '100%',
  cursor: 'pointer',
  objectFit: 'cover',
  borderRadius: 6,
  marginTop: theme.spacing(1.5),
  marginBottom: theme.spacing(0),
  [theme.breakpoints.up('md')]: {
    height: 240,
    minWidth: 96,
  },
}));

// ----------------------------------------------------------------------

type ChatMessageItemProps = {
  message: Message;
  conversation: Conversation;
  onOpenLightbox: (value: string) => void;
  isLastMessage: boolean;
  extraFunctionCalls?: Message['function_calls'];
};

function ChatMessageItem({
  message,
  conversation,
  onOpenLightbox,
  isLastMessage,
  extraFunctionCalls = [],
}: ChatMessageItemProps) {
  const theme = useTheme();

  const sender = conversation.participants.find(
    (participant) => participant.id === message.senderId
  );
  const senderDetails = message.senderId.includes('user')
    ? { type: 'me' }
    : { avatar: sender?.avatarUrl, name: sender?.name };

  const isMe = senderDetails.type === 'me';
  const isImage = message.contentType === 'image';

  const matchVid = message.body ? message.body.match(/\/(.*\.mp4)/) : null;
  const videoUrl = matchVid ? `/${matchVid[1]}` : '';

  const timeDistanceToNow = formatDistanceToNowStrict(new Date(message.createdAt), {
    addSuffix: true,
  });
  const isSystemMsg =
    message &&
    message.body &&
    (message.body.startsWith('(SYS)Working') || message.body.startsWith('(SYS)function'));
  const combinedFunctionCalls = [...(message.function_calls || []), ...(extraFunctionCalls || [])];

  function handleVideo(msgText: string) {
    try {
      if (isSystemMsg) return null;

      return (
        <>
          <Box>
            <Markdown children={msgText.replace('<eos>', '')} />
          </Box>
          <Box sx={{ py: 1.5 }}>
            <Paper
              sx={{
                pb: 2,
                pt: 2.9,
                display: 'flex',
                alignItems: 'center',
                borderRadius: '5px',
                backgroundColor: `${theme.palette.grey[900]}`,
              }}
            >
              <ReactPlayer
                url={videoUrl}
                width="100%"
                height="100%"
                playIcon={<Icon icon={playCircleFilled} width={36} height={36} />}
                controls
              />
            </Paper>
            {/* <Box component="img" src={cover} sx={{ borderRadius: 1.5, width: 1 }} /> */}
          </Box>
        </>
      );
    } catch (error) {
      return <Box sx={{ typography: 'body2' }}>{msgText.replace('<eos>', '')}</Box>;
    }
  }

  const handleMessage = (msg: string, _isMe: boolean) => {
    try {
      if (msg.startsWith('(SYS)function')) {
        return <>{!_isMe && <ChatMessageItemFuncHandler function_calls={combinedFunctionCalls} />}</>;
      }

      return (
        <>
          {handleText(msg, _isMe)}
          {!_isMe && <ChatMessageItemFuncHandler function_calls={combinedFunctionCalls} />}
        </>
      );
    } catch (e) {
      return (
        <>
          {msg ? (
            <Box sx={{ typography: 'body2' }}>{msg}</Box>
          ) : (
            handleText('Nothing to display', isMe)
          )}
        </>
      );
    }
  };

  return (
    <Stack sx={{ mb: 3, mt: isMe ? 1 : 3, width: '100%', display: 'flex' }}>
      {!isSystemMsg && !isMe && (
        <Stack direction="row" alignItems="center" sx={{ px: 0.15, mt: -1, mb: 0.5 }}>
          <Avatar
            alt="copilot"
            src="/assets/avatars/avatar_copilot.jpg"
            sx={{ width: 26, height: 26, mr: 1 }}
          />

          <ListItemText
            primary="AI Agent"
            secondary={timeDistanceToNow.startsWith('0 sec') ? 'Just now' : timeDistanceToNow}
            primaryTypographyProps={{
              mb: 0,
              noWrap: true,
              typography: 'body2',
              sx: { display: 'inline-block' },
            }}
            secondaryTypographyProps={{
              typography: 'caption',
              color: 'inherit',
              sx: {
                ml: 1,
                opacity: 0.64,
                fontSize: '11px',
                display: 'inline-block',
                transform: 'translateY(1.5px)',
              },
            }}
            sx={{ display: 'flex', alignItems: 'center' }}
          />
        </Stack>
      )}
      <Box
        sx={{
          width: isMe ? '100' : '100%', // Auto width for user messages
          maxWidth: isMe ? '100%' : '100%', // Limit maximum width
          alignSelf: isMe ? 'flex-end' : 'flex-start', // Right align user messages
          ...(isMe && {
            py: 1.5,
            px: 1.75,
            borderRadius: 1.75,
            color: 'text.primary',
            ...bgBlur({
              color: theme.palette.primary.main,
              opacity: 0.26,
            }),
          }),
        }}
      >
        <Stack alignItems={isMe ? 'flex-end' : 'flex-start'}>
          {isMe && (
            <InfoStyle
              variant="caption"
              sx={{ ...(isMe && { justifyContent: 'flex-end', mt: -0.25 }) }}
            >
              {timeDistanceToNow.startsWith('0 sec') ? 'Just now' : timeDistanceToNow}
            </InfoStyle>
          )}

          <Box sx={{ width: isMe ? 'auto' : '100%', maxWidth: '100%' }}>
            {isImage && (
              <MessageImgStyle
                alt="attachment"
                src={message.body}
                onClick={() => onOpenLightbox(message.body)}
              />
            )}

            {!!videoUrl && <>{handleVideo(message.body)}</>}

            {!isImage && !videoUrl && <>{handleMessage(message.body, isMe)}</>}

            {message.attachments !== undefined && message.attachments.length > 0 && (
              <>
                {message.attachments.map((attachment, index) => (
                  <MessageImgStyle
                    key={index}
                    alt="attachment"
                    src={attachment?.preview || ''}
                    onClick={() => onOpenLightbox(attachment.preview)}
                  />
                ))}
              </>
            )}
          </Box>
        </Stack>
      </Box>
    </Stack>
  );
}

function handleText(msgText: string, isMe: boolean) {
  try {
    return (
      <Box
        sx={{
          px: isMe ? 0 : 0.15,
          wordBreak: 'break-word',
          ...(isMe && { mt: -0.75, mb: -1.25 }),
        }}
      >
        <Markdown
          sx={{
            '& p': {
              fontSize: 15,
              lineHeight: '1.75 !important',
              ...(!isMe && { mb: 0 }),
            },
            '& ul, & ol': {
              paddingLeft: 2.5,
              marginTop: 0.5,
              marginBottom: 0.75,
            },
            '& li': {
              fontSize: 15,
              marginBottom: 0.75,
              lineHeight: '1.75 !important',
            },
            '& li p': { lineHeight: '1.75 !important', margin: 0 },
            '& code': {
              fontSize: 13,
              borderRadius: 0.5,
              mx: 0.25,
              whiteSpace: 'pre-wrap',
            },
            '& pre, & pre > code': { fontSize: 13, p: 0.75 },
          }}
          children={isJsonString(msgText) ? `\`\`\`json\n ${msgText}` : msgText}
        />
        {/* {msgText} */}
      </Box>
    );
  } catch (error) {
    return <Box sx={{ typography: 'body2' }}>{msgText}</Box>;
  }
}

export default React.memo(ChatMessageItem);
