// project imports
import { Message, Participant } from 'src/types/chat';

// ----------------------------------------------------------------------

interface MessageParams {
  message: Message;
  currentUserId: string;
  participants: Participant[];
}

interface MessageResult {
  me: boolean;
  hasImage: boolean;
  senderDetails: {
    type?: 'me';
    avatarUrl?: string;
    firstName?: string;
  };
}

export default function useParseMessage({
  message,
  participants,
  currentUserId,
}: MessageParams): MessageResult {
  const isCurrentUser = message.senderId === currentUserId;
  const messageParticipant = participants.find((p) => p.id === message.senderId);

  const isImageContent = message.contentType === 'image';

  const messageDetails = {
    me: isCurrentUser,
    hasImage: isImageContent,
    senderDetails: extractSenderInfo(isCurrentUser, messageParticipant),
  };

  return messageDetails;
}

function extractSenderInfo(
  isCurrentUser: boolean,
  participant?: Participant
): { type?: 'me'; avatarUrl?: string; firstName?: string } {
  if (isCurrentUser) {
    return { type: 'me' };
  }

  return {
    avatarUrl: participant?.avatarUrl,
    firstName: participant?.name?.split(' ')[0],
  };
}
