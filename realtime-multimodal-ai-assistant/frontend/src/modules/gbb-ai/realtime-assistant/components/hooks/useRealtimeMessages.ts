import { useRef, useState, useCallback } from 'react';

import type { Message, SendMessage } from 'src/types/chat';

import type { ConversationItemCreated } from '../types/realtime-events';
import {
  sortRealtimeMessages,
  type ConversationItemPosition,
} from '../utils/realtime-message-order';

export default function useRealtimeMessages(chatMode: string) {
  const [messages, setMessages] = useState<Message[]>([]);
  const conversationItemPositionsRef = useRef<Map<string, ConversationItemPosition>>(new Map());

  const handleConversationItemCreated = useCallback((message: ConversationItemCreated) => {
    const newItemId = message?.item?.id;
    const role = message?.item?.role;
    const previousItemId = message?.previous_item_id;

    if (!newItemId || !role || role === 'system') {
      return;
    }

    const itemPositions = conversationItemPositionsRef.current;
    const currentSequence = itemPositions.get(newItemId)?.sequence ?? itemPositions.size;

    itemPositions.set(newItemId, {
      previousItemId,
      sequence: currentSequence,
    });

    setMessages((prev) => sortRealtimeMessages(prev, itemPositions));
  }, []);

  const onSendMessage = useCallback(
    (conversation: SendMessage) => {
      const {
        messageId,
        message,
        contentType,
        sources,
        createdAt,
        senderId,
        mode,
        function_calls,
        realtimeItemId,
      } = conversation;

      const newMessage = {
        id: messageId,
        body: message,
        contentType,
        function_calls,
        sources,
        createdAt,
        senderId,
        mode,
        chatMode,
        realtimeItemId,
      } as Message;

      setMessages((prev) => {
        if (!message) {
          return [...prev];
        }

        const existedMsgIndex = prev.findIndex((msg) => msg.id === messageId);
        if (existedMsgIndex === -1) {
          return sortRealtimeMessages([...prev, newMessage], conversationItemPositionsRef.current);
        }

        const next = [...prev];
        next[existedMsgIndex] = {
          ...next[existedMsgIndex],
          body: next[existedMsgIndex].body + message,
          realtimeItemId: realtimeItemId ?? next[existedMsgIndex].realtimeItemId,
        };

        return sortRealtimeMessages(next, conversationItemPositionsRef.current);
      });
    },
    [chatMode]
  );

  const clearMessages = useCallback(() => {
    setMessages([]);
    conversationItemPositionsRef.current = new Map();
  }, []);

  return {
    messages,
    setMessages,
    onSendMessage,
    clearMessages,
    handleConversationItemCreated,
    conversationItemPositionsRef,
  };
}