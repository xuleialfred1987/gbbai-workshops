import { useRef, useCallback, type Dispatch, type SetStateAction, type MutableRefObject } from 'react';

import type { Message } from 'src/types/chat';

import { parseToolPayload, type ToolCallState } from '../utils/realtime-tool-utils';
import {
  sortRealtimeMessages,
  type ConversationItemPosition,
} from '../utils/realtime-message-order';
import type {
  ExtensionMiddleTierToolStatus,
  ExtensionMiddleTierToolResponse,
} from '../types/realtime-events';

type Props = {
  chatMode: string;
  setMessages: Dispatch<SetStateAction<Message[]>>;
  conversationItemPositionsRef: MutableRefObject<Map<string, ConversationItemPosition>>;
};

export default function useToolActivity({
  chatMode,
  setMessages,
  conversationItemPositionsRef,
}: Props) {
  const latestCompletedToolByNameRef = useRef<Map<string, ToolCallState>>(new Map());

  const upsertToolConversationPosition = useCallback(
    (messageId: string, previousItemId?: string) => {
      const itemPositions = conversationItemPositionsRef.current;
      const currentSequence = itemPositions.get(messageId)?.sequence ?? itemPositions.size;

      itemPositions.set(messageId, {
        previousItemId,
        sequence: currentSequence,
      });
    },
    [conversationItemPositionsRef]
  );

  const upsertToolActivityMessage = useCallback(
    (toolCall: ToolCallState) => {
      const messageId = toolCall.callId
        ? `tool-call-${toolCall.callId}`
        : `tool-call-${toolCall.funcName}`;

      upsertToolConversationPosition(messageId, toolCall.previousItemId);

      setMessages((prev) => {
        const existingIndex = prev.findIndex((msg) => msg.id === messageId);
        const toolMessage = {
          id: messageId,
          body: '(SYS)function',
          contentType: 'text',
          function_calls: [toolCall],
          sources: [],
          createdAt: existingIndex >= 0 ? prev[existingIndex].createdAt : new Date(),
          senderId: 'assistant',
          mode: 'new',
          chatMode,
          realtimeItemId: messageId,
        } as Message;

        if (existingIndex === -1) {
          return sortRealtimeMessages([...prev, toolMessage], conversationItemPositionsRef.current);
        }

        const next = [...prev];
        next[existingIndex] = {
          ...next[existingIndex],
          function_calls: [toolCall],
        } as Message;

        return sortRealtimeMessages(next, conversationItemPositionsRef.current);
      });
    },
    [chatMode, conversationItemPositionsRef, setMessages, upsertToolConversationPosition]
  );

  const removeToolActivityMessage = useCallback(
    (callId?: string, funcName?: string) => {
      let messageId: string | null = null;

      if (callId) {
        messageId = `tool-call-${callId}`;
      } else if (funcName) {
        messageId = `tool-call-${funcName}`;
      }

      if (!messageId) {
        return;
      }

      conversationItemPositionsRef.current.delete(messageId);

      setMessages((prev) => prev.filter((message) => message.id !== messageId));
    },
    [conversationItemPositionsRef, setMessages]
  );

  const mergeKbResultsIntoExistingGroundingMessage = useCallback(
    (kbResults: unknown, internalSearchCallId?: string, internalSearchFuncName?: string) => {
      if (!kbResults) {
        return;
      }

      setMessages((prev) => {
        let groundingIndex = -1;

        for (let index = prev.length - 1; index >= 0; index -= 1) {
          const toolCall = prev[index].function_calls?.[0];

          if (toolCall?.funcName === 'report_grounding' && toolCall.status === 'completed') {
            groundingIndex = index;
            break;
          }
        }

        if (groundingIndex === -1) {
          return prev;
        }

        const groundingMessage = prev[groundingIndex];
        const groundingToolCall = groundingMessage.function_calls?.[0];

        if (!groundingToolCall) {
          return prev;
        }

        const groundingPayload = parseToolPayload(groundingToolCall.results);

        const nextGroundingToolCall: ToolCallState = {
          funcName: groundingToolCall.funcName,
          status: groundingToolCall.status ?? 'completed',
          callId: groundingToolCall.callId,
          previousItemId: groundingToolCall.previousItemId,
          results: {
            ...(groundingPayload ?? {}),
            kb_results: kbResults,
          },
        };

        let internalSearchMessageId: string | null = null;

        if (internalSearchCallId) {
          internalSearchMessageId = `tool-call-${internalSearchCallId}`;
        } else if (internalSearchFuncName) {
          internalSearchMessageId = `tool-call-${internalSearchFuncName}`;
        }

        let next = [...prev];
        next[groundingIndex] = {
          ...groundingMessage,
          function_calls: [nextGroundingToolCall],
        } as Message;

        if (internalSearchMessageId) {
          next = next.filter((message) => message.id !== internalSearchMessageId);
        }

        return sortRealtimeMessages(next, conversationItemPositionsRef.current);
      });
    },
    [conversationItemPositionsRef, setMessages]
  );

  const handleToolStatus = useCallback(
    (message: ExtensionMiddleTierToolStatus) => {
      if (!message?.tool_name) {
        return;
      }

      upsertToolActivityMessage({
        funcName: message.tool_name,
        callId: message.call_id,
        previousItemId: message.previous_item_id,
        status: 'running',
      });
    },
    [upsertToolActivityMessage]
  );

  const handleToolResponse = useCallback(
    (message: ExtensionMiddleTierToolResponse) => {
      if (!message?.tool_name) {
        return;
      }

      let resultPayload = message.tool_result;

      if (message.tool_name === 'report_grounding') {
        const groundingPayload = parseToolPayload(message.tool_result);
        const latestKbSearch = latestCompletedToolByNameRef.current.get('internal_search');
        const kbPayload = latestKbSearch ? parseToolPayload(latestKbSearch.results) : null;

        if (groundingPayload && kbPayload) {
          resultPayload = {
            ...groundingPayload,
            kb_results: kbPayload,
          };
        }

        if (latestKbSearch) {
          removeToolActivityMessage(latestKbSearch.callId, latestKbSearch.funcName);
        }
      }

      const toolCall: ToolCallState = {
        funcName: message.tool_name,
        callId: message.call_id,
        previousItemId: message.previous_item_id,
        status: 'completed',
        results: resultPayload,
      };

      latestCompletedToolByNameRef.current.set(toolCall.funcName, toolCall);
      upsertToolActivityMessage(toolCall);

      if (message.tool_name === 'internal_search') {
        mergeKbResultsIntoExistingGroundingMessage(
          parseToolPayload(toolCall.results),
          toolCall.callId,
          toolCall.funcName
        );
      }
    },
    [mergeKbResultsIntoExistingGroundingMessage, removeToolActivityMessage, upsertToolActivityMessage]
  );

  const clearToolActivity = useCallback(() => {
    latestCompletedToolByNameRef.current.clear();
  }, []);

  return {
    clearToolActivity,
    handleToolResponse,
    handleToolStatus,
  };
}