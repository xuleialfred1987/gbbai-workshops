import type { Message } from 'src/types/chat';

export type ConversationItemPosition = {
  previousItemId?: string;
  sequence: number;
};

const getMessageTimestamp = (message: Message) => {
  const value = new Date(message.createdAt).getTime();
  return Number.isFinite(value) ? value : 0;
};

const buildConversationOrderIndex = (
  itemPositions: Map<string, ConversationItemPosition>
): Map<string, number> => {
  const childrenByPrevious = new Map<string | undefined, string[]>();
  const roots: string[] = [];
  const orderedItemIds: string[] = [];

  itemPositions.forEach(({ previousItemId, sequence }, itemId) => {
    const shouldBeRoot = !previousItemId || !itemPositions.has(previousItemId);
    if (shouldBeRoot) {
      roots.push(itemId);
      return;
    }

    const siblings = childrenByPrevious.get(previousItemId) ?? [];
    siblings.push(itemId);
    siblings.sort(
      (leftId, rightId) =>
        (itemPositions.get(leftId)?.sequence ?? Number.MAX_SAFE_INTEGER) -
        (itemPositions.get(rightId)?.sequence ?? Number.MAX_SAFE_INTEGER)
    );
    childrenByPrevious.set(previousItemId, siblings);
  });

  roots.sort(
    (leftId, rightId) =>
      (itemPositions.get(leftId)?.sequence ?? Number.MAX_SAFE_INTEGER) -
      (itemPositions.get(rightId)?.sequence ?? Number.MAX_SAFE_INTEGER)
  );

  const appendChain = (itemId: string) => {
    orderedItemIds.push(itemId);
    const children = childrenByPrevious.get(itemId) ?? [];
    children.forEach(appendChain);
  };

  roots.forEach(appendChain);

  itemPositions.forEach((_, itemId) => {
    if (!orderedItemIds.includes(itemId)) {
      orderedItemIds.push(itemId);
    }
  });

  return new Map(orderedItemIds.map((itemId, index) => [itemId, index]));
};

export const sortRealtimeMessages = (
  messages: Message[],
  itemPositions: Map<string, ConversationItemPosition>
) => {
  const orderIndex = buildConversationOrderIndex(itemPositions);

  return [...messages].sort((left, right) => {
    const leftIndex = orderIndex.get(left.realtimeItemId ?? left.id);
    const rightIndex = orderIndex.get(right.realtimeItemId ?? right.id);

    if (leftIndex !== undefined && rightIndex !== undefined && leftIndex !== rightIndex) {
      return leftIndex - rightIndex;
    }

    const timeDiff = getMessageTimestamp(left) - getMessageTimestamp(right);
    if (timeDiff !== 0) {
      return timeDiff;
    }

    return left.id.localeCompare(right.id);
  });
};