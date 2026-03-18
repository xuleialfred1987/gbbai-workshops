import { ColorType } from 'src/custom/color-set';

import { Maintainer } from './user';
import { IChatMessage, IChatParticipant } from './chat';

// ----------------------------------------------------------------------

export const AGENT_CATEGORIES = [
  'Image',
  'Productivity',
  'Programming',
  'Writing',
  'Lifestyle',
  'Education',
];

// ----------------------------------------------------------------------

export type DataView = 'lastWeek' | 'lastMonth' | 'lastQuater' | 'lastYear' | 'custom';

export type IAgentFilterValue = string | string[] | Date | null;

export type IAgentFilters = {
  name: string;
  categories: string[];
  startDate: Date | null;
  endDate: Date | null;
};

// ----------------------------------------------------------------------

export type AgentFilterStruct = {
  scenario: string[];
  category: string;
  colors: string[];
  priceRange: string;
  rating: string;
};

export type AgentFilterValueStruct = string | string[] | number | number[];

export type MultiAgentFramework = 'autogen' | 'none';

export type IAgentFleet = {
  id: string;
  type: string;
  unreadCount: number;
  messages: IChatMessage[];
  participants: IChatParticipant[];
  // for non-autogen mode
  framework?: MultiAgentFramework;
  planner?: string;
};

export type IAgent = {
  id: string;
  name: string;
  status: 'draft' | 'published';
  category: string;
  avatarUrl: string;
  coverType: string;
  scenarios: { title: string; color: ColorType }[];
  instruction: string;
  description: string;
  dateCreated: Date | number | string;
  dateModified: Date | number | string;
  lastActivity: Date | number | string;
  maintainers: Maintainer[];
  // ---extentions
  function: boolean;
  functionList?: string[];
  knowledge: boolean;
  knowledgeBase?: string;
  assistant: boolean;
  assistantName?: string;
  samplePrompts?: string[];
};

export type ILiteAgent = {
  id: string;
  name: string;
  instruction: string;
  description: string;
  function: boolean;
  functionList?: string[];
  knowledge: boolean;
  knowledgeBase?: string;
};
