import { ColorType } from 'src/custom/color-set';

import { Maintainer } from './user';

// ----------------------------------------------------------------------

export const GPT_CATEGORIES = [
  'Image',
  'Productivity',
  'Programming',
  'Writing',
  'Lifestyle',
  'Education',
];

// ----------------------------------------------------------------------

export type DataView = 'lastWeek' | 'lastMonth' | 'lastQuater' | 'lastYear' | 'custom';

export type ICustomGptFilterValue = string | string[] | Date | null;

export type ICustomGptFilters = {
  name: string;
  categories: string[];
  startDate: Date | null;
  endDate: Date | null;
};

// ----------------------------------------------------------------------

export type AppFilterStruct = {
  scenario: string[];
  category: string;
  colors: string[];
  priceRange: string;
  rating: string;
};

export type AppFilterValueStruct = string | string[] | number | number[];

export type MlAppStruct = {
  category: string;
  content: string;
  cover: string;
  status?: string;
  dateCreated: string;
  dateModified: string;
  id: string;
  rating: number;
  scenario: string;
  scenarios: { title: string; color: ColorType }[];
  title: string;
  source: 'built-in' | 'custom' | '3rd-party';
  type: string;
  tags: string[];
  maintainers: Maintainer[];
  totalRating: number;
};

export type ICustomGpt = {
  id: string;
  name: string;
  tags: string[];
  status: 'draft' | 'published';
  content: string;
  category: string;
  coverUrl: string;
  instruction: string;
  description: string;
  totalComments: number;
  totalFavorites: number;
  samplePrompts: string[];
  dateCreated: Date | number | string;
  dateModified: Date | number | string;
  favoritePerson: {
    name: string;
    avatarUrl: string;
  }[];
  author: {
    name: string;
    avatarUrl: string;
  };
  maintainers: Maintainer[];
  // ---extentions
  function: boolean;
  functionList?: string[];
  knowledge: boolean;
  knowledgeBase?: string;
  assistant: boolean;
  assistantName?: string;
};
