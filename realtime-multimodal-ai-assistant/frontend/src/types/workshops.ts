// ----------------------------------------------------------------------

export type IpFilterValue = string;

export type AgentCaseFilters = {
  publish: string;
};

// ----------------------------------------------------------------------

export type IPostHero = {
  title: string;
  coverUrl: string;
  createdAt?: Date;
  author?: {
    name: string;
    avatarUrl: string;
  };
};

export type IPostComment = {
  id: string;
  name: string;
  avatarUrl: string;
  message: string;
  postedAt: Date;
  users: {
    id: string;
    name: string;
    avatarUrl: string;
  }[];
  replyComment: {
    id: string;
    userId: string;
    message: string;
    postedAt: Date;
    tagUser?: string;
  }[];
};

export type AgentItem = {
  name: string;
  description: string;
  instruction: string;
  tools?: { name: string; avatar: string }[];
  kbName?: string;
};

export type WorkshopItem = {
  name: string;
  icon?: string;
  coverUrl: string;
  description: string;
  tags: string[];
  agentList: AgentItem[];
  overview: string;
  githubRepo: string;
  pptLink?: string;
  prompts?: string[];
};

export type AgentCaseItem = {
  id: string;
  title: string;
  tags?: string[];
  publish?: string;
  content?: string;
  coverUrl?: string;
  metaTitle?: string;
  totalViews: number;
  totalShares: number;
  description: string;
  totalComments: number;
  totalFavorites?: number;
  metaKeywords?: string[];
  metaDescription?: string;
  comments?: IPostComment[];
  createdAt: Date;
  favoritePerson?: {
    name: string;
    avatarUrl: string;
  }[];
  author: {
    name: string;
    avatarUrl: string;
  };
};
