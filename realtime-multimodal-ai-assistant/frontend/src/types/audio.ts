// ----------------------------------------------------------------------

export type IProject = {
  id: string;
  name: string;
  color: string;
  isStarred?: boolean;
};

export type IMaintainer = {
  name: string;
  email: string;
};

export type IRegion = {
  page: number;
  bbox: [number, number, number, number];
  pageUrl?: string;
};

export type ITargetElement = {
  keyword: string;
  regions: IRegion[];
};

export type IAudio = {
  id: string;
  folder: string;
  isFile: boolean;
  isStarred: boolean;
  isUnread: boolean;
  name: string;
  sourceText: string;
  sourceLang: string;
  targetText: string;
  targetLang: string;
  status: string;
  createdAt: Date;
  maintainer: IMaintainer;
  url?: string;
};

export type IAudios = {
  byId: Record<string, IAudio>;
  allIds: string[];
};
