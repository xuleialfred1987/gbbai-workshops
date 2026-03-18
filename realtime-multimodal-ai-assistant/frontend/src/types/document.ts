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

export type IDocument = {
  id: string;
  folder: string;
  isImportant: boolean;
  isStarred: boolean;
  isUnread: boolean;
  fileName: string;
  message: string;
  status: string;
  createdAt: Date;
  pageUrls: string[];
  maintainer: IMaintainer;
  targetElements: ITargetElement[];
};

export type IDocuments = {
  byId: Record<string, IDocument>;
  allIds: string[];
};
