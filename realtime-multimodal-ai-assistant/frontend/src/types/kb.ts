// ----------------------------------------------------------------------

export type IDatasetFilterValue = string | string[] | Date | null;

export type IChunkFilters = {
  id: string;
  content: string;
  pages: string[];
};

export type IChunkFilterValue = string | string[] | null;

export type IDatasetFilters = {
  name: string;
  tags: string[];
  status: string;
  startDate: Date | null;
  endDate: Date | null;
};

export type IQaTableFilters = {
  name: string;
  tags: string[];
  topics: string[];
  statuses: string[];
  startDate: Date | null;
  endDate: Date | null;
};

export type IKbTableFilters = {
  name: string;
  types: string[];
  statuses: string[];
  startDate: Date | null;
  endDate: Date | null;
};

export type IDatasetShared = {
  id: string;
  name: string;
  email: string;
  photo?: string;
  avatarUrl: string;
  permission: string;
};

export type Maintainer = {
  id: string;
  name: string;
  email: string;
  photo?: string;
  avatarUrl: string;
  permission?: string;
};

export type DatasetMetricManager = {
  totalDatasets: number;
  totalItems: number;
  totalTopics: number;
};

export type RagSourceManager = {
  id: string;
  name: string;
  type: string;
  index: string;
  status: string;
};

export type KbItemManager = {
  id: string;
  name: string;
  tags: string[];
  url: string;
  storage: { account: string; container: string };
  shared: Maintainer[];
  createdAt: Date | number | string;
  modifiedAt: Date | number | string;
  // data: QaItemManager[];
  type: string;
  status: string;
  error?: string;
  size: number;
  chunks: number;
  isFavorited: boolean;
  lastEvaluatedKey?: string;
};

export type DatasetManager = {
  id: string;
  title: string;
  tags: string[];
  type?: string;
  count?: number | string;
  description?: string;
  maintainers: Maintainer[];
  dateCreated: Date | number | string;
  dateModified: Date | number | string;
  isDeleting?: boolean;
};

// ----------------------------------------------------------------------

export type IDatasetManager = {
  id: string;
  name: string;
  size: number;
  type: string;
  tags: string[];
  isFavorited: boolean;
  index?: string;
  status: string;
  shared: IDatasetShared[] | null;
  createdAt: Date | number | string;
  modifiedAt: Date | number | string;
};

export type IDataset = IDatasetManager;
