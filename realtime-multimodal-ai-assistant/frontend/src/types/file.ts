// ----------------------------------------------------------------------

export type IFileFilterValue = string | string[] | Date | null;

export type IFileFilters = {
  name: string;
  type: string[];
  startDate: Date | null;
  endDate: Date | null;
};

// ----------------------------------------------------------------------

export type IFileShared = {
  id: string;
  name: string;
  email: string;
  photo?: string;
  avatarUrl: string;
  permission: string;
};

export type IFileManager = {
  id: string;
  name: string;
  size: number;
  type: string;
  url: string;
  tags: string[];
  isFavorited: boolean;
  shared: IFileShared[] | null;
  createdAt: Date | number | string;
  modifiedAt: Date | number | string;
};

export type IFile = IFileManager;
