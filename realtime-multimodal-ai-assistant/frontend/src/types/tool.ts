// ----------------------------------------------------------------------

export type ITool = {
  id: string;
  name: string;
  size: number;
  url: string;
  cover: string;
  code: string;
  meta: string;
  entryFunction: string;
  dependencies: string[];
  envVars: { key: string; value: string }[];
  apiAuth: { type: string; apiKey: string; authType: string } | null;
  description: string;
  params: { name: string; type: string; value: string; required?: boolean }[];
  type: string;
  tags: string[];
  isFavorited: boolean;
  status: string;
  shared: IToolMaintainer[] | null;
  createdAt: Date | number | string;
  modifiedAt: Date | number | string;
  response: string;
};

export type ILiteTool = {
  id: string;
  name: string;
  code: string;
  meta: string;
  type: string;
  params: { name: string; type: string; value: string; required?: boolean }[];
  apiAuth: { type: string; apiKey: string; authType: string } | null;
};

export type IToolFilterType = string | string[] | Date | null;

export type IToolFilters = {
  name: string;
  tags: string[];
  statuses: string[];
  startDate: Date | null;
  endDate: Date | null;
};

export type IToolMaintainer = {
  id: string;
  name: string;
  email: string;
  photo?: string;
  avatarUrl: string;
  permission: string;
};
