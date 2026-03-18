// ----------------------------------------------------------------------

export type IDashboardFilterValue = string | string[] | Date | null;

export type IDashboardFilters = {
  name: string;
  tags: string[];
  statuses: string[];
  startDate: Date | null;
  endDate: Date | null;
};

export type IDashboardShared = {
  id: string;
  name: string;
  email: string;
  avatarUrl: string;
  permission: string;
};

// ----------------------------------------------------------------------

export type IDashboardManager = {
  id: string;
  name: string;
  description: string;
  status: string;
  createdAt: Date | number | string;
  modifiedAt: Date | number | string;
  activity: number[];
};

export type IDashboard = IDashboardManager;
