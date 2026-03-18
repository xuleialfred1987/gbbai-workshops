// ----------------------------------------------------------------------
// Data table configuration interface
// ----------------------------------------------------------------------

/**
 * Core properties for the enhanced data table component
 */
export interface TableProps {
  // Display settings
  dense: boolean;
  page: number;
  rowsPerPage: number;

  // Sorting configuration
  order: 'asc' | 'desc';
  orderBy: string;

  // Selection management
  selected: string[];

  // Row selection handlers
  onSelectRow: (id: string) => void;
  onSelectAllRows: (isChecked: boolean, selectedIds: string[]) => void;

  // Navigation and state handlers
  onResetPage: () => void;
  onSort: (columnId: string) => void;
  onChangePage: (evt: unknown, pageNumber: number) => void;
  onChangeRowsPerPage: (evt: React.ChangeEvent<HTMLInputElement>) => void;
  onChangeDense: (evt: React.ChangeEvent<HTMLInputElement>) => void;

  // Deletion update handlers
  onUpdatePageDeleteRow: (remainingRowsCount: number) => void;
  onUpdatePageDeleteRows: (deletionStats: {
    totalRows: number;
    totalRowsInPage: number;
    totalRowsFiltered: number;
  }) => void;

  // State setters
  setPage: React.Dispatch<React.SetStateAction<number>>;
  setDense: React.Dispatch<React.SetStateAction<boolean>>;
  setOrder: React.Dispatch<React.SetStateAction<'desc' | 'asc'>>;
  setOrderBy: React.Dispatch<React.SetStateAction<string>>;
  setSelected: React.Dispatch<React.SetStateAction<string[]>>;
  setRowsPerPage: React.Dispatch<React.SetStateAction<number>>;
}
