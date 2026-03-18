import { useState, useCallback } from 'react';

// project imports
import { TableProps } from './types';

// ----------------------------------------------------------------------

interface TableHookOptions {
  defaultDense?: boolean;
  defaultOrder?: 'asc' | 'desc';
  defaultOrderBy?: string;
  defaultSelected?: string[];
  defaultRowsPerPage?: number;
  defaultCurrentPage?: number;
}

/**
 * Custom hook for managing table state and operations
 */
export default function useTable(props?: TableHookOptions): TableProps {
  // Initialize state with provided defaults or fallback values
  const [isCompact, setIsCompact] = useState<boolean>(Boolean(props?.defaultDense));
  const [currentPage, setCurrentPage] = useState<number>(props?.defaultCurrentPage || 0);
  const [sortField, setSortField] = useState<string>(props?.defaultOrderBy || 'name');
  const [itemsPerPage, setItemsPerPage] = useState<number>(props?.defaultRowsPerPage || 5);
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>(props?.defaultOrder || 'asc');
  const [checkedItems, setCheckedItems] = useState<string[]>(props?.defaultSelected || []);

  // Handle sorting column click
  const handleSort = useCallback(
    (columnId: string) => {
      if (columnId === '') return;

      const shouldReverseOrder = sortField === columnId && sortDirection === 'asc';
      setSortDirection(shouldReverseOrder ? 'desc' : 'asc');
      setSortField(columnId);
    },
    [sortField, sortDirection]
  );

  // Toggle selection of a single row
  const handleSelectRow = useCallback(
    (id: string) => {
      const isAlreadySelected = checkedItems.includes(id);
      const updatedSelection = isAlreadySelected
        ? checkedItems.filter((item) => item !== id)
        : [...checkedItems, id];

      setCheckedItems(updatedSelection);
    },
    [checkedItems]
  );

  // Handle rows per page change
  const handleRowsPerPageChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    setCurrentPage(0);
    setItemsPerPage(parseInt(event.target.value, 10));
  }, []);

  // Toggle dense/compact mode
  const handleDensityChange = useCallback((event: React.ChangeEvent<HTMLInputElement>) => {
    setIsCompact(event.target.checked);
  }, []);

  // Handle select/deselect all rows
  const handleBulkSelection = useCallback((selectAll: boolean, availableIds: string[]) => {
    setCheckedItems(selectAll ? availableIds : []);
  }, []);

  // Handle page change
  const handlePageChange = useCallback((event: unknown, newPage: number) => {
    setCurrentPage(newPage);
  }, []);

  // Reset to first page
  const resetToFirstPage = useCallback(() => {
    setCurrentPage(0);
  }, []);

  // Update page after deleting a row
  const handlePageUpdateAfterDelete = useCallback(
    (itemsInCurrentPage: number) => {
      setCheckedItems([]);

      if (currentPage > 0 && itemsInCurrentPage < 2) {
        setCurrentPage(currentPage - 1);
      }
    },
    [currentPage]
  );

  // Update page after bulk deletion
  const handlePageUpdateAfterBulkDelete = useCallback(
    (params: { totalRows: number; totalRowsInPage: number; totalRowsFiltered: number }) => {
      const { totalRows, totalRowsInPage, totalRowsFiltered } = params;
      const selectedCount = checkedItems.length;

      setCheckedItems([]);

      if (currentPage === 0) return;

      if (selectedCount === totalRowsInPage) {
        setCurrentPage(currentPage - 1);
      } else if (selectedCount === totalRowsFiltered) {
        setCurrentPage(0);
      } else if (selectedCount > totalRowsInPage) {
        const calculatedPage = Math.ceil((totalRows - selectedCount) / itemsPerPage) - 1;
        setCurrentPage(calculatedPage);
      }
    },
    [currentPage, itemsPerPage, checkedItems.length]
  );

  // Return all table state and handlers
  return {
    dense: isCompact,
    order: sortDirection,
    page: currentPage,
    orderBy: sortField,
    rowsPerPage: itemsPerPage,
    //
    selected: checkedItems,
    onSelectRow: handleSelectRow,
    onSelectAllRows: handleBulkSelection,
    //
    onSort: handleSort,
    onChangePage: handlePageChange,
    onChangeDense: handleDensityChange,
    onResetPage: resetToFirstPage,
    onChangeRowsPerPage: handleRowsPerPageChange,
    onUpdatePageDeleteRow: handlePageUpdateAfterDelete,
    onUpdatePageDeleteRows: handlePageUpdateAfterBulkDelete,
    //
    setPage: setCurrentPage,
    setDense: setIsCompact,
    setOrder: setSortDirection,
    setOrderBy: setSortField,
    setSelected: setCheckedItems,
    setRowsPerPage: setItemsPerPage,
  };
}
