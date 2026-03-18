// ----------------------------------------------------------------------

/**
 * Calculate number of empty rows needed for pagination
 * @param page Current page number
 * @param rowsPerPage Number of rows per page
 * @param arrayLength Total array length
 * @returns Number of empty rows
 */
export function voidRows(page: number, rowsPerPage: number, arrayLength: number) {
  if (!page) return 0;
  const totalExpectedRows = (1 + page) * rowsPerPage;
  return Math.max(0, totalExpectedRows - arrayLength);
}

/**
 * Helper function to determine sort order between two values
 */
function descendingComparator<T>(a: T, b: T, orderBy: keyof T) {
  // Handle null values first
  if (a[orderBy] === null && b[orderBy] !== null) return 1;
  if (b[orderBy] === null && a[orderBy] !== null) return -1;

  // Compare non-null values
  const valueA = a[orderBy];
  const valueB = b[orderBy];

  // Return sort priority
  if (valueB < valueA) return -1;
  if (valueB > valueA) return 1;

  // Equal values
  return 0;
}

/**
 * Creates a comparator function for sorting table data
 * @param order Sort direction ('asc' or 'desc')
 * @param orderBy Field to sort by
 * @returns Comparator function
 */
export function getComparator<Key extends keyof any>(
  order: 'asc' | 'desc',
  orderBy: Key
): (a: { [key in Key]: number | string }, b: { [key in Key]: number | string }) => number {
  // Create appropriate comparator based on sort direction
  const sortFactor = order === 'desc' ? 1 : -1;

  return (a, b) => sortFactor * descendingComparator(a, b, orderBy);
}
