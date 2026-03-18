import { useMemo } from 'react';
import { useParams as originalUseParams } from 'react-router-dom';

// ----------------------------------------------------------------------

export const useParams = () => {
  const params = originalUseParams();
  return useMemo(() => ({ ...params }), [params]);
};
