import { Navigate, useRoutes } from 'react-router-dom';

// project imports
import { DEFAULT_PATH } from 'src/config-global';

import { exceptionRoutes } from './exception';
import { singleAppRoutes } from './single-app';

// ----------------------------------------------------------------------

/**
 * Main router component that combines all application routes
 * @returns Router component with configured routes
 */
export default function ApplicationRouter() {
  // Root redirect
  const rootRedirect = {
    path: '/',
    element: <Navigate to={DEFAULT_PATH} replace />,
  };

  // Not found fallback
  const notFoundFallback = {
    path: '*',
    element: <Navigate to="/404" replace />,
  };

  // Combine all route configurations
  const appRoutes = [
    rootRedirect,
    ...exceptionRoutes,
    ...singleAppRoutes,
    notFoundFallback,
  ];

  // Generate routes configuration
  return useRoutes(appRoutes);
}
