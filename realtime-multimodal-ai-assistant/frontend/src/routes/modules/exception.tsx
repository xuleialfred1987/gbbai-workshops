import { lazy, Suspense } from 'react';
import { Outlet } from 'react-router-dom';

// project imports
import ExceptionLayout from 'src/layouts/exception';

import { LaunchDisplay } from 'src/widgets/progress-display';

// ----------------------------------------------------------------------

/**
 * Loads the 404 Not Found page component on demand
 */
const NotFoundPage = lazy(() => import('src/modules/general/404'));

/**
 * Exception route configuration
 * Contains routes for error pages and exceptional states
 */
const createExceptionRoutes = () => {
  // Define child routes for exceptions
  const errorPages = [
    {
      path: '404',
      element: <NotFoundPage />,
    },
  ];

  // Create the parent route structure with layout
  return [
    {
      element: (
        <ExceptionLayout>
          <Suspense fallback={<LaunchDisplay />}>
            <Outlet />
          </Suspense>
        </ExceptionLayout>
      ),
      children: errorPages,
    },
  ];
};

// Export the configured routes
export const exceptionRoutes = createExceptionRoutes();
