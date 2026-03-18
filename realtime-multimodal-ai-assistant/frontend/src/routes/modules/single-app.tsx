import { lazy, Suspense, ReactNode } from 'react';
import { Outlet, Navigate } from 'react-router-dom';

// project imports
import { paths } from 'src/routes/paths';

import SingleAppLayout from 'src/layouts/single-app';

import { LaunchDisplay } from 'src/widgets/progress-display';

// ----------------------------------------------------------------------

// Lazy load components
const RealtimeAssistantPage = lazy(
  () => import('src/modules/gbb-ai/realtime-assistant/page')
);

// Route configuration
const createSingleAppRoute = (children: ReactNode) => ({
  element: (
    <SingleAppLayout>
      <Suspense fallback={<LaunchDisplay />}>{children}</Suspense>
    </SingleAppLayout>
  ),
});

const REALTIME_ASSISTANT_ROUTE = 'realtime-assistant';

// Export routes configuration
export const singleAppRoutes = [
  {
    path: 'gbb-ai',
    ...createSingleAppRoute(<Outlet />),
    children: [
      {
        index: true,
        element: <Navigate to={paths.singleApp.app(REALTIME_ASSISTANT_ROUTE)} replace />,
      },
      {
        path: 'apps',
        children: [
          {
            index: true,
            element: <Navigate to={paths.singleApp.app(REALTIME_ASSISTANT_ROUTE)} replace />,
          },
          {
            path: REALTIME_ASSISTANT_ROUTE,
            element: <RealtimeAssistantPage />,
          },
          {
            path: ':id',
            element: <Navigate to={paths.singleApp.app(REALTIME_ASSISTANT_ROUTE)} replace />,
          },
        ],
      },
    ],
  },
];
