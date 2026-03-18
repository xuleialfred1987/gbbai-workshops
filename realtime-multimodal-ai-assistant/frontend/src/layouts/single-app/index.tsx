// mui
import { Box } from '@mui/material';

// project imports
import MainContent from './main';
import HeaderComponent from './header';

// ----------------------------------------------------------------------

interface SingleAppLayoutProps {
  children: React.ReactNode;
}

const SingleAppLayout = ({ children }: SingleAppLayoutProps): JSX.Element => (
  <div className="app-container">
    <HeaderComponent />

    <Box
      component="div"
      sx={{
        display: 'flex',
        minHeight: '100%',
        flexDirection: {
          xs: 'column',
          lg: 'row',
        },
      }}
    >
      <MainContent>{children}</MainContent>
    </Box>
  </div>
);

export default SingleAppLayout;
