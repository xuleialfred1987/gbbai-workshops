import { m } from 'framer-motion';
import { Helmet } from 'react-helmet-async';

import Button from '@mui/material/Button';
import Typography from '@mui/material/Typography';

import { NavigationLink } from 'src/routes/components';

import { NotFoundImage } from 'src/media';

import { varRipple, MotionContainer } from 'src/widgets/motion';

// ----------------------------------------------------------------------

export default function NotFoundPage() {
  return (
    <>
      <Helmet>
        <title> 404 Page Not Found!</title>
      </Helmet>

      <MotionContainer>
        <m.div variants={varRipple().in}>
          <Typography variant="h4" sx={{ mb: 3 }}>
            Page Not Found!
          </Typography>
        </m.div>

        <m.div variants={varRipple().in}>
          <NotFoundImage
            sx={{
              height: 240,
              my: { xs: 4, sm: 8 },
            }}
          />
        </m.div>

        <Button size="small" component={NavigationLink} path="/" variant="contained">
          Return to Homepage
        </Button>
      </MotionContainer>
    </>
  );
}
