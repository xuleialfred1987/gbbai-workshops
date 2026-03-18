import React, { FC, memo } from 'react';

// mui
import { Box } from '@mui/material';

// ----------------------------------------------------------------------

interface Props extends React.ComponentProps<typeof Box> {}

const ComingSoonImage: FC<Props> = (props) => {
  const { ...rest } = props;

  return (
    <Box
      component="svg"
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      height="100%"
      viewBox="0 0 480 360"
      {...rest}
    >
      <image
        aria-label="Under development illustration"
        href="/assets/drawing/under_dev.svg"
        height="360"
        x="0"
        y="0"
      />
    </Box>
  );
};

export default memo(ComingSoonImage);
