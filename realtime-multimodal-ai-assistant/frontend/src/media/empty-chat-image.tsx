import React, { FC, memo } from 'react';

// mui
import { Box } from '@mui/material';

// ----------------------------------------------------------------------

interface Props extends React.ComponentProps<typeof Box> {}

const EmptyChatImage: FC<Props> = (props) => {
  const { ...rest } = props;

  return (
    <Box
      component="svg"
      xmlns="http://www.w3.org/2000/svg"
      width="100%"
      height="100%"
      viewBox="0 0 124 120"
      {...rest}
    >
      <image
        aria-label="Empty chat illustration"
        href="/assets/drawing/empty_chat.svg"
        height="120"
        x="0"
        y="0"
      />
    </Box>
  );
};

export default memo(EmptyChatImage);
