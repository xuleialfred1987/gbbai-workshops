// mui
import { Box } from '@mui/material';

// project imports
import Image from '../img-wrap';

// ----------------------------------------------------------------------

interface SingleFilePreviewProps {
  imgUrl?: string;
}

function SingleFilePreview({ imgUrl = '' }: SingleFilePreviewProps) {
  const containerStyles = {
    padding: 1,
    position: 'absolute',
    top: 0,
    left: 0,
    width: '100%',
    height: '100%',
  };

  const imageStyles = {
    width: '100%',
    height: '100%',
    borderRadius: 1,
  };

  return (
    <Box sx={containerStyles}>
      <Image src={imgUrl} alt="file preview" sx={imageStyles} />
    </Box>
  );
}

export default SingleFilePreview;
