// mui
import Box from '@mui/material/Box';
import Stack from '@mui/material/Stack';
import Tooltip from '@mui/material/Tooltip';
import { Theme, SxProps } from '@mui/material/styles';

// project imports
import SaveButton from './save-button';
import { fileData, fileThumb, fileFormat } from './utils';

// ----------------------------------------------------------------------

type MediaPreviewProps = {
  file: File | string;
  tooltip?: boolean;
  imageView?: boolean;
  onDownload?: VoidFunction;
  onClick?: (img: string) => void;
  sx?: SxProps<Theme>;
  imgSx?: SxProps<Theme>;
};

export default function MediaPreview({
  file,
  tooltip,
  imageView,
  onDownload,
  onClick,
  sx,
  imgSx,
}: MediaPreviewProps) {
  const fileInfo = fileData(file);
  const { name = '', path = '', preview = '' } = fileInfo;

  const fileType = fileFormat(path || preview);

  const handleClick = () => {
    if (preview && onClick) {
      onClick(preview);
    }
  };

  const imageContent = (
    <Box
      component="img"
      src={preview}
      sx={{
        width: '100%',
        height: '100%',
        objectFit: 'cover',
        ...imgSx,
      }}
      onClick={handleClick}
    />
  );

  const thumbContent = (
    <Box
      component="img"
      src={fileThumb(fileType)}
      sx={{
        width: 28,
        height: 28,
        ...sx,
      }}
    />
  );

  const content = fileType === 'image' && imageView ? imageContent : thumbContent;

  const renderWithTooltip = (
    <Tooltip title={name}>
      <Stack
        component="span"
        alignItems="center"
        justifyContent="center"
        sx={{
          width: 'fit-content',
          height: 'inherit',
        }}
      >
        {content}
        {onDownload && <SaveButton onDownload={onDownload} />}
      </Stack>
    </Tooltip>
  );

  const renderWithoutTooltip = (
    <>
      {content}
      {onDownload && <SaveButton onDownload={onDownload} />}
    </>
  );

  return tooltip ? renderWithTooltip : renderWithoutTooltip;
}
