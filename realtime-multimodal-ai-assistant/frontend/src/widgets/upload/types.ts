import type { DropzoneOptions } from 'react-dropzone';

// mui
import type { Theme, SxProps } from '@mui/material/styles';

// ----------------------------------------------------------------------

/**
 * Extended File interface with additional properties for upload functionality
 */
export type CustomFile = File & {
  /** Optional file path */
  path?: string;
  /** URL preview representation of the file */
  preview?: string;
  /** Last modification date */
  lastModifiedDate?: Date;
};

/**
 * Properties for upload components that extend react-dropzone options
 */
export interface UploadProps extends DropzoneOptions {
  /** Whether the upload has an error */
  error?: boolean;

  /** Custom styling */
  sx?: SxProps<Theme>;

  /** Whether to show thumbnails for files */
  thumbnail?: boolean;

  /** Placeholder content when empty */
  placeholder?: React.ReactNode;

  /** Helper text to display */
  helperText?: React.ReactNode;

  /** Flag to disable multiple file selection */
  disableMultiple?: boolean;

  // Single file handling
  /** Currently selected file */
  file?: CustomFile | string | null;

  /** Callback when file is deleted */
  onDelete?: VoidFunction;

  // Multiple file handling
  /** Array of currently selected files */
  files?: Array<File | string>;

  /** Callback when an image is clicked */
  onClick?: (img: string) => void;

  /** Callback when files are uploaded */
  onUpload?: VoidFunction;

  /** Callback when a file is removed */
  onRemove?: (file: CustomFile | string) => void;

  /** Callback to remove all files */
  onRemoveAll?: VoidFunction;

  /** Flag to enable image view mode */
  imageView?: boolean;
}
