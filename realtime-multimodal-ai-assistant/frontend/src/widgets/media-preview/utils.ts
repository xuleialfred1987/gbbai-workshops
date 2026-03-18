import { EnhancedFile } from './types';

// ----------------------------------------------------------------------

const FORMATS = {
  md: ['md'],
  db: ['mysql', 'sql', 'postgresql'],
  pdf: ['pdf'],
  txt: ['txt'],
  web: ['https', 'http'],
  word: ['doc', 'docx'],
  excel: ['xls', 'xlsx'],
  tsv: ['tsv'],
  zip: ['zip', 'rar', 'iso'],
  illustrator: ['ai', 'esp'],
  powerpoint: ['ppt', 'pptx'],
  audio: ['wav', 'aif', 'mp3', 'aac'],
  recording: ['rec'],
  image: ['jpg', 'JPG', 'jpeg', 'JPEG', 'gif', 'GIF', 'bmp', 'png', 'PNG', 'svg'],
  video: ['m4v', 'avi', 'mpg', 'mp4', 'webm'],
  part: ['part'],
};

const getIconUrl = (icon: string) => `/assets/icons/files/${icon}.svg`;

// ----------------------------------------------------------------------

export function fileFormat(fileUrl: string | undefined) {
  if (!fileUrl) return '';

  const fileType = extractFileType(fileUrl);
  const format = Object.entries(FORMATS).find(([_, extensions]) => extensions.includes(fileType));
  return format ? format[0] : fileType;
}

// ----------------------------------------------------------------------

export function fileThumb(fileUrl: string) {
  const format = fileFormat(fileUrl);
  const icons: { [key: string]: string } = {
    md: 'ic_md',
    db: 'ic_mysql',
    web: 'ic_web',
    folder: 'ic_folder',
    csv: 'ic_csv',
    txt: 'ic_txt',
    tsv: 'ic_tsv',
    zip: 'ic_zip',
    audio: 'ic_audio',
    recording: 'ic_recording',
    video: 'ic_video',
    word: 'ic_word',
    excel: 'ic_excel',
    powerpoint: 'ic_power_point',
    pdf: 'ic_pdf',
    image: 'ic_img',
    part: 'ic_part',
  };

  return getIconUrl(icons[format] || 'ic_file');
}

// ----------------------------------------------------------------------

export function extractFileType(fileUrl = '') {
  return fileUrl.split('.').pop()?.split('?')[0] || '';
}

// ----------------------------------------------------------------------

export function extractFileName(fileUrl: string) {
  return fileUrl.split('/').pop();
}

// ----------------------------------------------------------------------

export function fileData(file: EnhancedFile | string) {
  if (typeof file === 'string') {
    return {
      key: file,
      preview: file,
      name: extractFileName(file),
      type: extractFileType(file),
    };
  }

  return {
    key: file.preview,
    name: file.name,
    size: file.size,
    path: file.path,
    type: file.type,
    preview: file.preview,
    lastModified: file.lastModified,
    lastModifiedDate: file.lastModifiedDate,
  };
}
