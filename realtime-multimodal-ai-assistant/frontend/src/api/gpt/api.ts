export const convertBase64 = (fileOrBase64: any, maxWidth = 720, quality = 0.8) =>
  new Promise((resolve, reject) => {
    const processImage = (dataUrl: string) => {
      const img = new Image();
      img.crossOrigin = 'Anonymous';

      img.onload = () => {
        const dimensions = { width: img.width, height: img.height };
        let { width, height } = dimensions;

        if (width > maxWidth) {
          height = Math.round((height * maxWidth) / width);
          width = maxWidth;
        }

        const canvas = document.createElement('canvas');
        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext('2d');
        if (!ctx) {
          reject(new Error('Could not get canvas context'));
          return;
        }

        ctx.drawImage(img, 0, 0, width, height);
        resolve(canvas.toDataURL('image/jpeg', quality));
      };

      img.onerror = () => {
        console.error('Failed to load image');
        resolve(dataUrl);
      };

      img.src = dataUrl;
    };

    if (typeof fileOrBase64 === 'string' && fileOrBase64.startsWith('data:')) {
      processImage(fileOrBase64);
      return;
    }

    const fileReader = new FileReader();
    fileReader.readAsDataURL(fileOrBase64);

    fileReader.onload = () => {
      processImage(fileReader.result as string);
    };

    fileReader.onerror = (error) => {
      reject(error);
      console.error('FileReader error:', error);
    };
  });
