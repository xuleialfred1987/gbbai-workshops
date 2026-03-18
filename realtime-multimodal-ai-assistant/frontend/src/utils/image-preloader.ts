// utils/image-preloader.ts
export class ImagePreloader {
  private static cache = new Map<string, Promise<void>>();

  private static loadedImages = new Set<string>();

  static async preload(url: string): Promise<void> {
    // If already loaded, resolve immediately
    if (this.loadedImages.has(url)) {
      return Promise.resolve();
    }

    // If currently loading, return existing promise
    if (this.cache.has(url)) {
      return this.cache.get(url)!;
    }

    // Create new preload promise
    const promise = new Promise<void>((resolve, reject) => {
      const img = new Image();
      
      img.onload = () => {
        this.loadedImages.add(url);
        this.cache.delete(url); // Clean up promise from cache
        resolve();
      };
      
      img.onerror = () => {
        this.cache.delete(url); // Clean up promise from cache
        reject(new Error(`Failed to load image: ${url}`));
      };
      
      img.src = url;
    });

    this.cache.set(url, promise);
    return promise;
  }

  static isLoaded(url: string): boolean {
    return this.loadedImages.has(url);
  }

  static isLoading(url: string): boolean {
    return this.cache.has(url);
  }

  static clearCache(): void {
    this.cache.clear();
    this.loadedImages.clear();
  }
}
