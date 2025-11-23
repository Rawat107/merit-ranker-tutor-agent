import pino from "pino";

const logger = pino();

export interface UnsplashImage {
  id: string;
  urls: {
    raw: string;
    full: string;
    regular: string;
    small: string;
    thumb: string;
  };
  alt_description: string | null;
  description: string | null;
  user: {
    name: string;
    username: string;
  };
  links: {
    download_location: string;
  };
}

export interface UnsplashSearchResponse {
  results: UnsplashImage[];
  total: number;
  total_pages: number;
}

export type LayoutType = 'vertical' | 'horizontal' | 'left' | 'right';

/**
 * Get image from Unsplash based on query
 * Returns the regular resolution URL (best for web presentations)
 */
export async function getImageFromUnsplash(
  query: string,
  layoutType?: LayoutType
): Promise<string | null> {
  try {
    if (!query || query.trim().length === 0) {
      query = 'presentation background';
    }

    logger.info({ query, layoutType }, '[ImageSearch] Searching Unsplash');

    // Determine orientation based on layout
    const orientationQuery =
      layoutType === 'vertical'
        ? '&orientation=landscape'
        : layoutType === 'left' || layoutType === 'right'
          ? '&orientation=portrait'
          : '&orientation=landscape';

    const response = await fetch(
      `https://api.unsplash.com/search/photos?query=${encodeURIComponent(query)}&page=1&per_page=1${orientationQuery}`,
      {
        headers: {
          'Authorization': `Client-ID ${process.env.UNSPLASH_ACCESS_KEY}`,
        },
      }
    );

    if (!response.ok) {
      throw new Error(`Unsplash API error: ${response.status}`);
    }

    const data: UnsplashSearchResponse = await response.json();

    if (!data.results || data.results.length === 0) {
      logger.warn({ query }, '[ImageSearch] No images found');
      return null;
    }

    const imageUrl = data.results[0]?.urls.regular;

    if (!imageUrl) {
      logger.warn({ query }, '[ImageSearch] No image URL found');
      return null;
    }

    logger.info({ imageUrl }, '[ImageSearch] Image found');
    return imageUrl;
  } catch (error) {
    logger.error({ error, query }, '[ImageSearch] Failed to get image');
    return null;
  }
}
