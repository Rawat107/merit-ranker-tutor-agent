import { WebSearchResult } from '../types/index.js';
import pino from 'pino';

// Simple stub; replace with SerpAPI / custom Playwright scraping.
export async function webSearchTool(query: string, logger?: pino.Logger): Promise<WebSearchResult[]> {
  logger?.info({ query }, 'webSearch stub invoked');
  return [
    { title: 'Example Source 1', url: 'https://example.com/1', snippet: `Info about ${query}`, relevance: 0.6 },
    { title: 'Example Source 2', url: 'https://example.com/2', snippet: `Further context on ${query}`, relevance: 0.55 }
  ];
}
