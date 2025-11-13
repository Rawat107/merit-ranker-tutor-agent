import { TavilySearch, type TopicType, type SearchDepth } from "@langchain/tavily";
import pino from 'pino';
import { Document } from '../types/index.js';

/**
 * Clean and extract meaningful text from content
 * Removes links, navigation, tables, and formatting
 */
function cleanContent(text: string): string {
  if (!text) return '';

  // Remove markdown links [text](url)
  text = text.replace(/\[([^\]]+)\]\([^)]*\)/g, '$1');

  // Remove HTML tags
  text = text.replace(/<[^>]*>/g, '');

  // Remove URLs
  text = text.replace(/https?:\/\/[^\s]+/g, '');

  // Remove common navigation text
  const navigationKeywords = [
    /\[.*?(jump|skip|top|content|bodyContent)\.*\]/gi,
    /Jump to content/gi,
    /Toggle.*?subsection/gi,
    /See also/gi,
    /Notes/gi,
    /References/gi,
    /Languages/gi,
    /Add topic/gi,
  ];

  navigationKeywords.forEach(regex => {
    text = text.replace(regex, '');
  });

  // Remove multiple spaces and newlines
  text = text.replace(/\s\s+/g, ' ');
  text = text.trim();

  // Take first 300 characters (good summary length)
  // if (text.length > 300) {
  //   text = text.substring(0, 300).trim() + '...';
  // }

  return text;
}

/**
 * Initialize Tavily search tool with LangChain
 */
function initializeTavilyTool(subject: string, logger?: pino.Logger): TavilySearch {
  const topicMap: Record<string, TopicType> = {
    current_affairs: 'news',
    general_knowledge: 'general',
    finance: 'finance',
    default: 'general',
  };

  const topic = topicMap[subject] || topicMap.default;
  const searchDepth: SearchDepth = topic === 'news' ? 'advanced' : 'basic';

  logger?.debug(
    { subject, topic, searchDepth },
    '[Tavily] Initializing search tool'
  );

  return new TavilySearch({
    maxResults: 5,
    tavilyApiKey: process.env.TAVILY_API_KEY || '',
    includeRawContent: false,
    searchDepth: searchDepth,
    topic: topic,
    includeAnswer: true,
  });
}

/**
 * Perform web search using LangChain Tavily tool
 */
export async function webSearchTool(
  query: string,
  subject: string = 'general',
  logger?: pino.Logger
): Promise<Document[]> {
  logger?.info(
    { query: query.substring(0, 100), subject },
    '[Tavily] Initiating web search'
  );

  if (!process.env.TAVILY_API_KEY) {
    logger?.error('[Tavily] TAVILY_API_KEY not set');
    return [];
  }

  try {
    const tavilyTool = initializeTavilyTool(subject, logger);
    const searchResults = await tavilyTool.invoke({ query });

    let results: any[] = [];

    if (typeof searchResults === 'string') {
      try {
        const parsed = JSON.parse(searchResults);
        results = parsed.results || [parsed];
      } catch (e) {
        results = [{ title: 'Search Result', answer: searchResults, url: '' }];
      }
    } else if (typeof searchResults === 'object' && searchResults !== null) {
      if ('results' in searchResults && Array.isArray(searchResults.results)) {
        results = searchResults.results;
      } else if (Array.isArray(searchResults)) {
        results = searchResults;
      } else {
        return [];
      }
    }

    if (!results || results.length === 0) {
      logger?.warn('[Tavily] No results returned');
      return [];
    }

    // IMPORTANT: Use answer field, clean it, DON'T use content
    const documents: Document[] = results.map((result, index) => {
      // Priority: answer > snippet > content
      let summary = result.answer || result.snippet || result.content || 'No summary available';

      // CLEAN THE SUMMARY
      summary = cleanContent(summary);

      return {
        id: `web-${index}`,
        text: `**${result.title || 'Untitled'}**\n\n${summary}`,
        metadata: {
          source: 'web-search',
          url: result.url || '',
          title: result.title || `Result ${index + 1}`,
          relevance: result.score || 0.7,
          topic: subject,
        },
        score: result.score || 0.7,
      };
    });

    logger?.info(
      { count: documents.length, topScore: documents[0]?.score },
      '[Tavily]  Web search complete'
    );

    return documents;
  } catch (error: any) {
    logger?.error(
      { error: error.message },
      '[Tavily]  Search failed'
    );
    return [];
  }
}
