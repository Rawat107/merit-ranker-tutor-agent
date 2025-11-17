import { TavilySearch, type TopicType, type SearchDepth } from "@langchain/tavily";
import pino from 'pino';
import { Document } from '../types/index.js';
import { tr } from "zod/v4/locales";

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
function initializeTavilyTool(subject: string, tavilyApiKey: string, logger?: pino.Logger): TavilySearch {
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
    tavilyApiKey: tavilyApiKey || '',
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
  tavilyApiKey: string,
  logger?: pino.Logger
): Promise<Document[]> {
  logger?.info(
    { query: query.substring(0, 100), subject },
    '[Tavily] Initiating web search'
  );

  if (!tavilyApiKey) {
    logger?.warn('[Tavily] TAVILY_API_KEY not set - returning empty results');
    logger?.warn('[Tavily] Set TAVILY_API_KEY in .env to enable web search');
    return [];
  }

  try {
    // Tavily has a max query length of 400 characters - truncate if needed
    const MAX_QUERY_LENGTH = 400;
    let truncatedQuery = query;
    
    if (query.length > MAX_QUERY_LENGTH) {
      truncatedQuery = query.substring(0, MAX_QUERY_LENGTH - 3) + '...';
      logger?.warn(
        { originalLength: query.length, truncatedLength: truncatedQuery.length },
        '[Tavily] Query truncated to fit 400 character limit'
      );
    }

    const tavilyTool = initializeTavilyTool(subject, tavilyApiKey, logger);
    const searchResults = await tavilyTool.invoke({ query: truncatedQuery });

    let results: any[] = [];

    // Handle string response
    if (typeof searchResults === 'string') {
      try {
        const parsed = JSON.parse(searchResults);
        
        // Check for error in parsed response
        if (parsed.error) {
          logger?.warn(
            { error: parsed.error, query: truncatedQuery.substring(0, 80) },
            '[Tavily] API returned error'
          );
          return [];
        }
        
        results = parsed.results || [parsed];
      } catch (e) {
        results = [{ title: 'Search Result', answer: searchResults, url: '' }];
      }
    } 
    // Handle object response
    else if (typeof searchResults === 'object' && searchResults !== null) {
      // Check for error field
      if ('error' in searchResults) {
        logger?.warn(
          { error: searchResults.error, query: truncatedQuery.substring(0, 80) },
          '[Tavily] API returned error - no web results found'
        );
        return [];
      }
      
      if ('results' in searchResults && Array.isArray(searchResults.results)) {
        results = searchResults.results;
      } else if (Array.isArray(searchResults)) {
        results = searchResults;
      } else {
        logger?.warn({ searchResults }, '[Tavily] Unexpected response format');
        return [];
      }
    }

    if (!results || results.length === 0) {
      logger?.warn('[Tavily] No results returned from API');
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
      '[Tavily] Web search complete'
    );

    return documents;
  } catch (error: any) {
    logger?.error(
      { error: error.message, stack: error.stack },
      '[Tavily] Search failed'
    );
    return [];
  }
}
