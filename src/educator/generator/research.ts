import pino from 'pino';
import { RunnableLambda } from '@langchain/core/runnables';
import { Document, EnrichedTopic, BlueprintInput, EnrichedBlueprint } from '../../types/index.js';
import { webSearchTool } from '../../tools/webSearch.js';
import { AWSKnowledgeBaseRetriever } from '../../retriever/AwsKBRetriever.js';

/**
 * RESEARCH BATCH PROCESSOR
 *
 */
export class ResearchBatchProcessor {
  private kbRetriever: AWSKnowledgeBaseRetriever;
  private logger: pino.Logger;
  private readonly retryLimit = 2;

  constructor(logger: pino.Logger) {
    this.logger = logger;
    this.kbRetriever = new AWSKnowledgeBaseRetriever(logger);
  }

  /**
   * Main entry point: Enrich all topics with research
   */
  async enrichBlueprint(
    blueprint: BlueprintInput,
    classification: { subject: string; level: string },
    maxConcurrency = 5
  ): Promise<EnrichedBlueprint> {
    const startTime = Date.now();

    this.logger.info(
      {
        topicsCount: blueprint.topics.length,
        subject: blueprint.subject,
        examTags: blueprint.examTags,
        maxConcurrency,
      },
      '[Research] Starting parallel research for all topics using .batch()'
    );

    try {
      // Create runnable for batch processing
      const runnableResearch = RunnableLambda.from(async (topic: { topicName: string; level: string[]; noOfQuestions: number }) => {
        return await this.enrichSingleTopic(topic, blueprint, classification);
      });

      // Process all topics with maxConcurrency control
      const enrichedTopics = await runnableResearch.batch(
        blueprint.topics,
        { maxConcurrency }
      );

      const duration = Date.now() - startTime;

      this.logger.info(
        {
          topicsCount: enrichedTopics.length,
          topicNamesEnriched: enrichedTopics.map(t => t.topicName),
          duration,
          successfulTopics: enrichedTopics.filter(t => t.research.web.length > 0 || t.research.kb.length > 0).length,
        },
        '[Research] ✅ All topics enriched with research'
      );

      return {
        examTags: blueprint.examTags,
        subject: blueprint.subject,
        totalQuestions: blueprint.totalQuestions,
        topics: enrichedTopics,
      };
    } catch (error) {
      this.logger.error({ error }, '[Research] ❌ Research batching failed');
      throw error;
    }
  }

  /**
   * Enrich a single topic with 3 web + 3 KB results (all parallel)
   */
  private async enrichSingleTopic(
    topic: { topicName: string; level: string[]; noOfQuestions: number },
    blueprint: BlueprintInput,
    classification: { subject: string; level: string }
  ): Promise<EnrichedTopic> {
    const startTime = Date.now();

    this.logger.info(
      { topicName: topic.topicName, level: topic.level },
      '[Research] Starting research for topic'
    );

    try {
      // Build 3 different search queries for diverse results
      const queries = this.buildSearchQueries(topic, blueprint);

      // Run 6 parallel research calls: 3 web queries + 3 KB queries
      const [webResults, kbResults] = await Promise.all([
        // Web searches with 3 different queries
        this.fetchMultipleWebResults(queries, blueprint.subject),
        
        // KB retrieval with 3 different queries
        this.fetchMultipleKBResults(queries, classification),
      ]);

      const duration = Date.now() - startTime;

      this.logger.info(
        {
          topicName: topic.topicName,
          webCount: webResults.length,
          kbCount: kbResults.length,
          duration,
        },
        '[Research] Topic research complete'
      );

      // Return topic with research results
      return {
        topicName: topic.topicName,
        level: topic.level,
        noOfQuestions: topic.noOfQuestions,
        research: {
          web: webResults,
          kb: kbResults,
        },
      };
    } catch (error) {
      const duration = Date.now() - startTime;
      
      this.logger.error(
        { topicName: topic.topicName, error, duration },
        '[Research] Topic research failed, returning empty results'
      );

      // Return topic with empty results on error
      return {
        topicName: topic.topicName,
        level: topic.level,
        noOfQuestions: topic.noOfQuestions,
        research: {
          web: [],
          kb: [],
        },
      };
    }
  }

  /**
   * Build 3 different search queries 
   */
  private buildSearchQueries(
    topic: { topicName: string; level: string[]; noOfQuestions: number },
    blueprint: BlueprintInput
  ): string[] {
    const examContext = blueprint.examTags.join(' ');
    const subject = blueprint.subject;
    const topicName = topic.topicName;

    // Query 1: Pattern-focused
    const patternQuery = `${topicName} question patterns ${examContext} ${subject} question types common mistakes`;

    // Query 2: Summary-focused
    const summaryQuery = `${topicName} ${examContext} syllabus topic overview ${subject} concepts`;

    // Query 3: Examples-focused
    const examplesQuery = `${topicName} solved examples ${examContext} ${subject} practice questions sample problems`;

    return [patternQuery, summaryQuery, examplesQuery];
  }

  /**
   * Fetch web results for multiple queries in parallel
   */
  private async fetchMultipleWebResults(
    queries: string[],
    subject: string
  ): Promise<Document[]> {
    try {
      // Run all 3 web searches in parallel
      const resultArrays = await Promise.all(
        queries.map(query => this.fetchWebResultsWithRetry(query, subject))
      );

      // Flatten and deduplicate by ID/URL
      const allResults = resultArrays.flat();
      const seen = new Set<string>();
      const deduped: Document[] = [];

      for (const doc of allResults) {
        const key = doc.id || (doc.metadata?.url as string) || doc.text.substring(0, 50);
        if (!seen.has(key)) {
          seen.add(key);
          deduped.push(doc);
        }
      }

      return deduped;
    } catch (error) {
      this.logger.warn({ error }, '[Research] Multiple web searches failed');
      return [];
    }
  }

  /**
   * Fetch KB results for multiple queries in parallel
   */
  private async fetchMultipleKBResults(
    queries: string[],
    classification: { subject: string; level: string }
  ): Promise<Document[]> {
    try {
      // Run all 3 KB retrievals in parallel
      const resultArrays = await Promise.all(
        queries.map(query => this.fetchKBResultsWithRetry(query, classification))
      );

      // Flatten and deduplicate by ID
      const allResults = resultArrays.flat();
      const seen = new Set<string>();
      const deduped: Document[] = [];

      for (const doc of allResults) {
        if (!seen.has(doc.id)) {
          seen.add(doc.id);
          deduped.push(doc);
        }
      }

      return deduped;
    } catch (error) {
      this.logger.warn({ error }, '[Research] Multiple KB retrievals failed');
      return [];
    }
  }

  /**
   * Fetch web results with retry and fallback
   */
  private async fetchWebResultsWithRetry(
    query: string,
    subject: string,
    attempt: number = 0
  ): Promise<Document[]> {
    try {
      const results = await webSearchTool(
        query,
        subject,
        process.env.TAVILY_API_KEY || '',
        this.logger
      );

      if (results.length > 0) {
        return results.slice(0, 3); // Top 3 per query
      }

      // If empty and retries left, try broader query
      if (attempt < this.retryLimit) {
        const fallbackQuery = this.buildFallbackQuery(query, subject);
        this.logger.info(
          { originalQuery: query, fallbackQuery, attempt: attempt + 1 },
          '[Research] Retrying with fallback query'
        );
        return this.fetchWebResultsWithRetry(fallbackQuery, subject, attempt + 1);
      }

      return [];
    } catch (error) {
      if (attempt < this.retryLimit) {
        this.logger.warn(
          { query, attempt: attempt + 1, error },
          '[Research] Web search failed, retrying'
        );
        const fallbackQuery = this.buildFallbackQuery(query, subject);
        return this.fetchWebResultsWithRetry(fallbackQuery, subject, attempt + 1);
      }

      this.logger.warn({ query, error }, '[Research] Web search exhausted retries');
      return [];
    }
  }

  /**
   * Fetch KB results with retry
   */
  private async fetchKBResultsWithRetry(
    query: string,
    classification: { subject: string; level: string },
    attempt: number = 0
  ): Promise<Document[]> {
    try {
      const results = await this.kbRetriever.getRelevantDocuments(query, {
        subject: classification.subject,
        level: classification.level,
        k: 3, // Get top 3 per query
      });

      return results.slice(0, 3);
    } catch (error) {
      if (attempt < this.retryLimit) {
        this.logger.warn(
          { query, attempt: attempt + 1, error },
          '[Research] KB retrieval failed, retrying'
        );
        // Wait a bit before retry
        await new Promise(resolve => setTimeout(resolve, 500));
        return this.fetchKBResultsWithRetry(query, classification, attempt + 1);
      }

      this.logger.warn({ query, error }, '[Research] KB retrieval exhausted retries');
      return [];
    }
  }

  /**
   * Build a fallback query by removing exam tags or broadening scope
   */
  private buildFallbackQuery(originalQuery: string, subject: string): string {
    // Remove specific exam tags and make query broader
    const tokens = originalQuery.split(' ').filter(t => 
      !t.includes('SSC') && 
      !t.includes('UPSC') && 
      !t.includes('GATE') &&
      !t.includes('CAT') &&
      !t.includes('JEE') &&
      !t.includes('NEET') &&
      !t.includes('Tier')
    );

    // If still long, keep first half and add subject
    if (tokens.length > 5) {
      return tokens.slice(0, 5).join(' ') + ' ' + subject;
    }

    return tokens.join(' ') + ' ' + subject + ' questions';
  }
}

/**
 * Factory function
 */
export function createResearchBatchProcessor(
  logger: pino.Logger
): ResearchBatchProcessor {
  return new ResearchBatchProcessor(logger);
}
