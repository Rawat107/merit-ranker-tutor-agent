import { ChatRequest, AITutorResponse, Classification, Document, CacheEntry } from '../types/index.js';
import { Classifier } from '../classifier/Classifier.js';
import { SemanticCache } from '../cache/SemanticCache.js';
import { UpstashRetriever } from '../retriever/UpstashRetriever.js';
import { Reranker } from '../reranker/Reranker.js';
import { ModelSelector } from '../llm/ModelSelector.js';
import { webSearchTool } from '../tools/webSearch.js';
import { buildTutorPrompt } from '../utils/promptTemplates.js';
import pino from 'pino';

// --- Hardcoded user preferences for demo ---
const USER_PREFS = {
  language: 'en',
  exam: 'SSC_CGL_2025',
  tutorStyle: 'concise, step-by-step, exam-oriented',
  verification: 'Always verify math and reasoning answers. If unsure, say so.',
};

export interface StreamingHandlers {
  onToken: (token: string) => void;
  onMetadata: (metadata: any) => void;
  onComplete: (result: AITutorResponse) => void;
  onError: (error: Error) => void;
}

/**
 * Main LangChain-based Tutor Chain that orchestrates the entire AI tutoring flow
 */
export class TutorChain {
  private classifier: Classifier;
  private cache: SemanticCache;
  private retriever: UpstashRetriever;
  private reranker: Reranker;
  private modelSelector: ModelSelector;
  private logger: pino.Logger;

  constructor(
    classifier: Classifier,
    cache: SemanticCache,
    retriever: UpstashRetriever,
    reranker: Reranker,
    modelSelector: ModelSelector,
    logger: pino.Logger
  ) {
    this.classifier = classifier;
    this.cache = cache;
    this.retriever = retriever;
    this.reranker = reranker;
    this.modelSelector = modelSelector;
    this.logger = logger;
  }

  /**
   * Main execution method - implements full AI tutor flow
   */
  private async execute(request: ChatRequest): Promise<AITutorResponse> {
    this.logger.info(`Processing tutor request: ${request.message.substring(0, 100)}...`);

    // 1. Classification
    const classification = await this.classifyQuery(request);
    this.logger.info(`Query classified as: ${classification.subject}/${classification.level}`);

    // 2. Cache Layer: Exact match
    const cacheKey = this.cache.generateKey(request.message, classification.subject, USER_PREFS.language);
    const exactCache = await this.cache.lookupByKey(cacheKey);
    if (exactCache) {
      this.logger.info('Exact cache hit');
      return {
        answer: exactCache.response,
        sources: [],
        classification,
        cached: true,
        confidence: 0.98
      };
    }

    // 3. Semantic Cache (vector similarity)
    // (Assume embedder is available in cache or injected)
    let semanticCacheHit: CacheEntry | null = null;
    let semanticCacheSim = 0;
    if (this.cache.lookupByEmbedding) {
      const semResult = await this.cache.lookupByEmbedding(request.message, classification.subject, USER_PREFS.language);
      if (semResult && semResult.similarity >= 0.94 && semResult.entry) {
        this.logger.info('Semantic cache strong hit');
        return {
          answer: semResult.entry.response,
          sources: [],
          classification,
          cached: true,
          confidence: 0.95
        };
      }
      if (semResult && semResult.similarity >= 0.82 && semResult.entry) {
        semanticCacheHit = semResult.entry;
        semanticCacheSim = semResult.similarity;
        this.logger.info('Semantic cache soft hit, using as context seed');
      }
    }

    // 4. Retrieval: Knowledge base, Upstash, user/teacher notes
    const retrievalTasks: Promise<Document[]>[] = [
      this.retriever.getRelevantDocuments(request.message, {
        subject: classification.subject,
        level: classification.level,
        k: 8
      })
    ];
    // TODO: Add Bedrock KB, user/teacher notes, etc.

    // If semantic cache soft hit, use as context
    if (semanticCacheHit) {
      retrievalTasks.push(Promise.resolve([{
        id: 'cache-soft-hit',
        text: semanticCacheHit.response,
        metadata: { source: 'semantic-cache', sim: semanticCacheSim }
      }]));
    }

    // Web search fallback if needed
    if (classification.subject === 'current_affairs' || this.needsWebSearch(request.message)) {
      retrievalTasks.push(this.performWebSearch(request.message));
    }

    // Merge and rank all retrieved docs
    const retrievalResults = await Promise.allSettled(retrievalTasks);
    let allDocs: Document[] = [];
    retrievalResults.forEach(r => {
      if (r.status === 'fulfilled') allDocs.push(...r.value);
    });

    // 5. Rerank with cross-encoder
    let reranked: Document[] = [];
    if (allDocs.length > 0) {
      const rerankResults = await this.reranker.rerank(allDocs, request.message, 5);
      reranked = rerankResults.filter(r => r.score >= 0.5).map(r => r.document);
      // Fallback to web search if reranker confidence is low
      if (rerankResults.length && rerankResults[0].score < 0.8) {
        this.logger.info('Low reranker confidence, adding web search');
        const webDocs = await this.performWebSearch(request.message);
        reranked = reranked.concat(webDocs);
      }
    }

    // 6. Compose prompt
    const prompt = buildTutorPrompt(
      request,
      classification,
      reranked,
      USER_PREFS
    );

    // 7. Model selection (leave as is for now)
    const llm = await this.modelSelector.getLLM(classification, request.userSubscription);

    // 8. Generate answer (streaming not used here)
    const response = await llm.generate(prompt);

    // 9. Cache result
    await this.cache.upsert(request.message, response, {
      subject: classification.subject,
      level: classification.level,
      language: USER_PREFS.language
    });

    return {
      answer: response,
      sources: reranked,
      classification,
      cached: false,
      confidence: 0.85
    };
  }

  /**
   * Streaming execution for real-time responses
   */
  async runStreaming(request: ChatRequest, handlers: StreamingHandlers): Promise<void> {
    try {
      handlers.onToken('[Starting AI Tutor...]');

      // 1. Classification
      const classification = await this.classifyQuery(request);
      handlers.onMetadata({ classification, step: 'classification' });

      // 2. Cache Layer: Exact match
      const cacheKey = this.cache.generateKey(request.message, classification.subject, USER_PREFS.language);
      const exactCache = await this.cache.lookupByKey(cacheKey);
      if (exactCache) {
        handlers.onToken(exactCache.response);
        handlers.onComplete({
          answer: exactCache.response,
          sources: [],
          classification,
          cached: true,
          confidence: 0.98
        });
        return;
      }

      // 3. Semantic Cache (vector similarity)
      let semanticCacheHit: CacheEntry | null = null;
      let semanticCacheSim = 0;
      if (this.cache.lookupByEmbedding) {
        const semResult = await this.cache.lookupByEmbedding(request.message, classification.subject, USER_PREFS.language);
        if (semResult && semResult.similarity >= 0.94 && semResult.entry) {
          handlers.onToken(semResult.entry.response);
          handlers.onComplete({
            answer: semResult.entry.response,
            sources: [],
            classification,
            cached: true,
            confidence: 0.95
          });
          return;
        }
        if (semResult && semResult.similarity >= 0.82 && semResult.entry) {
          semanticCacheHit = semResult.entry;
          semanticCacheSim = semResult.similarity;
          handlers.onMetadata({ semanticCacheSim, step: 'semantic-cache-soft' });
        }
      }

      // 4. Retrieval
      handlers.onToken('[Retrieving context...]');
      const retrievalTasks: Promise<Document[]>[] = [
        this.retriever.getRelevantDocuments(request.message, {
          subject: classification.subject,
          level: classification.level,
          k: 8
        })
      ];
      if (semanticCacheHit) {
        retrievalTasks.push(Promise.resolve([{
          id: 'cache-soft-hit',
          text: semanticCacheHit.response,
          metadata: { source: 'semantic-cache', sim: semanticCacheSim }
        }]));
      }
      if (classification.subject === 'current_affairs' || this.needsWebSearch(request.message)) {
        retrievalTasks.push(this.performWebSearch(request.message));
      }
      const retrievalResults = await Promise.allSettled(retrievalTasks);
      let allDocs: Document[] = [];
      retrievalResults.forEach(r => {
        if (r.status === 'fulfilled') allDocs.push(...r.value);
      });

      // 5. Rerank
      let reranked: Document[] = [];
      if (allDocs.length > 0) {
        const rerankResults = await this.reranker.rerank(allDocs, request.message, 5);
        reranked = rerankResults.filter(r => r.score >= 0.5).map(r => r.document);
        if (rerankResults.length && rerankResults[0].score < 0.8) {
          handlers.onToken('[Low confidence, adding web search]');
          const webDocs = await this.performWebSearch(request.message);
          reranked = reranked.concat(webDocs);
        }
      }
      handlers.onMetadata({ sources: reranked.length, step: 'retrieval' });

      // 6. Compose prompt
      const prompt = buildTutorPrompt(
        request,
        classification,
        reranked,
        USER_PREFS
      );

      // 7. Model selection (leave as is for now)
      const llm = await this.modelSelector.getLLM(classification, request.userSubscription);

      // 8. Streaming answer
      let fullResponse = '';
      await llm.stream(prompt, {
        onToken: (token: string) => {
          fullResponse += token;
          handlers.onToken(token);
        },
        onError: handlers.onError,
        onComplete: async () => {
          await this.cache.upsert(request.message, fullResponse, {
            subject: classification.subject,
            level: classification.level,
            language: USER_PREFS.language
          });
          handlers.onComplete({
            answer: fullResponse,
            sources: reranked,
            classification,
            cached: false,
            confidence: 0.85
          });
        }
      });

    } catch (error) {
      this.logger.error(error, 'Streaming tutor chain failed');
      handlers.onError(error as Error);
    }
  }

  async run(request: ChatRequest): Promise<AITutorResponse> {
    return this.execute(request);
  }

  private async classifyQuery(request: ChatRequest): Promise<Classification> {
    if (request.subject && request.level) {
      return {
        subject: request.subject,
        level: request.level as 'basic' | 'intermediate' | 'advanced',
        confidence: 1.0
      };
    }
    return await this.classifier.classify(request.message);
  }

  private needsWebSearch(query: string): boolean {
    const webSearchKeywords = [
      'latest', 'recent', 'current', 'news', 'today', 'now',
      '2024', '2025', 'happening', 'update'
    ];
    const queryLower = query.toLowerCase();
    return webSearchKeywords.some(keyword => queryLower.includes(keyword));
  }

  private async performWebSearch(query: string): Promise<Document[]> {
    try {
      const webResults = await webSearchTool(query);
      return webResults.map((result, index) => ({
        id: `web-${index}`,
        text: `${result.title}: ${result.snippet}`,
        metadata: {
          source: 'web',
          url: result.url,
          title: result.title,
          relevance: result.relevance || 0.5
        }
      }));
    } catch (error) {
      this.logger.warn(error, 'Web search failed');
      return [];
    }
  }
}