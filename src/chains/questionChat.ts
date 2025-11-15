import { ChatRequest, AITutorResponse, Classification, Document } from '../types/index.js';
import { Classifier } from '../classifier/Classifier.js';
import { AWSKnowledgeBaseRetriever } from '../retriever/AwsKBRetriever.js';
import { ModelSelector } from '../llm/ModelSelector.js';
import { Reranker } from '../reranker/Reranker.js';
import { webSearchTool } from '../tools/webSearch.js';
import { EvaluatePrompt, EvaluatePromptInput, EvaluatePromptOutput } from '../prompts/evaluatorPrompt.js';
import { RedisCache } from '../cache/RedisCache.js';
import { ChatMemory } from '../cache/ChatMemory.js';
import { createClient } from 'redis';
import { appConfig } from '../config/modelConfig.js';
import { validateResponse } from '../utils/responseValidator.js';
import pino from 'pino';

const USER_PREFS = {
  language: 'en',
  exam: 'SSC_CGL_2025',
  tutorStyle: 'concise, step-by-step, exam-oriented',
  verification: 'Always verify math and reasoning answers.',
};

export interface StreamingHandlers {
  onToken: (token: string) => void;
  onMetadata: (metadata: any) => void;
  onComplete: (result: AITutorResponse) => void;
  onError: (error: Error) => void;
}

export class TutorChain {
  private classifier: Classifier;
  private retriever: AWSKnowledgeBaseRetriever;
  private modelSelector: ModelSelector;
  private reranker: Reranker;
  private evaluatePrompt: EvaluatePrompt;
  private logger: pino.Logger;
  private redisCache: RedisCache;
  private chatMemory: ChatMemory;

  constructor(
    classifier: Classifier,
    retriever: AWSKnowledgeBaseRetriever,
    modelSelector: ModelSelector,
    reranker: Reranker,
    logger: pino.Logger
  ) {
    this.classifier = classifier;
    this.retriever = retriever;
    this.modelSelector = modelSelector;
    this.reranker = reranker;
    this.logger = logger;
    this.evaluatePrompt = new EvaluatePrompt(modelSelector, logger);
    this.redisCache = new RedisCache(logger);
    
    // Initialize ChatMemory with Redis client
    const redisClient = createClient({
      url: appConfig.redis.url,
      password: appConfig.redis.password,
    });
    
    // Connect Redis client (lazy connection will be handled by ChatMemory methods)
    redisClient.connect().catch((err) => 
      logger.error(err, 'Failed to connect Redis client for ChatMemory')
    );
    
    this.chatMemory = new ChatMemory(redisClient as any, logger);
    
    logger.info('TutorChain initialized with ChatMemory');
  }

  /**
   * Core retrieval logic - shared by both streaming and non-streaming
   * ALWAYS tries web search as fallback when KB has no results
   * Bypasses reranking if no documents are retrieved
   */
  private async retrieveDocuments(
    query: string,
    classification: Classification,
    onProgress?: (message: string) => void
  ): Promise<Document[]> {
    const isAcademicSubject = this.isAcademicSubject(classification.subject);
    const isCurrentAffairs =
      classification.subject === 'current_affairs' ||
      classification.subject === 'general_knowledge';

    let retrievedDocs: Document[] = [];

    // ROUTE 1: Current affairs - go straight to web search
    if (isCurrentAffairs) {
      this.logger.info('[TutorChain] Route: Web Search (Current Affairs)');
      onProgress?.('🌐 Searching the web for latest information...\\n');
      retrievedDocs = await webSearchTool(query, classification.subject, this.logger);

      if (retrievedDocs.length === 0) {
        this.logger.warn('[TutorChain] Web search returned no results - will bypass reranking');
      }

      return retrievedDocs;
    }

    // ROUTE 2 & 3: Try KB first, fallback to web search if needed
    this.logger.info('[TutorChain] Route: Knowledge Base');
    onProgress?.('📖 Searching knowledge base...\\n');

    retrievedDocs = await this.retriever.getRelevantDocuments(query, {
      subject: classification.subject,
      level: classification.level,
      k: 5,
    });

    const topScore =
      retrievedDocs.length > 0 && retrievedDocs[0]?.score !== undefined
        ? retrievedDocs[0].score
        : 0;

    const shouldFallback = retrievedDocs.length === 0 || topScore < 0.5;

    // ALWAYS fallback to web search if KB fails or has low relevance
    if (shouldFallback) {
      const reason =
        retrievedDocs.length === 0
          ? 'KB empty'
          : `KB relevance too low (${(topScore * 100).toFixed(1)}%)`;

      this.logger.warn(`[TutorChain] ${reason}, falling back to web search`);
      onProgress?.(
        `${
          retrievedDocs.length === 0
            ? 'No KB results found'
            : `Low relevance (${(topScore * 100).toFixed(1)}%)`
        }\\n🌐 Searching the web instead...\\n`
      );

      const webDocs = await webSearchTool(query, classification.subject, this.logger);

      if (webDocs.length > 0) {
        retrievedDocs = webDocs;
        this.logger.info(
          { count: webDocs.length },
          '[TutorChain] Web search fallback successful'
        );
      } else {
        this.logger.warn('[TutorChain] Web search fallback also returned no results - will bypass reranking');
      }
    }

    return retrievedDocs;
  }

  /**
   * Main execution method (classification + retrieval)
   * Cache checks are SKIPPED on initial request to show classification quickly
   * Cache is only used during evaluate phase
   */
  private async execute(request: ChatRequest, sessionId?: string): Promise<AITutorResponse> {
    this.logger.info(
      { query: request.message.substring(0, 100) },
      '[TutorChain] Processing query'
    );

    try {
      // STEP 1: Classification (FAST - show immediately)
      const classification = await this.classifyQuery(request);

      this.logger.info(
        {
          subject: classification.subject,
          confidence: classification.confidence,
          intent: (classification as any).intent,
          sessionId,
        },
        '[TutorChain] Classification complete'
      );

      // STEP 2: Retrieve documents (existing flow)
      const retrievedDocs = await this.retrieveDocuments(request.message, classification);

      this.logger.info(
        { sourcesCount: retrievedDocs.length },
        '[TutorChain] Retrieval complete'
      );

      // Handle case when no documents are retrieved - bypass reranking
      if (retrievedDocs.length === 0) {
        this.logger.warn('[TutorChain] No documents retrieved - bypassing reranking');
        return {
          answer: '',
          sources: [],
          classification,
          cached: false,
          confidence: classification.confidence,
          metadata: {
            stage: 'no_results',
            message: 'No relevant information found. Try rephrasing your question or use a different query.',
          },
        };
      }

      // Return for UI to show rerank button
      return {
        answer: '',
        sources: retrievedDocs,
        classification,
        cached: false,
        confidence: classification.confidence,
        metadata: {
          stage: 'ready_for_reranking',
          message: 'Click "Rerank Documents" to rank by relevance',
        },
      };
    } catch (error) {
      this.logger.error(error, '[TutorChain] Error');
      throw error;
    }
  }

  /**
   * STREAMING EVALUATE: Generate final response after reranking with streaming
   * This is called when user clicks "Evaluate" button for streaming responses
   */
  async evaluateStreaming(
    userQuery: string,
    classification: Classification,
    documents: Document[],
    subscription: string = 'free',
    callbacks: {
      onToken: (token: string) => void;
      onMetadata: (metadata: any) => void;
      onComplete: (result: EvaluatePromptOutput) => void;
      onError: (error: Error) => void;
    },
    sessionId?: string // ADD: session ID for chat history
  ): Promise<void> {
    this.logger.info(
      {
        query: userQuery.substring(0, 80),
        subject: classification.subject,
        confidence: classification.confidence,
        docCount: documents.length,
        sessionId, // ADD: log session
      },
      '[TutorChain] Streaming evaluate step started'
    );

    try {
      // CHECK CACHE FIRST before streaming
      this.logger.debug('Checking cache before streaming...');
      
      // Check Direct Cache (exact match)
      const directCache = await this.redisCache.checkDirectCache(
        userQuery,
        classification.subject
      );

      if (directCache) {
        this.logger.info('✓ Returning from DIRECT cache (streaming)');
        callbacks.onToken(directCache.response);
        callbacks.onComplete({
          answer: directCache.response,
          modelUsed: directCache.metadata.modelUsed || 'cached',
          levelUsed: directCache.metadata.levelUsed || 'cached',
          latency: 0,
        });
        return;
      }

      // Check Semantic Cache (similarity match)
      const semanticCache = await this.redisCache.checkSemanticCache(
        userQuery,
        classification.subject
      );

      if (semanticCache) {
        this.logger.info({ score: semanticCache.score }, '✓ Returning from SEMANTIC cache (streaming)');
        callbacks.onToken(semanticCache.response);
        callbacks.onComplete({
          answer: semanticCache.response,
          modelUsed: semanticCache.metadata.modelUsed || 'cached',
          levelUsed: semanticCache.metadata.levelUsed || 'cached',
          latency: 0,
        });
        return;
      }

      this.logger.debug('Cache miss - starting streaming generation...');

      // Get top document as context
      const topDocument = documents.length > 0 ? documents[0] : null;

      this.logger.debug(
        { hasTopDoc: !!topDocument },
        '[TutorChain] Top document selected for streaming'
      );

      // STEP 1: Load conversation history
      let conversationHistory = '';
      let userName: string | null = null;
      
      if (sessionId) {
        const messages = await this.chatMemory.load(sessionId, 5);
        const formatted = this.chatMemory.format(messages, true);
        conversationHistory = formatted.historyText;
        userName = formatted.userName;
        
        this.logger.info({ sessionId, messageCount: messages.length, userName }, '✓ Loaded conversation history');
      } else {
        this.logger.warn('No sessionId provided - chat history disabled');
      }

      // Send initial metadata
      callbacks.onMetadata({
        step: 'preparation',
        docCount: documents.length,
        classification,
        hasHistory: conversationHistory.length > 0,
        userName,
      });

      // STEP 2: Call EvaluatePrompt with streaming and formatted history
      const evaluateInput: EvaluatePromptInput = {
        userQuery,
        classification,
        topDocument,
        userPrefs: USER_PREFS,
        subscription,
        conversationHistory,
        userName,
      };

      // Wrap onComplete to capture the answer for caching
      const originalOnComplete = callbacks.onComplete;
      let finalAnswer = '';
      let capturedResult: EvaluatePromptOutput | undefined = undefined;

      const wrappedCallbacks = {
        ...callbacks,
        onComplete: (result: EvaluatePromptOutput) => {
          finalAnswer = result.answer;
          capturedResult = result;
          originalOnComplete(result);
        },
      };

      await this.evaluatePrompt.evaluateStreaming(evaluateInput, wrappedCallbacks);

      // Store in cache after streaming completes
      if (capturedResult !== undefined) {
        this.logger.debug('Validating streaming result before caching...');
        
        const result = capturedResult as EvaluatePromptOutput;
        
        // Validate response quality before caching
        const validation = validateResponse(finalAnswer);
        
        if (validation.isValid) {
          this.logger.info({ score: validation.score.toFixed(2) }, 'Response passed quality check');
          
          await Promise.all([
            this.redisCache.storeDirectCache(
              userQuery,
              finalAnswer,
              classification.subject,
              {
                confidence: classification.confidence,
                modelUsed: result.modelUsed,
                levelUsed: result.levelUsed,
              }
            ),
            this.redisCache.storeSemanticCache(
              userQuery,
              finalAnswer,
              classification.subject,
              {
                confidence: classification.confidence,
                modelUsed: result.modelUsed,
                levelUsed: result.levelUsed,
              }
            ),
          ]);

          this.logger.info('✓ Stored streaming result in both caches');

          // STEP 3: Save conversation to memory
          if (sessionId) {
            await this.chatMemory.save(sessionId, userQuery, finalAnswer);
            this.logger.info({ sessionId }, '✓ Saved streaming conversation to memory');
          }
        } else {
          this.logger.warn(
            { 
              reason: validation.reason, 
              score: validation.score.toFixed(2),
              preview: finalAnswer.substring(0, 100)
            }, 
            '✗ Skipped caching - response failed quality check'
          );
        }
      }
    } catch (error) {
      this.logger.error({ error }, '[TutorChain] Streaming evaluate failed');
      callbacks.onError(error as Error);
    }
  }

  /**
   * EVALUATE: Generate final response after reranking
   * This is called when user clicks "Evaluate" button
   * UPDATED: Store in cache and chat history
   */
  async evaluate(
    userQuery: string,
    classification: Classification,
    documents: Document[],
    subscription?: string,
    sessionId?: string // ADD: session ID for chat history
  ): Promise<EvaluatePromptOutput> {
    this.logger.info(
      {
        query: userQuery.substring(0, 80),
        subject: classification.subject,
        confidence: classification.confidence,
        docCount: documents.length,
        sessionId, // ADD: log session
      },
      '[TutorChain] Evaluate step started'
    );

    try {
      // CHECK CACHE FIRST before generating answer
      this.logger.debug('Checking cache before generation...');
      
      // Check Direct Cache (exact match)
      const directCache = await this.redisCache.checkDirectCache(
        userQuery,
        classification.subject
      );

      if (directCache) {
        this.logger.info('✓ Returning from DIRECT cache (evaluate)');
        return {
          answer: directCache.response,
          modelUsed: directCache.metadata.modelUsed || 'cached',
          levelUsed: directCache.metadata.levelUsed || 'cached',
          latency: 0,
        };
      }

      // Check Semantic Cache (similarity match)
      const semanticCache = await this.redisCache.checkSemanticCache(
        userQuery,
        classification.subject
      );

      if (semanticCache) {
        this.logger.info({ score: semanticCache.score }, '✓ Returning from SEMANTIC cache (evaluate)');
        return {
          answer: semanticCache.response,
          modelUsed: semanticCache.metadata.modelUsed || 'cached',
          levelUsed: semanticCache.metadata.levelUsed || 'cached',
          latency: 0,
        };
      }

      this.logger.debug('Cache miss - generating new answer...');

      // Get top document as context
      const topDocument = documents.length > 0 ? documents[0] : null;

      this.logger.debug(
        { hasTopDoc: !!topDocument },
        '[TutorChain] Top document selected'
      );

      // STEP 1: Load conversation history
      let conversationHistory = '';
      let userName: string | null = null;
      
      if (sessionId) {
        const messages = await this.chatMemory.load(sessionId, 5);
        const formatted = this.chatMemory.format(messages, true);
        conversationHistory = formatted.historyText;
        userName = formatted.userName;
        
        this.logger.debug({ sessionId, messageCount: messages.length, userName }, 'Loaded conversation history');
      }

      // STEP 2: Call EvaluatePrompt with formatted history
      const evaluateInput: EvaluatePromptInput = {
        userQuery,
        classification,
        topDocument,
        userPrefs: USER_PREFS,
        subscription,
        conversationHistory,
        userName,
      };

      const output = await this.evaluatePrompt.evaluate(evaluateInput);

      this.logger.info(
        {
          modelUsed: output.modelUsed,
          levelUsed: output.levelUsed,
          latency: output.latency,
          answerLength: output.answer.length,
        },
        '[TutorChain] Evaluate complete'
      );

      // Validate response quality before caching
      const validation = validateResponse(output.answer);
      
      if (validation.isValid) {
        this.logger.info({ score: validation.score.toFixed(2) }, 'Response passed quality check');
        
        // Store in BOTH caches (direct and semantic)
        await Promise.all([
          this.redisCache.storeDirectCache(
            userQuery,
            output.answer,
            classification.subject,
            {
              confidence: classification.confidence,
              modelUsed: output.modelUsed,
              levelUsed: output.levelUsed,
            }
          ),
          this.redisCache.storeSemanticCache(
            userQuery,
            output.answer,
            classification.subject,
            {
              confidence: classification.confidence,
              modelUsed: output.modelUsed,
              levelUsed: output.levelUsed,
            }
          ),
        ]);

        // STEP 3: Save conversation to memory
        if (sessionId) {
          await this.chatMemory.save(sessionId, userQuery, output.answer);
          this.logger.info({ sessionId }, '✓ Saved conversation exchange to memory');
        }
      } else {
        this.logger.warn(
          { 
            reason: validation.reason, 
            score: validation.score.toFixed(2),
            preview: output.answer.substring(0, 100)
          }, 
          '✗ Skipped caching - response failed quality check'
        );
      }

      return output;
    } catch (error) {
      this.logger.error({ error }, '[TutorChain] Evaluate failed');
      throw error;
    }
  }

  /**
   * Check if subject is academic
   */
  private isAcademicSubject(subject: string): boolean {
    const academicSubjects = [
      'math',
      'science',
      'history',
      'literature',
      'reasoning',
      'english_grammer',
    ];
    return academicSubjects.includes(subject);
  }

  /**
   * Public run method
   * UPDATED: Accept sessionId for cache and chat history
   */
  async run(request: ChatRequest, sessionId?: string): Promise<AITutorResponse> {
    return this.execute(request, sessionId);
  }

  /**
   * Streaming version
   * UPDATED: Accept sessionId for cache and chat history
   */
  async runStreaming(
    request: ChatRequest,
    handlers: StreamingHandlers,
    sessionId?: string // ADD: session ID
  ): Promise<void> {
    try {
      handlers.onToken('[AI Tutor]\\n');
      handlers.onToken('Analyzing your question...\\n\\n');

      const classification = await this.classifyQuery(request);
      handlers.onMetadata({ classification, step: 'classification', sessionId });
      handlers.onToken(
        `Subject: ${classification.subject} (${(classification.confidence * 100).toFixed(0)}% confidence)\\n`
      );

      // ADD: Check direct cache before retrieval
      const directCache = await this.redisCache.checkDirectCache(
        request.message,
        classification.subject
      );

      if (directCache) {
        this.logger.info('✓ Returning from DIRECT cache in stream');
        handlers.onToken('\\n📦 Cached Response Found\\n\\n');
        handlers.onToken(directCache.response);
        handlers.onComplete({
          answer: directCache.response,
          sources: [],
          classification,
          cached: true,
          confidence: classification.confidence,
          metadata: { cacheType: 'direct' },
        });
        return;
      }

      // ADD: Check semantic cache
      const semanticCache = await this.redisCache.checkSemanticCache(
        request.message,
        classification.subject
      );

      if (semanticCache) {
        this.logger.info('✓ Returning from SEMANTIC cache in stream');
        handlers.onToken(`\\n📦 Similar Response Found (${(semanticCache.score * 100).toFixed(1)}% match)\\n\\n`);
        handlers.onToken(semanticCache.response);
        handlers.onComplete({
          answer: semanticCache.response,
          sources: [],
          classification,
          cached: true,
          confidence: semanticCache.score,
          metadata: { cacheType: 'semantic' },
        });
        return;
      }

      const retrievedDocs = await this.retrieveDocuments(
        request.message,
        classification,
        (message) => handlers.onToken(message)
      );

      handlers.onMetadata({ sources: retrievedDocs.length, step: 'retrieval', sessionId });
      handlers.onToken(`Found ${retrievedDocs.length} relevant sources\\n\\n`);

      // Handle case when no documents are retrieved - bypass reranking
      if (retrievedDocs.length === 0) {
        this.logger.warn('[TutorChain-Stream] No documents retrieved - bypassing reranking');
        handlers.onComplete({
          answer: '',
          sources: [],
          classification,
          cached: false,
          confidence: classification.confidence,
          metadata: {
            stage: 'no_results',
            message: 'No relevant information found. Try rephrasing your question or use a different query.',
          },
        });
        return;
      }

      handlers.onComplete({
        answer: '',
        sources: retrievedDocs,
        classification,
        cached: false,
        confidence: classification.confidence,
        metadata: {
          stage: 'ready_for_reranking',
          message: 'Click "Rerank Documents" to rank by relevance',
        },
      });

      this.logger.info('[TutorChain-Stream] Complete');
    } catch (error) {
      this.logger.error(error, '[TutorChain-Stream] Error');
      handlers.onError(error as Error);
    }
  }

  /**
   * Private classification logic
   */
  private async classifyQuery(request: ChatRequest): Promise<Classification> {
    if (request.subject && request.level) {
      return {
        subject: request.subject,
        level: request.level as 'basic' | 'intermediate' | 'advanced',
        confidence: 1.0,
      };
    }

    return await this.classifier.classify(request.message);
  }
}

export function createTutorChain(
  classifier: Classifier,
  retriever: AWSKnowledgeBaseRetriever,
  modelSelector: ModelSelector,
  reranker: Reranker,
  logger: pino.Logger
): TutorChain {
  return new TutorChain(classifier, retriever, modelSelector, reranker, logger);
}