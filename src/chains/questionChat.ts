
import { RunnableSequence } from "@langchain/core/runnables";
import { Classification, Document, ChatRequest, AITutorResponse } from '../types/index.js';
import { Classifier } from '../classifier/Classifier.js';
import { AWSKnowledgeBaseRetriever } from '../retriever/AwsKBRetriever.js';
import { ModelSelector } from '../llm/ModelSelector.js';
import { Reranker } from '../reranker/Reranker.js';
import { EvaluatePrompt, EvaluatePromptInput, EvaluatePromptOutput } from '../prompts/evaluatorPrompt.js';
import { RedisCache } from '../cache/RedisCache.js'; 
import { ChatMemory } from '../cache/ChatMemory.js';
import { webSearchTool } from '../tools/webSearch.js';
import { validateResponse } from '../utils/responseValidator.js';
import pino from 'pino';

const USER_PREFS = {
  language: 'en',
  exam: 'SSC_CGL_2025',
  tutorStyle: 'concise, step-by-step, exam-oriented',
  verification: 'Always verify math and reasoning answers.',
};

/**
 * TutorChain - Main tutoring chain using RunnableSequence
 * Automatically traces all steps in LangSmith
 */
export class TutorChain {
  private sequence: RunnableSequence;
  private redisCache: RedisCache; // ← ADD THIS

  constructor(
    private classifier: Classifier,
    private retriever: AWSKnowledgeBaseRetriever,
    private modelSelector: ModelSelector,
    private reranker: Reranker,
    private evaluatePrompt: EvaluatePrompt,
    private chatMemory: ChatMemory,
    private logger: pino.Logger,
    private secrets: any 
  ) {

    // Initialize RedisCache
    this.redisCache = new RedisCache(logger); 
    
    // Build the RunnableSequence
    this.sequence = this.buildSequence();
  }

  /**
   * Build the RunnableSequence with all steps
   */
  private buildSequence(): RunnableSequence<any, any> {
    return RunnableSequence.from([
      // STEP 1: Load conversation history and pass through fields
      {
        message: (input: any) => input.message,
        sessionId: (input: any) => input.sessionId,
        subject: (input: any) => input.subject,
        level: (input: any) => input.level,
        conversationHistory: async (input: any) => {
          if (!input.sessionId) {
            return '';
          }
          
          try {
            const messages = await this.chatMemory.load(input.sessionId, 5);
            // For standalone rewrite: only user queries, no AI responses
            const userQueriesOnly = this.chatMemory.formatUserQueriesOnly(messages);
            this.logger.debug('[TutorChain] Loaded user queries for standalone rewrite');
            return userQueriesOnly;
          } catch (error) {
            this.logger.warn({ error, sessionId: input.sessionId }, '[TutorChain] Failed to load conversation history');
            return '';
          }
        },

      },
      async (input: any) => {
      try {
        // STEP 1: Get standalone query first (if history exists)
        let standaloneQuery = input.message;
        
        // Only rewrite if we have actual conversation history
        if (input.conversationHistory && input.conversationHistory.trim().length > 20) {
          try {
            // Rewrite to standalone using classifier's method
            standaloneQuery = await this.classifier['rewriteToStandalone'](
              input.message,
              input.conversationHistory
            );
            this.logger.info({
              original: input.message.substring(0, 60),
              standalone: standaloneQuery.substring(0, 60)
            }, '[TutorChain] Query rewritten to standalone');
          } catch (error) {
            this.logger.warn({ error }, '[TutorChain] Standalone rewrite failed, using original');
            standaloneQuery = input.message;
          }
        } else {
          this.logger.debug('[TutorChain] No history or too short, using original query');
        }

        // STEP 2: Launch classification and retrieval in parallel (both use standalone)
        const isCurrentAffairs =
          input.subject === 'current_affairs' || input.subject === 'general_knowledge';

        const classifyPromise = this.classifier.classify(standaloneQuery);


        let retrievePromise;
        if (isCurrentAffairs) {
          retrievePromise = webSearchTool(
            standaloneQuery,  
            input.subject || 'general_knowledge',
            this.secrets.tavilyApiKey,
            this.logger
          );
        } else {
          retrievePromise = this.retriever.getRelevantDocuments(
            standaloneQuery,  
            {
              subject: input.subject,
              level: input.level,
              k: 5,
            }
          );
        }

        // Wait for both in parallel
        const [classification, retrievedDocs] = await Promise.all([
          classifyPromise,
          retrievePromise,
        ]);

        // Fallback to web search if KB retrieval weak
        let finalDocs = retrievedDocs;
        if (!isCurrentAffairs && (retrievedDocs.length === 0 || (retrievedDocs[0]?.score ?? 0) < 0.5)) {
          this.logger.warn('[TutorChain] Fallback to web search');
          finalDocs = await webSearchTool(
            standaloneQuery,  
            classification.subject,
            this.secrets.tavilyApiKey,
            this.logger
          );
        }

        this.logger.info({
          subject: classification.subject,
          level: classification.level,
          confidence: classification.confidence
        }, '[TutorChain] Classification complete');
        
        this.logger.info({ count: finalDocs.length }, '[TutorChain] Retrieval complete');
        
        return { ...input, classification, retrievedDocs: finalDocs, standaloneQuery };
      } catch (error) {
        this.logger.error({ error, query: input.message }, '[TutorChain] Parallel classify/retrieve failed');
        throw error;
      }
    },



      // STEP 3: Rerank documents (documents-first)
      async (input: any) => {
        try {
          const docs = input.retrievedDocs || []; // defensive
          if (docs.length === 0) {
            this.logger.warn('[TutorChain] No documents to rerank');
            return { ...input, rerankedDocs: [] };
          }

          const queryForRerank = input.standaloneQuery || input.message;

          this.logger.info({ docCount: docs.length }, '[TutorChain] Reranking documents');

          // documents, query, topK
          const reranked = await this.reranker.rerank(
            docs,
            queryForRerank,
            3
          );

          this.logger.info(
            { originalCount: docs.length, rerankedCount: reranked.length },
            '[TutorChain] Reranking complete'
          );

          return { ...input, rerankedDocs: reranked };
        } catch (error) {
          this.logger.error({ error }, '[TutorChain] Reranking failed, using original order');
          return { ...input, rerankedDocs: input.retrievedDocs || [] };
        }
      },


      // STEP 4: Evaluate and generate response
      async (input: any) => {
        try {
          const topDocument = input.rerankedDocs.length > 0 ? input.rerankedDocs[0] : null;

          const evaluateInput: EvaluatePromptInput = {
            userQuery: input.message,
            classification: input.classification,
            topDocument,
            userPrefs: USER_PREFS,
            subscription: 'free',
            conversationHistory: input.conversationHistory,
          };

          this.logger.info('[TutorChain] Starting evaluation');

          const output = await this.evaluatePrompt.evaluate(evaluateInput);

          this.logger.info(
            { modelUsed: output.modelUsed, answerLength: output.answer.length },
            '[TutorChain] Evaluation complete'
          );

          return {
            ...input,
            evaluationOutput: output,
          };
        } catch (error) {
          this.logger.error({ error }, '[TutorChain] Evaluation failed');
          throw error;
        }
      },

      // STEP 5: Save to memory and validate
      async (input: any) => {
        try {
          const validation = validateResponse(input.evaluationOutput.answer);

          if (validation.isValid && input.sessionId) {
            await this.chatMemory.save(input.sessionId, input.message, input.evaluationOutput.answer);
            this.logger.info({ sessionId: input.sessionId }, '[TutorChain] Saved to memory');
          }

          const response: AITutorResponse = {
            answer: input.evaluationOutput.answer,
            sources: input.rerankedDocs,
            classification: input.classification,
            cached: false,
            confidence: input.classification.confidence,
            metadata: {
              modelUsed: input.evaluationOutput.modelUsed,
              validationScore: validation.score,
            },
          };

          return response;
        } catch (error) {
          this.logger.error({ error }, '[TutorChain] Finalize failed');
          throw error;
        }
      },
    ]).withConfig({
      runName: 'TutorChain',
      metadata: {
        project: process.env.LANGCHAIN_PROJECT || 'merit-ranker-tutor-agent',
      },
    }) as RunnableSequence<any, any>;
  }

  async classifyAndRetrieve(
    request: ChatRequest,
    sessionId?: string
  ): Promise<{ classification: Classification, sources: Document[], cached: boolean, answer?: string }> {
    const { message, subject, level } = request;
    
    // First, check cache like in run() method
    const directCache = await this.redisCache.checkDirectCache(
      message,
      subject || 'general'
    );
    if (directCache) {
      this.logger.info('✓ DIRECT CACHE HIT in classifyAndRetrieve');
      const classification = {
          subject: directCache.metadata.subject || subject || 'general',
          level: directCache.metadata.level || 'intermediate',
          confidence: directCache.metadata.confidence || 1.0,
      };
      return {
        classification: classification,
        sources: [], // No documents to return for a cached answer
        cached: true,
        answer: directCache.response,
      };
    }
    const semanticCache = await this.redisCache.checkSemanticCache(
      message,
      subject || 'general'
    );
    if (semanticCache) {
      this.logger.info({ score: semanticCache.score }, '✓ SEMANTIC CACHE HIT in classifyAndRetrieve');
      const classification = {
          subject: semanticCache.metadata.subject || subject || 'general',
          level: semanticCache.metadata.level || 'intermediate',
          confidence: semanticCache.metadata.confidence || semanticCache.score,
      };
      return {
        classification: classification,
        sources: [], // No documents to return for a cached answer
        cached: true,
        answer: semanticCache.response,
      };
    }

    // Manually run the initial parts of the chain if no cache hit
    let standaloneQuery = message;
    if (sessionId) {
      const messages = await this.chatMemory.load(sessionId, 5);
      const userQueriesOnly = this.chatMemory.formatUserQueriesOnly(messages);
      if (userQueriesOnly && userQueriesOnly.trim().length > 20) {
        try {
          standaloneQuery = await this.classifier['rewriteToStandalone'](
            message,
            userQueriesOnly
          );
          this.logger.info({
            original: message.substring(0, 60),
            standalone: standaloneQuery.substring(0, 60)
          }, '[TutorChain] Query rewritten to standalone');
        } catch (error) {
          this.logger.warn({ error }, '[TutorChain] Standalone rewrite failed, using original');
        }
      }
    }

    const isCurrentAffairs = subject === 'current_affairs' || subject === 'general_knowledge';
    const classifyPromise = this.classifier.classify(standaloneQuery);
    
    let retrievePromise;
    if (isCurrentAffairs) {
      retrievePromise = webSearchTool(standaloneQuery, subject || 'general_knowledge', this.secrets.tavilyApiKey, this.logger);
    } else {
      retrievePromise = this.retriever.getRelevantDocuments(standaloneQuery, { subject, level, k: 5 });
    }

    const [classification, retrievedDocs] = await Promise.all([classifyPromise, retrievePromise]);
    
    let finalDocs = retrievedDocs;
    if (!isCurrentAffairs && (retrievedDocs.length === 0 || (retrievedDocs[0]?.score ?? 0) < 0.5)) {
      this.logger.warn('[TutorChain] Fallback to web search');
      finalDocs = await webSearchTool(standaloneQuery, classification.subject, this.secrets.tavilyApiKey, this.logger);
    }

    return { classification, sources: finalDocs, cached: false };
  }

  /**
 * Run the chain
 */
async run(request: ChatRequest, sessionId?: string): Promise<AITutorResponse> {
  try {
    const input = {
      message: request.message,
      sessionId: sessionId || request.sessionId,
      subject: request.subject,
      level: request.level,
    };

    //check cache first
    this.logger.debug({ query: input.message.substring(0, 50) }, 'Checking cache before pipeline...');

    // 1. Check Direct Cache (exact match)
    const directCache = await this.redisCache.checkDirectCache(
      input.message,
      input.subject || 'general'
    );

    if (directCache) {
      this.logger.info('✓ DIRECT CACHE HIT - returning instantly');
      return {
        answer: directCache.response,
        classification: {
          subject: directCache.metadata.subject || input.subject || 'general',
          level: directCache.metadata.level || input.level || 'intermediate',
          confidence: directCache.metadata.confidence || 1.0,
        },
        cached: true,
        confidence: directCache.metadata.confidence || 1.0,
        metadata: {
          modelUsed: directCache.metadata.modelUsed,
          validationScore: 1.0,
        },
      } as AITutorResponse;
    }

    // 2. Check Semantic Cache (similarity match)
    const semanticCache = await this.redisCache.checkSemanticCache(
      input.message,
      input.subject || 'general'
    );

    if (semanticCache) {
      this.logger.info({ score: semanticCache.score }, '✓ SEMANTIC CACHE HIT - returning instantly');
      return {
        answer: semanticCache.response,
        classification: {
          subject: semanticCache.metadata.subject || input.subject || 'general',
          level: semanticCache.metadata.level || input.level || 'intermediate',
          confidence: semanticCache.metadata.confidence || semanticCache.score,
        },
        cached: true,
        confidence: semanticCache.score,
        metadata: {
          modelUsed: semanticCache.metadata.modelUsed,
          validationScore: semanticCache.score,
        },
      } as AITutorResponse;
    }

    this.logger.debug('Cache miss - running full pipeline...');

    // no cache hit - run full sequence
    const result = await this.sequence.invoke(input);

    // caching logic, store after generation
    if (result.answer) {
      const validation = validateResponse(result.answer);
      
      if (validation.isValid) {
        this.logger.info({ score: validation.score }, 'Caching response after generation');
        
        await Promise.all([
          this.redisCache.storeDirectCache(
            input.message,
            result.answer,
            result.classification.subject,
            {
              confidence: result.classification.confidence,
              modelUsed: result.metadata?.modelUsed,
              level: result.classification.level,
            }
          ),
          this.redisCache.storeSemanticCache(
            input.message,
            result.answer,
            result.classification.subject,
            {
              confidence: result.classification.confidence,
              modelUsed: result.metadata?.modelUsed,
              level: result.classification.level,
            }
          ),
        ]);
        
        this.logger.info('✓ Stored in both caches');
      } else {
        this.logger.warn({ reason: validation.reason }, 'Skipped caching - quality check failed');
      }
    }

    return result;
  } catch (error) {
    this.logger.error({ error }, '[TutorChain] Run failed');
    throw error;
  }
}

  /**
   * Stream the chain results
   */
  async runStreaming(
    request: ChatRequest,
    sessionId?: string,
    callbacks?: { onToken: (token: string) => void; onError: (error: Error) => void; onComplete: () => void }
  ) {
    try {
      const input = {
        message: request.message,
        sessionId: sessionId || request.sessionId,
        subject: request.subject,
        level: request.level,
      };

      const stream = await this.sequence.stream(input);
      
      if (callbacks?.onToken) {
        for await (const chunk of stream) {
          if (chunk?.evaluationOutput?.answer) {
            callbacks.onToken(chunk.evaluationOutput.answer);
          }
        }
      }

      if (callbacks?.onComplete) {
        callbacks.onComplete();
      }

      return stream;
    } catch (error) {
      this.logger.error({ error }, '[TutorChain] Streaming failed');
      if (callbacks?.onError) {
        callbacks.onError(error as Error);
      }
      throw error;
    }
  }

  /**
   * STREAMING EVALUATE - Generate final response after reranking with streaming
   * This is called when user clicks Evaluate button for streaming responses
   */
  async evaluateStreaming(
    userQuery: string,
    classification: Classification,
    documents: Document[],
    subscription: string = 'free',
    callbacks: {
      onToken: (token: string) => void;
      onMetadata?: (metadata: any) => void;
      onComplete: (result: EvaluatePromptOutput) => void;
      onError: (error: Error) => void;
    },
    sessionId?: string
  ): Promise<void> {
    this.logger.info(
      {
        query: userQuery.substring(0, 80),
        subject: classification.subject,
        confidence: classification.confidence,
        docCount: documents.length,
        sessionId,
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
          modelUsed: directCache.metadata.modelUsed,
          levelUsed: directCache.metadata.levelUsed,
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
          modelUsed: semanticCache.metadata.modelUsed,
          levelUsed: semanticCache.metadata.levelUsed,
          latency: 0,
        });
        return;
      }

      this.logger.debug('Cache miss - starting streaming generation...');

      // Get top document as context
      const topDocument = documents.length > 0 ? documents[0] : null;
      this.logger.debug({ hasTopDoc: !!topDocument }, '[TutorChain] Top document selected for streaming');

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
      callbacks.onMetadata?.({
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
      let finalAnswer: string | undefined;
      let capturedResult: EvaluatePromptOutput | undefined;

      const wrappedCallbacks = {
        ...callbacks,
        onMetadata: callbacks.onMetadata || (() => {}),
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
        const validation = validateResponse(finalAnswer!);

        if (validation.isValid) {
          this.logger.info({ score: validation.score.toFixed(2) }, 'Response passed quality check');

          await Promise.all([
            this.redisCache.storeDirectCache(
              userQuery,
              finalAnswer!,
              classification.subject,
              {
                confidence: classification.confidence,
                modelUsed: result.modelUsed,
                levelUsed: result.levelUsed,
              }
            ),
            this.redisCache.storeSemanticCache(
              userQuery,
              finalAnswer!,
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
            await this.chatMemory.save(sessionId, userQuery, finalAnswer!);
            this.logger.info({ sessionId }, '✓ Saved streaming conversation to memory');
          }
        } else {
          this.logger.warn(
            { reason: validation.reason, score: validation.score.toFixed(2), preview: finalAnswer!.substring(0, 100) },
            '✗ Skipped caching - response failed quality check'
          );
        }
      }
    } catch (error) {
      this.logger.error({ error }, '[TutorChain] Streaming evaluate failed');
      callbacks.onError(error as Error);
    }
  }

  async chatStream(
    request: ChatRequest,
    callbacks: {
      onToken: (token: string) => void;
      onMetadata?: (metadata: any) => void;
      onComplete: (result: EvaluatePromptOutput) => void;
      onError: (error: Error) => void;
    },
    sessionId?: string
  ): Promise<void> {
    const { message, subject, level } = request;
    this.logger.info(
      {
        query: message.substring(0, 80),
        subject: subject,
        level: level,
        sessionId,
      },
      '[TutorChain] Chat stream started'
    );

    try {
      // Manually run the initial parts of the chain
      let standaloneQuery = message;
      if (sessionId) {
        const messages = await this.chatMemory.load(sessionId, 5);
        const userQueriesOnly = this.chatMemory.formatUserQueriesOnly(messages);
        if (userQueriesOnly && userQueriesOnly.trim().length > 20) {
          try {
            standaloneQuery = await this.classifier['rewriteToStandalone'](
              message,
              userQueriesOnly
            );
            this.logger.info({
              original: message.substring(0, 60),
              standalone: standaloneQuery.substring(0, 60)
            }, '[TutorChain] Query rewritten to standalone');
          } catch (error) {
            this.logger.warn({ error }, '[TutorChain] Standalone rewrite failed, using original');
          }
        }
      }

      const isCurrentAffairs = subject === 'current_affairs' || subject === 'general_knowledge';
      const classifyPromise = this.classifier.classify(standaloneQuery);
      
      let retrievePromise;
      if (isCurrentAffairs) {
        retrievePromise = webSearchTool(standaloneQuery, subject || 'general_knowledge', this.secrets.tavilyApiKey, this.logger);
      } else {
        retrievePromise = this.retriever.getRelevantDocuments(standaloneQuery, { subject, level, k: 5 });
      }

      const [classification, retrievedDocs] = await Promise.all([classifyPromise, retrievePromise]);
      
      let finalDocs = retrievedDocs;
      if (!isCurrentAffairs && (retrievedDocs.length === 0 || (retrievedDocs[0]?.score ?? 0) < 0.5)) {
        this.logger.warn('[TutorChain] Fallback to web search');
        finalDocs = await webSearchTool(standaloneQuery, classification.subject, this.secrets.tavilyApiKey, this.logger);
      }
      
      const rerankedResults = await this.reranker.rerank(finalDocs, standaloneQuery, 3);
      
      if (callbacks.onMetadata) {
        callbacks.onMetadata({
            classification: classification,
            sources: rerankedResults,
        });
      }
      
      const rerankedDocs = rerankedResults.map(result => result.document);

      // Now, call evaluateStreaming with the results
      await this.evaluateStreaming(
        message, // pass original message to evaluate
        classification,
        rerankedDocs,
        request.userSubscription || 'free',
        callbacks,
        sessionId
      );

    } catch (error) {
      this.logger.error({ error }, '[TutorChain] Chat stream failed');
      callbacks.onError(error as Error);
    }
  }

  /**
   * For compatibility: invoke method (same as run)
   */
  async invoke(request: any): Promise<AITutorResponse> {
    return this.run(request, request.sessionId);
  }
}

export function createTutorChain(
  classifier: Classifier,
  retriever: AWSKnowledgeBaseRetriever,
  modelSelector: ModelSelector,
  reranker: Reranker,
  evaluatePrompt: EvaluatePrompt,
  chatMemory: ChatMemory,
  logger: pino.Logger,
  secrets: any 
): TutorChain {
  return new TutorChain(classifier, retriever, modelSelector, reranker, evaluatePrompt, chatMemory, logger, secrets);
}


