
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
            const formatted = this.chatMemory.format(messages, false);
            this.logger.debug('[TutorChain] Loaded conversation history');
            return formatted.historyText;
          } catch (error) {
            this.logger.warn({ error, sessionId: input.sessionId }, '[TutorChain] Failed to load conversation history');
            return '';
          }
        },
      },

      async (input: any) => {
      try {
        // Launch both classification and (conditional) document retrieval in parallel
        const isCurrentAffairs =
          input.subject === 'current_affairs' || input.subject === 'general_knowledge';

        const classifyPromise = this.classifier.classify(
          input.message,
          input.conversationHistory
        );

        let retrievePromise;
        if (isCurrentAffairs) {
          retrievePromise = webSearchTool(
            input.message,
            input.subject || 'general_knowledge',
            this.secrets.tavilyApiKey,
            this.logger
          );
        } else {
          retrievePromise = this.retriever.getRelevantDocuments(
            input.message,
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

        // Fallback to web search if KB retrieval weak, as in your original logic
        let finalDocs = retrievedDocs;
        if (!isCurrentAffairs && (retrievedDocs.length === 0 || (retrievedDocs[0]?.score ?? 0) < 0.5)) {
          this.logger.warn('[TutorChain] Fallback to web search');
          finalDocs = await webSearchTool(
            input.message,
            classification.subject,
            this.secrets.tavilyApiKey,
            this.logger
          );
        }

        this.logger.info({ subject: classification.subject, level: classification.level, confidence: classification.confidence }, '[TutorChain] Classification complete');
        this.logger.info({ count: finalDocs.length }, '[TutorChain] Retrieval complete');
        return { ...input, classification, retrievedDocs: finalDocs };
      } catch (error) {
        this.logger.error({ error, query: input.message }, '[TutorChain] Parallel classify/retrieve failed');
        throw error;
      }
    },

      // STEP 4: Rerank documents (documents-first)
      async (input: any) => {
        try {
          const docs = input.retrievedDocs || []; // defensive
          if (docs.length === 0) {
            this.logger.warn('[TutorChain] No documents to rerank');
            return { ...input, rerankedDocs: [] };
          }

          this.logger.info({ docCount: docs.length }, '[TutorChain] Reranking documents');

          // documents, query, topK
          const reranked = await this.reranker.rerank(
            docs,
            input.message,
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


      // STEP 5: Evaluate and generate response
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

      // STEP 6: Save to memory and validate
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
          modelUsed: directCache.metadata.modelUsed || 'cached',
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
          modelUsed: semanticCache.metadata.modelUsed || 'cached',
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


