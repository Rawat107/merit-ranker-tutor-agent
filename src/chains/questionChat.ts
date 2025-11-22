
import { RunnableSequence } from "@langchain/core/runnables";
import { traceable } from "langsmith/traceable";
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
  private classifyAndRetrieveChain: RunnableSequence;
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
    
    // Build the RunnableSequences
    this.classifyAndRetrieveChain = this.buildClassifyAndRetrieveChain();
    this.sequence = this.buildSequence();
  }

  private buildClassifyAndRetrieveChain(): RunnableSequence<any, any> {
    const historyAndPassthrough = {
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
          const userQueriesOnly = this.chatMemory.formatUserQueriesOnly(messages);
          this.logger.debug('[TutorChain] Loaded user queries for standalone rewrite');
          return userQueriesOnly;
        } catch (error) {
          this.logger.warn({ error, sessionId: input.sessionId }, '[TutorChain] Failed to load conversation history');
          return '';
        }
      },
    };

    const classifyAndRetrieveStep = async (input: any) => {
      try {
        // Step 1: Rewrite query to be standalone (if history is present)
        let standaloneQuery = input.message;
        if (input.conversationHistory && input.conversationHistory.trim().length > 20) {
          try {
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
          }
        } else {
          this.logger.debug('[TutorChain] No history or too short, using original query');
        }

        // Step 2: Classify the standalone query to determine intent
        const classification = await this.classifier.classify(standaloneQuery);
        this.logger.info({
          subject: classification.subject,
          level: classification.level,
          intent: classification.intent,
          confidence: classification.confidence
        }, '[TutorChain] Classification complete');

        let retrievedDocs: Document[] = [];
        const smallTextIntents = ['summarize', 'change_tone', 'proofread', 'make_email_professional'];
        const urlRegex = /https?:\/\/[^\s]+/g;
        const containsUrl = urlRegex.test(standaloneQuery);

        // Step 3: Decide on retrieval strategy based on intent
        if (classification.intent && smallTextIntents.includes(classification.intent)) {
          // This is a small text task. Only use web search if there's a URL.
          this.logger.info({ intent: classification.intent }, '[TutorChain] Small text intent detected.');
          if (containsUrl) {
            this.logger.info('[TutorChain] URL detected in small text task, using web search.');
            retrievedDocs = await webSearchTool(
              standaloneQuery,
              classification.subject,
              this.secrets.tavilyApiKey,
              this.logger
            );
          } else {
            this.logger.info('[TutorChain] No URL in small text task, skipping retrieval.');
            // No retrieval needed, the text to be processed is in the query itself.
            retrievedDocs = [];
          }
        } else {
          // This is a knowledge-based question, use the standard retrieval logic.
          this.logger.info('[TutorChain] Knowledge-based intent detected, proceeding with standard retrieval.');
          const isCurrentAffairs = classification.subject === 'current_affairs' || classification.subject === 'general_knowledge';

          if (isCurrentAffairs) {
            this.logger.info('[TutorChain] Current affairs/GK subject, using web search.');
            retrievedDocs = await webSearchTool(
              standaloneQuery,
              classification.subject,
              this.secrets.tavilyApiKey,
              this.logger
            );
          } else {
            this.logger.info('[TutorChain] Using AWS KB Retriever.');
            retrievedDocs = await this.retriever.getRelevantDocuments(
              standaloneQuery,
              {
                subject: classification.subject,
                level: classification.level,
                k: 5,
              }
            );

            // Fallback to web search if KB retrieval is poor
            if (retrievedDocs.length === 0 || (retrievedDocs[0]?.score ?? 0) < 0.5) {
              this.logger.warn('[TutorChain] Fallback to web search due to poor KB results.');
              retrievedDocs = await webSearchTool(
                standaloneQuery,
                classification.subject,
                this.secrets.tavilyApiKey,
                this.logger
              );
            }
          }
        }
        
        this.logger.info({ count: retrievedDocs.length }, '[TutorChain] Retrieval complete');
        
        return { ...input, classification, retrievedDocs, standaloneQuery };
      } catch (error) {
        this.logger.error({ error, query: input.message }, '[TutorChain] Classify/retrieve step failed');
        throw error;
      }
    };

    return RunnableSequence.from([
      historyAndPassthrough,
      classifyAndRetrieveStep,
    ]).withConfig({
      runName: 'ClassifyAndRetrieveChain',
    }) as RunnableSequence<any, any>;
  }

  private _rerankAndSelectModel = traceable(async (input: any) => {
    const queryForRerank = input.standaloneQuery || input.message;
    let finalDocs = input.retrievedDocs || [];

    const shouldRerank = finalDocs.length > 0 && (finalDocs[0].score || 0) < 0.85;

    if (shouldRerank) {
        this.logger.info('[TutorChain] Reranking documents...');
        finalDocs = await this.reranker.rerank(finalDocs, queryForRerank, 3);
    } else {
        this.logger.info('[TutorChain] Skipping rerank - high-quality docs');
        finalDocs = finalDocs.slice(0, 3);
    }
    
    return { ...input, rerankedDocs: finalDocs };
  }, { name: "Rerank" });

  private _evaluate = traceable(async (input: any) => {
    const topDocument = input.rerankedDocs[0]?.document || null;
      
    const evaluateInput: EvaluatePromptInput = {
        userQuery: input.message,
        classification: input.classification,
        topDocument,
        userPrefs: USER_PREFS,
        subscription: 'free',
        conversationHistory: input.conversationHistory,
    };

    const evaluationOutput = await this.evaluatePrompt.evaluate(evaluateInput);
    return { ...input, evaluationOutput };
  }, { name: "Evaluate" });

  private _finalize = traceable(async (input: any) => {
    try {
        const { evaluationOutput } = input;
        const validation = validateResponse(evaluationOutput.answer);

        if (validation.isValid && input.sessionId) {
            await this.chatMemory.save(input.sessionId, input.message, evaluationOutput.answer);
            this.logger.info({ sessionId: input.sessionId }, '[TutorChain] Saved to memory');
        }

        const response: AITutorResponse = {
            answer: evaluationOutput.answer,
            sources: input.rerankedDocs,
            classification: input.classification,
            cached: false,
            confidence: input.classification.confidence,
            metadata: {
                modelUsed: evaluationOutput.modelUsed,
                validationScore: validation.score,
            },
        };

        return response;
    } catch (error) {
        this.logger.error({ error }, '[TutorChain] Finalize failed');
        throw error;
    }
  }, { name: "Finalize" });

  /**
   * Build the RunnableSequence with all steps
   */
  private buildSequence(): RunnableSequence<any, any> {
    return RunnableSequence.from([
      this.classifyAndRetrieveChain,
      this._rerankAndSelectModel,
      this._evaluate,
      this._finalize,
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

    const result = await this.classifyAndRetrieveChain.invoke({ message, subject, level, sessionId });
    
    return { 
        classification: result.classification, 
        sources: result.retrievedDocs, 
        cached: false 
    };
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

      // First, check cache for a quick win
      this.logger.debug({ query: input.message.substring(0, 50) }, '[TutorChain] Checking cache before pipeline...');
      const cacheResult = await this.redisCache.checkUnified(input.message, input.subject || 'general');

      if (cacheResult.hit) {
        this.logger.info(`✓ ${cacheResult.type.toUpperCase()} CACHE HIT - returning instantly`);
        return {
          answer: cacheResult.response!,
          classification: {
            subject: cacheResult.metadata?.subject || input.subject || 'general',
            level: cacheResult.metadata?.level || input.level || 'intermediate',
            confidence: cacheResult.metadata?.confidence || cacheResult.score || 1.0,
          },
          cached: true,
          confidence: cacheResult.metadata?.confidence || cacheResult.score || 1.0,
          metadata: {
            modelUsed: cacheResult.metadata?.modelUsed,
            validationScore: cacheResult.score || 1.0,
            cacheType: cacheResult.type,
          },
        } as AITutorResponse;
      }
      this.logger.debug('Cache miss - proceeding with classification...');

      // --- Start of Fast Path Logic ---
      
      // Step 1: Get history and rewrite the query to be standalone.
      const conversationHistory = input.sessionId ? await this.chatMemory.load(input.sessionId, 5).then(msgs => this.chatMemory.formatUserQueriesOnly(msgs)) : '';
      const standaloneQuery = await this.classifier['rewriteToStandalone'](input.message, conversationHistory);
      
      // Step 2: Classify the standalone query.
      const classification = await this.classifier.classify(standaloneQuery);
      
      const smallTextIntents = ['summarize', 'change_tone', 'proofread', 'make_email_professional'];
      
      // Step 3: Check if the intent is a "small text" task.
      if (classification.intent && smallTextIntents.includes(classification.intent)) {
        this.logger.info({ intent: classification.intent }, '[TutorChain] Fast path: Small text intent detected. Using classifier LLM directly.');

        // Get the classifier's own LLM (the small, fast model).
        const llm = await this.modelSelector.getClassifierLLM();
        this.logger.info({ modelUsed: (llm as any).modelId }, '[TutorChain] Classifier LLM selected for fast path execution.');
        
        // Simple prompt for the task.
        const fastPathPrompt = `You are a helpful assistant. Please ${classification.intent.replace(/_/g, ' ')} the following text:\n\n${standaloneQuery}`;
        
        // Invoke the model directly.
        const response = await llm.generate(fastPathPrompt);
 
        const answer = typeof response === 'string' ? response : JSON.stringify(response);

        // Construct a valid AITutorResponse and return.
        const fastPathResponse: AITutorResponse = {
          answer,
          classification,
          cached: false,
          confidence: classification.confidence,
          sources: [], // No sources for this path
          metadata: {
            modelUsed: (llm as any).modelId || 'classifier-model',
            validationScore: 1,
            fastPath: true
          },
        };

        // Save to memory and cache
        await this.chatMemory.save(input.sessionId!, input.message, answer!);
        await this.redisCache.storeDirectCache(input.message, answer, classification.subject, { confidence: classification.confidence, modelUsed: fastPathResponse.metadata?.modelUsed, level: classification.level });

        return fastPathResponse;
      }

      // --- End of Fast Path Logic ---

      // If it's not a small text task, proceed with the full sequence.
      this.logger.info({ intent: classification.intent }, '[TutorChain] Standard Path: Intent is not a small text task, running full pipeline...');
      const result = await this.sequence.invoke(input);

      // Caching logic for the full pipeline result.
      if (result.answer) {
        const validation = validateResponse(result.answer);
        if (validation.isValid) {
          this.logger.info({ score: validation.score }, 'Caching full pipeline response...');
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
      this.logger.debug('[TutorChain] Checking cache before streaming...');

      const cacheResult = await this.redisCache.checkUnified(
        userQuery,
        classification.subject
      );

      if (cacheResult.hit) {
        this.logger.info(
          { score: cacheResult.score },
          `✓ Returning from ${cacheResult.type.toUpperCase()} cache (streaming)`
        );
        callbacks.onToken(cacheResult.response!);
        callbacks.onComplete({
          answer: cacheResult.response!,
          modelUsed: cacheResult.metadata?.modelUsed,
          levelUsed: cacheResult.metadata?.levelUsed,
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
    const traceableChatStream = traceable(async (req: ChatRequest, sessId?: string) => {
        const { message, subject, level } = req;
        this.logger.info(
          {
            query: message.substring(0, 80),
            subject: subject,
            level: level,
            sessionId: sessId,
          },
          '[TutorChain] Chat stream started'
        );

        try {
          // Use the chain for classification and retrieval
          const classifyAndRetrieveResult = await this.classifyAndRetrieveChain.invoke({ message, subject, level, sessionId: sessId });
          const { classification, retrievedDocs, standaloneQuery } = classifyAndRetrieveResult;

          // Rerank documents
          const rerankedResults = await this.reranker.rerank(retrievedDocs, standaloneQuery, 3);
          
          if (callbacks.onMetadata) {
            callbacks.onMetadata({
                classification: classification,
                sources: rerankedResults,
            });
          }
          
          const rerankedDocs = rerankedResults.map(result => result.document);

          // Evaluate and stream the final answer
          await this.evaluateStreaming(
            message, // pass original message to evaluate
            classification,
            rerankedDocs,
            req.userSubscription || 'free',
            callbacks,
            sessId
          );

        } catch (error) {
          this.logger.error({ error }, '[TutorChain] Chat stream failed');
          callbacks.onError(error as Error);
        }
    }, { name: 'TutorStreamingChain' });
    
    await traceableChatStream(request, sessionId);
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


