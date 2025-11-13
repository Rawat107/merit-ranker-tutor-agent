import { ChatRequest, AITutorResponse, Classification, Document } from '../types/index.js';
import { Classifier } from '../classifier/Classifier.js';
import { AWSKnowledgeBaseRetriever } from '../retriever/AwsKBRetriever.js';
import { ModelSelector } from '../llm/ModelSelector.js';
import { Reranker } from '../reranker/Reranker.js';
import { webSearchTool } from '../tools/webSearch.js';
import { EvaluatePrompt, EvaluatePromptInput, EvaluatePromptOutput } from '../prompts/evaluatorPrompt.js';
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
      onProgress?.('Searching web...\\n');
      retrievedDocs = await webSearchTool(query, classification.subject, this.logger);
      
      if (retrievedDocs.length === 0) {
        this.logger.warn('[TutorChain] Web search returned no results - will bypass reranking');
      }
      
      return retrievedDocs;
    }

    // ROUTE 2 & 3: Try KB first, fallback to web search if needed
    this.logger.info('[TutorChain] Route: Knowledge Base');
    onProgress?.('Searching knowledge base...\\n');
    
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
            ? 'No KB results'
            : `Low relevance (${(topScore * 100).toFixed(1)}%)`
        }, trying web search...\\n`
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
   */
  private async execute(request: ChatRequest): Promise<AITutorResponse> {
    this.logger.info(
      { query: request.message.substring(0, 100) },
      '[TutorChain] Processing query'
    );

    try {
      // STEP 1: Classification
      const classification = await this.classifyQuery(request);

      this.logger.info(
        {
          subject: classification.subject,
          confidence: classification.confidence,
          intent: (classification as any).intent,
        },
        '[TutorChain] Classification complete'
      );

      // STEP 2: Retrieve documents
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
    }
  ): Promise<void> {
    this.logger.info(
      {
        query: userQuery.substring(0, 80),
        subject: classification.subject,
        confidence: classification.confidence,
        docCount: documents.length,
      },
      '[TutorChain] Streaming evaluate step started'
    );

    try {
      // Get top document as context
      const topDocument = documents.length > 0 ? documents[0] : null;

      this.logger.debug(
        { hasTopDoc: !!topDocument },
        '[TutorChain] Top document selected for streaming'
      );

      // Send initial metadata
      callbacks.onMetadata({
        step: 'preparation',
        docCount: documents.length,
        classification,
      });

      // Call EvaluatePrompt with streaming
      const evaluateInput: EvaluatePromptInput = {
        userQuery,
        classification,
        topDocument,
        userPrefs: USER_PREFS,
        subscription,
      };

      await this.evaluatePrompt.evaluateStreaming(evaluateInput, callbacks);
    } catch (error) {
      this.logger.error({ error }, '[TutorChain] Streaming evaluate failed');
      callbacks.onError(error as Error);
    }
  }

  /**
   * EVALUATE: Generate final response after reranking
   * This is called when user clicks "Evaluate" button
   */
  async evaluate(
    userQuery: string,
    classification: Classification,
    documents: Document[],
    subscription?: string
  ): Promise<EvaluatePromptOutput> {
    this.logger.info(
      {
        query: userQuery.substring(0, 80),
        subject: classification.subject,
        confidence: classification.confidence,
        docCount: documents.length,
      },
      '[TutorChain] Evaluate step started'
    );

    try {
      // Get top document as shot
      const topDocument = documents.length > 0 ? documents[0] : null;

      this.logger.debug(
        { hasTopDoc: !!topDocument },
        '[TutorChain] Top document selected'
      );

      // Call EvaluatePrompt with all context
      const evaluateInput: EvaluatePromptInput = {
        userQuery,
        classification,
        topDocument,
        userPrefs: USER_PREFS,
        subscription,
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
   */
  async run(request: ChatRequest): Promise<AITutorResponse> {
    return this.execute(request);
  }

  /**
   * Streaming version
   */
  async runStreaming(request: ChatRequest, handlers: StreamingHandlers): Promise<void> {
    try {
      handlers.onToken('[AI Tutor]\\n');

      const classification = await this.classifyQuery(request);
      handlers.onMetadata({ classification, step: 'classification' });
      handlers.onToken(
        `Subject: ${classification.subject} (${(classification.confidence * 100).toFixed(0)}%)\\n`
      );

      const retrievedDocs = await this.retrieveDocuments(
        request.message,
        classification,
        (message) => handlers.onToken(message)
      );

      handlers.onMetadata({ sources: retrievedDocs.length, step: 'retrieval' });
      handlers.onToken(`Found ${retrievedDocs.length} results\\n`);

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