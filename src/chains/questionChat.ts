import { ChatRequest, AITutorResponse, Classification, Document } from '../types/index.js';
import { Classifier } from '../classifier/Classifier.js';
import { AWSKnowledgeBaseRetriever } from '../retriever/AwsKBRetriever.js';
import { ModelSelector } from '../llm/ModelSelector.js';
import { webSearchTool } from '../tools/webSearch.js';
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
  private logger: pino.Logger;

  constructor(
    classifier: Classifier,
    retriever: AWSKnowledgeBaseRetriever,
    modelSelector: ModelSelector,
    logger: pino.Logger
  ) {
    this.classifier = classifier;
    this.retriever = retriever;
    this.modelSelector = modelSelector;
    this.logger = logger;
  }

  /**
   * Main execution method
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
        { subject: classification.subject, confidence: classification.confidence },
        '[TutorChain] Classification: ' + classification.subject
      );

      // STEP 2: Routing Logic
      let retrievedDocs: Document[] = [];

      // Check subject - KB FIRST for academic subjects
      const isAcademicSubject = this.isAcademicSubject(classification.subject);
      const isCurrentAffairs = classification.subject === 'current_affairs' || 
                               classification.subject === 'general_knowledge';

      if (isCurrentAffairs) {
        // ROUTE 1: Web Search for current affairs
        this.logger.info('[TutorChain] Route: Web Search (Current Affairs)');
        retrievedDocs = await webSearchTool(
          request.message,
          classification.subject,
          this.logger
        );
      } else if (isAcademicSubject) {
        // ROUTE 2: Knowledge Base for academic subjects (FIRST)
        this.logger.info('[TutorChain] Route: Knowledge Base (Academic)');
        retrievedDocs = await this.retriever.getRelevantDocuments(
          request.message,
          {
            subject: classification.subject,
            level: classification.level,
            k: 5,
          }
        );

        // Fallback to web search if KB has no results
        if (retrievedDocs.length === 0) {
          this.logger.warn('[TutorChain] KB empty, falling back to web search');
          retrievedDocs = await webSearchTool(
            request.message,
            classification.subject,
            this.logger
          );
        }
      } else {
        // ROUTE 3: Try KB for general subjects
        this.logger.info('[TutorChain] Route: Knowledge Base (General)');
        retrievedDocs = await this.retriever.getRelevantDocuments(
          request.message,
          {
            subject: classification.subject,
            level: classification.level,
            k: 5,
          }
        );

        // Fallback to web search if no KB results
        if (retrievedDocs.length === 0) {
          this.logger.warn('[TutorChain] KB empty, falling back to web search');
          retrievedDocs = await webSearchTool(
            request.message,
            classification.subject,
            this.logger
          );
        }
      }

      this.logger.info(
        { sourcesCount: retrievedDocs.length },
        '[TutorChain] Complete'
      );

      return {
        answer: '',
        sources: retrievedDocs,
        classification,
        cached: false,
        confidence: classification.confidence,
      };
    } catch (error) {
      this.logger.error(error, '[TutorChain] Error');
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

  async run(request: ChatRequest): Promise<AITutorResponse> {
    return this.execute(request);
  }

  async runStreaming(
    request: ChatRequest,
    handlers: StreamingHandlers
  ): Promise<void> {
    try {
      handlers.onToken('[AI Tutor]\n');

      const classification = await this.classifyQuery(request);
      handlers.onMetadata({ classification, step: 'classification' });
      handlers.onToken(`Subject: ${classification.subject} (${(classification.confidence * 100).toFixed(0)}%)\n`);

      let retrievedDocs: Document[] = [];
      const isAcademicSubject = this.isAcademicSubject(classification.subject);
      const isCurrentAffairs = classification.subject === 'current_affairs' ||
                               classification.subject === 'general_knowledge';

      if (isCurrentAffairs) {
        handlers.onToken('Searching web...\n');
        retrievedDocs = await webSearchTool(
          request.message,
          classification.subject,
          this.logger
        );
      } else if (isAcademicSubject) {
        handlers.onToken('Searching knowledge base...\n');
        retrievedDocs = await this.retriever.getRelevantDocuments(
          request.message,
          {
            subject: classification.subject,
            level: classification.level,
            k: 5,
          }
        );

        if (retrievedDocs.length === 0) {
          handlers.onToken('No KB results, trying web search...\n');
          retrievedDocs = await webSearchTool(
            request.message,
            classification.subject,
            this.logger
          );
        }
      } else {
        handlers.onToken('Searching knowledge base...\n');
        retrievedDocs = await this.retriever.getRelevantDocuments(
          request.message,
          {
            subject: classification.subject,
            level: classification.level,
            k: 5,
          }
        );

        if (retrievedDocs.length === 0) {
          handlers.onToken('No KB results, trying web search...\n');
          retrievedDocs = await webSearchTool(
            request.message,
            classification.subject,
            this.logger
          );
        }
      }

      handlers.onMetadata({ sources: retrievedDocs.length, step: 'retrieval' });
      handlers.onToken(`Found ${retrievedDocs.length} results\n`);

      handlers.onComplete({
        answer: '',
        sources: retrievedDocs,
        classification,
        cached: false,
        confidence: classification.confidence,
      });

      this.logger.info('[TutorChain-Stream] Complete');
    } catch (error) {
      this.logger.error(error, '[TutorChain-Stream] Error');
      handlers.onError(error as Error);
    }
  }

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
