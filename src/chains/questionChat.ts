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
   * Core retrieval logic - shared by both streaming and non-streaming
   */
  private async retrieveDocuments(
    query: string,
    classification: Classification,
    onProgress?: (message: string) => void
  ): Promise<Document[]> {
    const isAcademicSubject = this.isAcademicSubject(classification.subject);
    const isCurrentAffairs = classification.subject === 'current_affairs' || 
                             classification.subject === 'general_knowledge';

    let retrievedDocs: Document[] = [];

    if (isCurrentAffairs) {
      // ROUTE 1: Web Search for current affairs
      this.logger.info('[TutorChain] Route: Web Search (Current Affairs)');
      onProgress?.('Searching web...\n');
      retrievedDocs = await webSearchTool(query, classification.subject, this.logger);
    } else if (isAcademicSubject) {
      // ROUTE 2: Knowledge Base for academic subjects with fallback
      this.logger.info('[TutorChain] Route: Knowledge Base (Academic)');
      onProgress?.('Searching knowledge base...\n');
      
      retrievedDocs = await this.retriever.getRelevantDocuments(query, {
        subject: classification.subject,
        level: classification.level,
        k: 5,
      });

      // Check if fallback needed
      const topScore = retrievedDocs.length > 0 && retrievedDocs[0]?.score !== undefined 
        ? retrievedDocs[0].score 
        : 0;
      const shouldFallback = retrievedDocs.length === 0 || topScore < 0.5;
      
      if (shouldFallback) {
        const reason = retrievedDocs.length === 0 
          ? 'KB empty' 
          : `KB relevance too low (${(topScore * 100).toFixed(1)}%)`;
        
        this.logger.warn(`[TutorChain] ${reason}, falling back to web search`);
        onProgress?.(`${retrievedDocs.length === 0 ? 'No KB results' : `Low relevance (${(topScore * 100).toFixed(1)}%)`}, trying web search...\n`);
        
        retrievedDocs = await webSearchTool(query, classification.subject, this.logger);
      }
    } else {
      // ROUTE 3: Try KB for general subjects with fallback
      this.logger.info('[TutorChain] Route: Knowledge Base (General)');
      onProgress?.('Searching knowledge base...\n');
      
      retrievedDocs = await this.retriever.getRelevantDocuments(query, {
        subject: classification.subject,
        level: classification.level,
        k: 5,
      });

      // Check if fallback needed
      const topScore = retrievedDocs.length > 0 && retrievedDocs[0]?.score !== undefined 
        ? retrievedDocs[0].score 
        : 0;
      const shouldFallback = retrievedDocs.length === 0 || topScore < 0.5;
      
      if (shouldFallback) {
        const reason = retrievedDocs.length === 0 
          ? 'KB empty' 
          : `KB relevance too low (${(topScore * 100).toFixed(1)}%)`;
        
        this.logger.warn(`[TutorChain] ${reason}, falling back to web search`);
        onProgress?.(`${retrievedDocs.length === 0 ? 'No KB results' : `Low relevance (${(topScore * 100).toFixed(1)}%)`}, trying web search...\n`);
        
        retrievedDocs = await webSearchTool(query, classification.subject, this.logger);
      }
    }

    return retrievedDocs;
  }

  /**
   * Main execution method (non-streaming)
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

      // STEP 2: Retrieve documents using shared logic
      const retrievedDocs = await this.retrieveDocuments(
        request.message,
        classification
      );

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

      // STEP 1: Classification
      const classification = await this.classifyQuery(request);
      handlers.onMetadata({ classification, step: 'classification' });
      handlers.onToken(`Subject: ${classification.subject} (${(classification.confidence * 100).toFixed(0)}%)\n`);

      // STEP 2: Retrieve documents using shared logic with progress updates
      const retrievedDocs = await this.retrieveDocuments(
        request.message,
        classification,
        (message) => handlers.onToken(message) // Pass progress callback
      );

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
