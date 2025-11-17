import { CohereRerank } from '@langchain/cohere';
import { traceable } from 'langsmith/traceable';
import { Document as LangChainDocument } from '@langchain/core/documents';
import { Document, RerankerResult } from '../types/index.js';
import pino from 'pino';

/**
 * Cohere Reranker using LangChain
 * Simple, fast, and returns clean data with relevance scores
 */
export class Reranker {
  private cohereRerank: CohereRerank;

  constructor(private logger: pino.Logger, private cohereApiKey: string) {
    this.logger = logger;

    // Initialize Cohere Rerank
    this.cohereRerank = new CohereRerank({
      apiKey: this.cohereApiKey,
      model: 'rerank-english-v3.0', // Latest model
      topN: 10, // We'll filter this later based on topK parameter
    });

    this.logger.info(
      {
        model: 'rerank-english-v3.0',
        provider: 'Cohere',
      },
      '[Reranker] Initialized Cohere Reranker'
    );
  }

  rerank = traceable(
    async (
    documents: Document[],
    query: string,
    topK = 3
  ): Promise<RerankerResult[]> => {
    this.logger.info(
      { docCount: documents.length, topK, query: query.substring(0, 50) },
      '[Reranker] Starting reranking'
    );

    if (!documents || documents.length === 0) {
      this.logger.warn('[Reranker] No documents to rerank');
      return [];
    }

    try {
      // Convert our Document type to LangChain Document format
      const langchainDocs = documents.map(
        (doc) =>
          new LangChainDocument({
            pageContent: doc.text,
            metadata: doc.metadata || {},
          })
      );

      // Use compressDocuments to get reranked results with scores
      const rerankedDocs = await this.cohereRerank.compressDocuments(
        langchainDocs,
        query
      );

      this.logger.info(
        { rerankedCount: rerankedDocs.length },
        '[Reranker] Cohere reranking complete'
      );

      // Map back to our RerankerResult format
      const results: RerankerResult[] = rerankedDocs.map((doc) => {
        const score = doc.metadata.relevanceScore || 0;

        // Generate reason based on relevance score
        let reason = '';
        if (score >= 0.9) {
          reason = 'Highly relevant - Directly answers the query';
        } else if (score >= 0.7) {
          reason = 'Very relevant - Contains key information';
        } else if (score >= 0.5) {
          reason = 'Moderately relevant - Partially addresses query';
        } else if (score >= 0.3) {
          reason = 'Somewhat relevant - Related information';
        } else {
          reason = 'Low relevance - Minimal connection';
        }

        // Find original document by matching text
        const originalDoc = documents.find((d) => d.text === doc.pageContent) || {
          id: 'unknown',
          text: doc.pageContent,
          metadata: doc.metadata,
          score: score,
        };

        return {
          document: originalDoc,
          score,
          reason: `${reason} (${(score * 100).toFixed(1)}%)`,
        };
      });

      // Take only topK results (Cohere already returns sorted by score)
      const topResults = results.slice(0, topK);

      this.logger.info(
        {
          originalCount: documents.length,
          rerankedCount: topResults.length,
          topScore: topResults[0] ? (topResults[0].score * 100).toFixed(1) + '%' : 'N/A',
        },
        '[Reranker] Reranking complete'
      );

      return topResults;
    } catch (error) {
      this.logger.error({ error }, '[Reranker] Error during reranking');
      // Fallback: return documents with score 0.5
      return documents.slice(0, topK).map((doc) => ({
        document: doc,
        score: 0.5,
        reason: 'Fallback (rerank error)',
      }));
    }
  }, { name: 'CohereReranker', run_type: 'chain' }
  )
}