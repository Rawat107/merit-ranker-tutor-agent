import {
  BedrockAgentRuntimeClient,
  RerankCommand,
  RerankCommandInput,
} from '@aws-sdk/client-bedrock-agent-runtime';
import { Document, RerankerResult } from '../types/index.js';
import { modelConfigService } from '../config/modelConfig.js';
import pino from 'pino';

/**
 * AWS Bedrock Reranker using native RerankCommand
 * Config-driven, uses modelConfigService
 * 
 * Enhanced: Detects direct factual questions and boosts scores appropriately
 */
export class Reranker {
  private client: BedrockAgentRuntimeClient;
  private rerankerConfig: any;

  constructor(private logger: pino.Logger) {
    this.logger = logger;
    this.rerankerConfig = modelConfigService.getRerankerConfig();

    // Initialize Bedrock Agent Runtime client
    this.client = new BedrockAgentRuntimeClient({
      region: 'ap-northeast-1', // amazon.rerank-v1:0 is available here
      credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID || '',
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || '',
      },
    });

    this.logger.info(
      {
        modelId: this.rerankerConfig.modelId,
        region: 'ap-northeast-1',
      },
      '[Reranker] Initialized BedrockAgentRuntimeClient'
    );
  }

  /**
   * Detect if this is a direct factual question
   * e.g., "When did World War 2 happen?", "How long is Japan's longest road?"
   */
  private isDirectFactualQuestion(query: string): boolean {
    const directPatterns = [
      /^(when|where|what|who|how many|how long|how far|how old|what year|what time|what date)\b/i,
      /\b(happened|occurred|was born|died|was founded|was built|is located|took place)\b.*\?$/i,
    ];
    
    return directPatterns.some(pattern => pattern.test(query));
  }

  /**
   * Extract key entities/facts from query for better matching
   * e.g., "When did World War 2 happen?" → ["world war 2", "happen", "when"]
   */
  private extractQueryEntities(query: string): string[] {
    const entities: string[] = [];
    
    // Remove question marks and extra spaces
    const cleaned = query.replace(/[?!.]/g, '').trim();
    
    // Extract key noun phrases and dates
    const words = cleaned.toLowerCase().split(/\s+/);
    
    // Combine consecutive words that might be proper nouns (World War 2)
    for (let i = 0; i < words.length; i++) {
      // Add individual words
      if (words[i].length > 3) entities.push(words[i]);
      
      // Add pairs of words (for multi-word entities)
      if (i < words.length - 1) {
        const pair = `${words[i]} ${words[i + 1]}`;
        if (pair.length > 5) entities.push(pair);
      }
    }
    
    return [...new Set(entities)]; // Remove duplicates
  }

  /**
   * Check how well a document matches the query entities
   * Used for boosting direct factual questions
   */
  private calculateEntityMatchScore(docText: string, queryEntities: string[]): number {
    const docLower = docText.toLowerCase();
    const matches = queryEntities.filter(entity => docLower.includes(entity));
    return matches.length > 0 ? matches.length / queryEntities.length : 0;
  }

  async rerank(
    documents: Document[],
    query: string,
    topK = 3
  ): Promise<RerankerResult[]> {
    this.logger.info(
      { docCount: documents.length, topK, query: query.substring(0, 50) },
      '[Reranker] Starting reranking'
    );

    if (!documents || documents.length === 0) {
      this.logger.warn('[Reranker] No documents to rerank');
      return [];
    }

    try {
      const input: RerankCommandInput = {
        queries: [
          {
            type: 'TEXT',
            textQuery: {
              text: query,
            },
          },
        ],
        sources: documents.map((doc) => ({
          type: 'INLINE',
          inlineDocumentSource: {
            type: 'TEXT',
            textDocument: {
              text: doc.text.substring(0, 2000), // Limit to 2000 chars
            },
          },
        })),
        rerankingConfiguration: {
          type: 'BEDROCK_RERANKING_MODEL',
          bedrockRerankingConfiguration: {
            numberOfResults: documents.length,
            modelConfiguration: {
              modelArn: 'arn:aws:bedrock:ap-northeast-1::foundation-model/amazon.rerank-v1:0',
            },
          },
        },
      };

      this.logger.info('[Reranker] Sending RerankCommand request to Bedrock');
      const command = new RerankCommand(input);
      const response = await this.client.send(command);

      // Detect if this is a direct factual question
      const isDirectQuestion = this.isDirectFactualQuestion(query);
      const queryEntities = this.extractQueryEntities(query);

      this.logger.debug(
        { isDirectQuestion, entityCount: queryEntities.length },
        '[Reranker] Query analysis'
      );

      // Map response to RerankerResult with entity matching boost
      const results: RerankerResult[] =
        response.results?.map((r) => {
          const originalDoc = documents[r.index ?? 0];
          let score = r.relevanceScore ?? 0;

          // Boost scores for direct factual questions that match key entities
          if (isDirectQuestion && score < 0.9) {
            const entityMatch = this.calculateEntityMatchScore(originalDoc.text, queryEntities);
            
            // If document contains key query entities, boost the score
            if (entityMatch > 0.5) {
              const boost = entityMatch * 0.3; // Up to 30% boost
              score = Math.min(0.95, score + boost);
              
              this.logger.debug(
                { 
                  originalScore: r.relevanceScore,
                  boostedScore: score.toFixed(3),
                  entityMatch: entityMatch.toFixed(2),
                  docId: originalDoc.id
                },
                '[Reranker] Applied entity match boost'
              );
            }
          }

          // Generate reason based on relevance score
          let reason = '';
          if (score >= 0.9) {
            reason = 'Highly relevant - Directly answers the query';
          } else if (score >= 0.7) {
            reason = 'Very relevant - Contains key information related to query';
          } else if (score >= 0.5) {
            reason = 'Moderately relevant - Partially addresses the query';
          } else if (score >= 0.3) {
            reason = 'Somewhat relevant - Contains related information';
          } else {
            reason = 'Low relevance - Minimal connection to query';
          }

          return {
            document: originalDoc,
            score,
            reason: `${reason} (score: ${score.toFixed(3)})`,
          };
        }) || [];

      // Sort by descending score and return only topK results
      const sorted = results.sort((a, b) => b.score - a.score).slice(0, topK);

      this.logger.info(
        {
          originalCount: documents.length,
          rerankedCount: sorted.length,
          topScore: sorted[0]?.score.toFixed(3),
          isDirectQuestion,
        },
        '[Reranker] Reranking complete'
      );

      return sorted;
    } catch (error) {
      this.logger.error({ error }, '[Reranker] Error during reranking');
      // Fallback: return documents with score 0.5
      return documents.slice(0, topK).map((doc) => ({
        document: doc,
        score: 0.5,
        reason: 'Fallback (rerank error)',
      }));
    }
  }
}