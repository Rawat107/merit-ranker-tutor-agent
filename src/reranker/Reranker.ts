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

  async rerank(
    documents: Document[],
    query: string,
    topK = 3
  ): Promise<RerankerResult[]> {
    this.logger.info(
      { docCount: documents.length, topK, query: query.substring(0, 50) },
      '[Reranker]  Starting reranking'
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
            numberOfResults: documents.length, // Get scores for all documents
            modelConfiguration: {
              modelArn: 'arn:aws:bedrock:ap-northeast-1::foundation-model/amazon.rerank-v1:0',
            },
          },
        },
      };

      this.logger.info('[Reranker]  Sending RerankCommand request to Bedrock');
      const command = new RerankCommand(input);
      const response = await this.client.send(command);

      // Map response to RerankerResult
      const results: RerankerResult[] =
        response.results?.map((r) => {
          const originalDoc = documents[r.index ?? 0];
          const score = r.relevanceScore ?? 0;
          
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
        },
        '[Reranker]  Reranking complete'
      );

      return sorted;
    } catch (error) {
      this.logger.error({ error }, '[Reranker]  Error during reranking');
      
      // Fallback: return documents with score 0.5
      return documents.slice(0, topK).map((doc) => ({
        document: doc,
        score: 0.5,
        reason: 'Fallback (rerank error)',
      }));
    }
  }
}
