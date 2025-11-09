import { Document, RerankerResult } from '../types/index.js';
import { modelConfigService } from '../config/modelConfig.js';
import { ModelSelector } from '../llm/ModelSelector.js';
import pino from 'pino';

/**
 * LangChain-compatible Document Reranker
 * Implements both LLM-based cross-encoder reranking and local fallback
 */
export class Reranker {
  constructor(private modelSelector: ModelSelector, private logger: pino.Logger, private useLocalFallback = false) {}

  /**
   * Rerank documents based on relevance to query using LangChain patterns
   */
  async rerank(documents: Document[], query: string, topK: number = 5): Promise<RerankerResult[]> {
    if (documents.length === 0) return [];

    this.logger.info(`Reranking ${documents.length} documents for query, returning top ${topK}`);

    try {
      let results: RerankerResult[];
      
      if (this.useLocalFallback) {
        results = await this.localRerank(documents, query);
      } else {
        results = await this.llmRerank(documents, query);
      }
      
      // Return top K results
      const topResults = results.slice(0, topK);
      this.logger.info(`Reranking completed, returning ${topResults.length} results`);
      
      return topResults;
    } catch (error) {
      this.logger.error(error, 'Reranking failed, falling back to local reranking');
      return (await this.localRerank(documents, query)).slice(0, topK);
    }
  }

  /**
   * LLM-based reranking using cross-encoder approach
   */
  private async llmRerank(documents: Document[], query: string): Promise<RerankerResult[]> {
    this.logger.info('Using LLM-based reranking');
    
  const rerankerLLM = await this.modelSelector.getRerankerLLM();
    const rerankerConfig = modelConfigService.getRerankerConfig();
    const results: RerankerResult[] = [];

    // Process documents in batches to avoid rate limits
    const batchSize = 5;
    for (let i = 0; i < documents.length; i += batchSize) {
      const batch = documents.slice(i, i + batchSize);
      
      const batchPromises = batch.map(async (document) => {
        try {
          const prompt = this.buildCrossEncoderPrompt(query, document.text);
          const response = await rerankerLLM.generate(prompt, {
            maxTokens: rerankerConfig.maxTokens,
            temperature: rerankerConfig.temperature,
            systemPrompt: rerankerConfig.systemPrompt,
          });

          const parsed = this.parseRerankerResponse(response);
          return {
            document,
            score: parsed.score,
            reason: parsed.reason,
          } as RerankerResult;
        } catch (error) {
          this.logger.warn(error, `Failed to rerank document ${document.id}, using fallback`);
          return {
            document,
            score: this.calculateTextSimilarity(query, document.text),
            reason: 'LLM reranking failed, used similarity fallback',
          } as RerankerResult;
        }
      });

      const batchResults = await Promise.all(batchPromises);
      results.push(...batchResults);
    }

    // Sort by score descending
    return results.sort((a, b) => b.score - a.score);
  }

  /**
   * Local fallback reranking using enhanced text similarity
   */
  private async localRerank(documents: Document[], query: string): Promise<RerankerResult[]> {
    this.logger.info('Using local similarity-based reranking');

    const results: RerankerResult[] = documents.map(document => {
      const score = this.calculateEnhancedSimilarity(query, document);
      return {
        document,
        score,
        reason: 'Local enhanced similarity scoring',
      };
    });

    // Sort by score descending
    return results.sort((a, b) => b.score - a.score);
  }

  /**
   * Build cross-encoder prompt for LLM reranking
   */
  private buildCrossEncoderPrompt(query: string, passage: string): string {
    const truncatedPassage = passage.length > 1000 ? passage.substring(0, 1000) + '...' : passage;
    
    return `Query: "${query}"

Passage: "${truncatedPassage}"

Rate the relevance of this passage to the query on a scale of 0.0 to 1.0:
- 1.0: Highly relevant, directly answers the query
- 0.8-0.9: Very relevant, contains useful information
- 0.6-0.7: Moderately relevant, some useful context
- 0.4-0.5: Slightly relevant, limited usefulness
- 0.0-0.3: Not relevant or misleading

Consider:
- Semantic relevance and conceptual alignment
- Factual accuracy and completeness
- Educational value for the query context

Respond with JSON: {"score": 0.85, "reason": "brief explanation"}`;
  }

  /**
   * Parse reranker LLM response
   */
  private parseRerankerResponse(response: string): { score: number; reason: string } {
    try {
      // Clean response and extract JSON
      const cleaned = response.trim().replace(/```json|```/g, '').replace(/^\s*[\{\[]/, '').replace(/[\}\]]\s*$/, '');
      const jsonMatch = response.match(/\{[^}]+\}/);
      
      if (jsonMatch) {
        const parsed = JSON.parse(jsonMatch[0]);
        return {
          score: Math.max(0, Math.min(1, parsed.score || 0)),
          reason: parsed.reason || 'No reason provided',
        };
      } else {
        // Try to extract score from text
        const scoreMatch = response.match(/score["\s]*[:=]\s*([0-9.]+)/i);
        const score = scoreMatch ? parseFloat(scoreMatch[1]) : 0.5;
        return {
          score: Math.max(0, Math.min(1, score)),
          reason: 'Extracted from partial response',
        };
      }
    } catch (error) {
      this.logger.warn(error, 'Failed to parse reranker response');
      return { score: 0.5, reason: 'Parse error, default score' };
    }
  }

  /**
   * Calculate enhanced text similarity with multiple signals
   */
  private calculateEnhancedSimilarity(query: string, document: Document): number {
    const queryTerms = this.preprocessText(query);
    const docTerms = this.preprocessText(document.text);
    
    // Multiple similarity metrics
    const exactMatch = this.calculateExactTermOverlap(queryTerms, docTerms);
    const semanticSim = this.calculateSemanticSimilarity(queryTerms, docTerms);
    const positionBonus = this.calculatePositionBonus(query, document.text);
    const lengthPenalty = this.calculateLengthPenalty(document.text);
    const metadataBonus = this.calculateMetadataBonus(query, document.metadata);
    
    // Weighted combination
    const finalScore = (
      exactMatch * 0.4 + 
      semanticSim * 0.3 + 
      positionBonus * 0.1 + 
      metadataBonus * 0.2
    ) * lengthPenalty;
    
    return Math.max(0, Math.min(1, finalScore));
  }

  /**
   * Calculate traditional text similarity
   */
  private calculateTextSimilarity(query: string, text: string): number {
    const queryWords = query.toLowerCase().split(/\s+/);
    const textWords = text.toLowerCase().split(/\s+/);
    
    const overlap = queryWords.filter(word => 
      textWords.some(textWord => textWord.includes(word) || word.includes(textWord))
    ).length;

    const similarity = overlap / queryWords.length;
    const lengthBonus = Math.min(0.2, textWords.length / 1000);
    
    return Math.min(1, similarity + lengthBonus);
  }

  /**
   * Preprocess text for similarity calculation
   */
  private preprocessText(text: string): string[] {
    return text
      .toLowerCase()
      .replace(/[^\w\s]/g, ' ')
      .split(/\s+/)
      .filter(term => term.length > 2);
  }

  /**
   * Calculate exact term overlap
   */
  private calculateExactTermOverlap(queryTerms: string[], docTerms: string[]): number {
    const overlap = queryTerms.filter(term => docTerms.includes(term)).length;
    return queryTerms.length > 0 ? overlap / queryTerms.length : 0;
  }

  /**
   * Calculate semantic similarity (simplified)
   */
  private calculateSemanticSimilarity(queryTerms: string[], docTerms: string[]): number {
    // Simple semantic matching based on word stems and synonyms
    const stemOverlap = queryTerms.filter(qTerm => 
      docTerms.some(dTerm => 
        qTerm.startsWith(dTerm.substring(0, 4)) || 
        dTerm.startsWith(qTerm.substring(0, 4))
      )
    ).length;
    
    return queryTerms.length > 0 ? stemOverlap / queryTerms.length : 0;
  }

  /**
   * Calculate position bonus (earlier occurrences score higher)
   */
  private calculatePositionBonus(query: string, text: string): number {
    const firstOccurrence = text.toLowerCase().indexOf(query.toLowerCase());
    if (firstOccurrence === -1) return 0;
    
    const textLength = text.length;
    const positionRatio = (textLength - firstOccurrence) / textLength;
    return Math.min(0.3, positionRatio);
  }

  /**
   * Calculate length penalty (very short or very long documents score lower)
   */
  private calculateLengthPenalty(text: string): number {
    const wordCount = text.split(/\s+/).length;
    if (wordCount < 50) return 0.7; // Too short
    if (wordCount > 2000) return 0.8; // Too long
    return 1.0; // Optimal length
  }

  /**
   * Calculate metadata bonus
   */
  private calculateMetadataBonus(query: string, metadata: Record<string, any>): number {
    let bonus = 0;
    const queryLower = query.toLowerCase();
    
    // Subject/topic matching
    if (metadata.subject && queryLower.includes(metadata.subject.toLowerCase())) {
      bonus += 0.2;
    }
    
    // Source quality (if available)
    if (metadata.source === 'verified' || metadata.quality === 'high') {
      bonus += 0.1;
    }
    
    // Recency bonus for current affairs
    if (metadata.date && queryLower.includes('recent')) {
      const docDate = new Date(metadata.date);
      const daysSince = (Date.now() - docDate.getTime()) / (1000 * 60 * 60 * 24);
      if (daysSince < 30) bonus += 0.1;
    }
    
    return Math.min(0.3, bonus);
  }
}