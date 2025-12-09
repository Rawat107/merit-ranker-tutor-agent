// cache/RedisCache.ts

import { BedrockEmbeddings } from '@langchain/aws';
import { createClient, RedisClientType } from 'redis';
import pino from 'pino';
import { appConfig } from '../config/modelConfig.js';

export interface CacheEntry {
  response: string;
  metadata: Record<string, any>;
  timestamp: number;
}

export interface SemanticCacheResult {
  response: string;
  score: number;
  metadata: Record<string, any>;
}

export interface CacheCheckResult {
  hit: boolean;
  response?: string;
  score?: number;
  metadata?: Record<string, any>;
  type: 'direct' | 'semantic' | 'none';
}

export class RedisCache {
  private redis: RedisClientType;
  private embeddings: BedrockEmbeddings;
  private logger: pino.Logger;
  private ttl: number = 86400; // 24 hours
  private semanticThreshold: number = 0.75;
  private isConnecting: boolean = false; 
  private connectionPromise: Promise<void> | null = null; 

  constructor(logger: pino.Logger) {
    this.logger = logger;

    this.redis = createClient({
      url: appConfig.redis.url,
      password: appConfig.redis.password,
    });

    this.redis.on('error', (err) => this.logger.error(err, 'Redis Client Error'));

    this.embeddings = new BedrockEmbeddings({
      region: process.env.AWS_REGION || 'ap-south-1',
      model: 'amazon.titan-embed-text-v2:0',
      credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID!,
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY!,
      },
    });
  }

  //WSingleton connection pattern
  async connect(): Promise<void> {
    if (this.redis.isOpen) {
      return; // Already connected
    }

    if (this.isConnecting && this.connectionPromise) {
      return this.connectionPromise; // Wait for ongoing connection
    }

    this.isConnecting = true;
    this.connectionPromise = (async () => {
      try {
        await this.redis.connect();
        this.logger.info('Redis cache connected');
      } finally {
        this.isConnecting = false;
        this.connectionPromise = null;
      }
    })();

    return this.connectionPromise;
  }

  async disconnect(): Promise<void> {
    if (this.redis.isOpen) {
      await this.redis.quit(); 
      this.logger.info('Redis cache disconnected');
    }
  }

  //  Add graceful shutdown method
  async shutdown(): Promise<void> {
    await this.disconnect();
  }

  // --- DIRECT CACHE ---
  private generateCacheKey(query: string, subject: string): string {
    const normalized = `${subject}:${query.toLowerCase().trim()}`;
    let hash = 0;
    for (let i = 0; i < normalized.length; i++) {
      hash = (hash << 5) - hash + normalized.charCodeAt(i);
    }
    return `cache:direct:${Math.abs(hash)}`;
  }

  async checkDirectCache(query: string, subject: string): Promise<CacheEntry | null> {
    try {
      await this.connect();
      const key = this.generateCacheKey(query, subject);
      const raw = await this.redis.get(key);

      if (!raw) {
        this.logger.debug({ key }, 'Direct cache miss');
        return null;
      }

      const entry = JSON.parse(raw) as CacheEntry;
      this.logger.info({ key }, 'Direct cache HIT ✓');
      return entry;
    } catch (error) {
      this.logger.error(error, 'Direct cache check failed');
      return null;
    }
  }

  async storeDirectCache(
    query: string,
    response: string,
    subject: string,
    metadata: Record<string, any> = {}
  ): Promise<void> {
    try {
      await this.connect();
      const key = this.generateCacheKey(query, subject);
      const entry: CacheEntry = {
        response,
        metadata: { ...metadata, subject, query },
        timestamp: Date.now(),
      };

      const serialized = JSON.stringify(entry);
      await this.redis.setEx(key, this.ttl, serialized);
      
      this.logger.info(
        { 
          key, 
          subject,
          topic: metadata.topic,
          dataLength: serialized.length,
          ttl: this.ttl,
        }, 
        '✅ Stored in direct cache (24h TTL)'
      );
    } catch (error) {
      this.logger.error(
        { 
          error: error instanceof Error ? error.message : String(error),
          query: query.substring(0, 50),
          subject,
        }, 
        '❌ Direct cache store failed'
      );
      throw error; // Re-throw to see the error in parent
    }
  }

  // --- SEMANTIC CACHE ---
  private generateQueryHash(query: string): string {
    const normalized = query.toLowerCase().trim();
    let hash = 0;
    for (let i = 0; i < normalized.length; i++) {
      hash = (hash << 5) - hash + normalized.charCodeAt(i);
    }
    return `query:${Math.abs(hash)}`;
  }

  async checkSemanticCache(query: string, subject: string): Promise<SemanticCacheResult | null> {
    try {
      await this.connect();

      const queryEmbedding = await this.embeddings.embedQuery(query);
      const hashKey = `cache:semantic:${subject}`;
      const entries = await this.redis.hGetAll(hashKey);

      if (!entries || Object.keys(entries).length === 0) {
        this.logger.debug({ hashKey }, 'Semantic cache empty for subject');
        return null;
      }

      let bestMatch: SemanticCacheResult | null = null;
      let bestScore = 0;

      // This part can still be slow if a subject has thousands of entries.
      // For now, it's a huge improvement over KEYS.
      for (const field in entries) {
        const raw = entries[field];
        if (!raw) continue;

        const cached = JSON.parse(raw) as CacheEntry & { embedding: number[] };
        const score = this.cosineSimilarity(queryEmbedding, cached.embedding);

        if (score > bestScore && score >= this.semanticThreshold) {
          bestScore = score;
          bestMatch = {
            response: cached.response,
            score,
            metadata: cached.metadata,
          };
        }
      }

      if (bestMatch) {
        this.logger.info({ score: bestMatch.score }, 'Semantic cache HIT ✓');
        return bestMatch;
      }

      this.logger.debug('Semantic cache miss');
      return null;
    } catch (error) {
      this.logger.error(error, 'Semantic cache check failed');
      return null;
    }
  }

  /**
   * More efficient batch embedding using embedDocuments
   */
  private async getEmbeddingBatch(queries: string[]): Promise<number[][]> {
    return this.embeddings.embedDocuments(queries);
  }

  async storeSemanticCache(
    query: string,
    response: string,
    subject: string,
    metadata: Record<string, any> = {}
  ): Promise<void> {
    try {
      await this.connect();

      const embedding = await this.embeddings.embedQuery(query);
      const timestamp = Date.now();
      const hashKey = `cache:semantic:${subject}`;
      const fieldKey = this.generateQueryHash(query);

      const entry = {
        response,
        embedding,
        metadata: { ...metadata, subject, query },
        timestamp,
      };

      const serialized = JSON.stringify(entry);

      // Use a pipeline to ensure atomicity of HSET and EXPIRE
      await this.redis
        .multi()
        .hSet(hashKey, fieldKey, serialized)
        .expire(hashKey, this.ttl)
        .exec();
        
      this.logger.info(
        { 
          key: hashKey, 
          field: fieldKey,
          subject,
          topic: metadata.topic,
          dataLength: serialized.length,
          embeddingLength: embedding.length,
        }, 
        '✅ Stored in semantic cache (24h TTL)'
      );
    } catch (error) {
      this.logger.error(
        { 
          error: error instanceof Error ? error.message : String(error),
          query: query.substring(0, 50),
          subject,
        }, 
        '❌ Semantic cache store failed'
      );
      throw error; // Re-throw to see the error in parent
    }
  }

  private cosineSimilarity(a: number[], b: number[]): number {
    if (a.length !== b.length) return 0;

    let dotProduct = 0;
    let normA = 0;
    let normB = 0;

    for (let i = 0; i < a.length; i++) {
      dotProduct += a[i] * b[i];
      normA += a[i] * a[i];
      normB += b[i] * b[i];
    }

    return dotProduct / (Math.sqrt(normA) * Math.sqrt(normB));
  }

   async checkUnified(
    query: string,
    subject: string = 'general'
  ): Promise<CacheCheckResult> {
    // Run both cache checks in parallel
    const [directCache, semanticCache] = await Promise.all([
      this.checkDirectCache(query, subject),
      this.checkSemanticCache(query, subject)
    ]);

    // Direct cache has priority (exact match)
    if (directCache) {
      this.logger.info('✓ DIRECT CACHE HIT');
      return {
        hit: true,
        response: directCache.response,
        metadata: directCache.metadata,
        type: 'direct'
      };
    }

    // Then semantic cache
    if (semanticCache) {
      this.logger.info({ score: semanticCache.score }, '✓ SEMANTIC CACHE HIT');
      return {
        hit: true,
        response: semanticCache.response,
        score: semanticCache.score,
        metadata: semanticCache.metadata,
        type: 'semantic'
      };
    }

    // No cache hit
    return { hit: false, type: 'none' };
  }

  // --------------------------------------------------------------------------
  // Advanced Cache Matching Methods (4-step semantic lookup)
  // --------------------------------------------------------------------------

  /**
   * Parse questions from cached response
   */
  parseQuestionsFromCache(
    response: string,
    topicName: string,
    source: 'exact' | 'semantic'
  ): any[] {
    try {
      const parsed = JSON.parse(response);

      if (!Array.isArray(parsed)) {
        this.logger.warn({ topicName, source }, '[Cache] Cached response is not an array');
        return [];
      }

      return parsed.map((q: any) => ({
        questionId: q.questionId || q.slotId || q.id || this.generateQuestionId(),
        question: q.question || q.q || '',
        options: q.options || [],
        correctAnswer: q.correctAnswer || q.options?.[q.answer - 1] || '',
        explanation: q.explanation,
        marks: q.marks,
        negativeMarks: q.negativeMarks,
        section: q.section,
        difficulty: q.difficulty || 'intermediate',
        topic: topicName,
        subject: q.subject,
        format: q.format || 'standard',
        cacheSource: source,
      }));
    } catch (error) {
      this.logger.warn({ error, topicName, source }, '[Cache] Failed to parse cached questions');
      return [];
    }
  }

  private generateQuestionId(): string {
    return `q_${Date.now()}_${Math.random().toString(36).substring(7)}`;
  }

  /**
   * Fetch exact fingerprint matches (Step 1)
   */
  async fetchExactFingerprint(
    topicName: string,
    subject: string,
    examTags: string[],
    limit: number
  ): Promise<any[]> {
    try {
      const cacheQuery = `Generate questions on ${topicName} for ${subject} (${examTags.join(', ')})`;
      const result = await this.checkDirectCache(cacheQuery, subject);

      if (!result) {
        this.logger.debug({ cacheQuery }, '[Cache] No exact fingerprint match');
        return [];
      }

      const questions = this.parseQuestionsFromCache(result.response, topicName, 'exact');
      
      return questions.slice(0, limit).map((q: any) => ({
        id: q.questionId,
        question: q,
        topicNorm: topicName.toLowerCase(),
        subjectNorm: subject.toLowerCase(),
        similarity: null,
        accepted: true,
        reason: 'exact_fingerprint',
      }));
    } catch (error) {
      this.logger.warn({ error, topic: topicName }, '[Cache] Exact fingerprint fetch failed');
      return [];
    }
  }

  /**
   * Fetch from topic set (Step 2) - Placeholder for future implementation
   */
  async fetchTopicSet(
    topicNorm: string,
    subjectNorm: string,
    limit: number
  ): Promise<any[]> {
    try {
      const topicSetKey = `cache:subject:${subjectNorm}:topic:${topicNorm}`;
      this.logger.debug({ topicSetKey }, '[Cache] Topic set fetch not yet implemented');
      return [];
    } catch (error) {
      this.logger.warn({ error, topicNorm }, '[Cache] Topic set fetch failed');
      return [];
    }
  }

  /**
   * Semantic match with context (Step 3)
   */
  async fetchSemanticWithContext(
    normalized: any,
    subjectNorm: string,
    subject: string,
    examTags: string[],
    limit: number,
    config: any
  ): Promise<any[]> {
    try {
      const queries = [
        `${normalized.stripped} question types for ${examTags.join(', ')} ${subjectNorm}`,
        `${normalized.stripped} sample questions`,
        `${normalized.stripped} ${subjectNorm}`,
      ];

      const candidates: any[] = [];

      for (const query of queries) {
        const result = await this.checkSemanticCache(query, subject);
        
        if (!result) continue;

        const questions = this.parseQuestionsFromCache(result.response, normalized.original, 'semantic');

        for (const q of questions) {
          if (candidates.length >= limit * 2) break;

          const similarity = result.score;
          let accepted = false;
          let reason = '';

          const metadataTopicNorm = (result.metadata.topic || '').toLowerCase().trim();
          const metadataSubjectNorm = (result.metadata.subject || '').toLowerCase().trim();

          if (metadataTopicNorm === normalized.stripped) {
            if (similarity >= config.semanticThreshold) {
              accepted = true;
              reason = 'topic_match';
            } else {
              reason = `similarity_below_threshold (${similarity.toFixed(3)} < ${config.semanticThreshold})`;
            }
          }
          else if (normalized.aliases.includes(metadataTopicNorm)) {
            if (similarity >= config.aliasBoostThreshold) {
              accepted = true;
              reason = 'alias_match';
            } else {
              reason = `alias_similarity_below_threshold (${similarity.toFixed(3)} < ${config.aliasBoostThreshold})`;
            }
          }
          else if (metadataSubjectNorm === subjectNorm && similarity >= 0.82) {
            accepted = true;
            reason = 'subject_match_high_similarity';
          }
          else {
            reason = `no_match (topic: ${metadataTopicNorm}, subject: ${metadataSubjectNorm}, sim: ${similarity.toFixed(3)})`;
          }

          candidates.push({
            id: q.questionId,
            question: q,
            topicNorm: metadataTopicNorm,
            subjectNorm: metadataSubjectNorm,
            similarity,
            accepted,
            reason,
          });
        }
      }

      return candidates.slice(0, limit);
    } catch (error) {
      this.logger.warn({ error, topic: normalized.original }, '[Cache] Semantic with context fetch failed');
      return [];
    }
  }

  /**
   * Cross-subject fallback (Step 4)
   */
  async fetchCrossSubjectFallback(
    normalized: any,
    subjectNorm: string,
    limit: number,
    config: any
  ): Promise<any[]> {
    try {
      const query = `${normalized.stripped} questions`;
      const result = await this.checkSemanticCache(query, 'general');

      if (!result || result.score < config.crossSubjectThreshold) {
        this.logger.debug(
          { query, score: result?.score, threshold: config.crossSubjectThreshold },
          '[Cache] Cross-subject fallback found nothing above threshold'
        );
        return [];
      }

      const questions = this.parseQuestionsFromCache(result.response, normalized.original, 'semantic');

      return questions.slice(0, limit).map((q: any) => {
        const similarity = result.score;
        const accepted = similarity >= config.crossSubjectThreshold;

        return {
          id: q.questionId,
          question: q,
          topicNorm: (result.metadata.topic || '').toLowerCase(),
          subjectNorm: (result.metadata.subject || '').toLowerCase(),
          similarity,
          accepted,
          reason: accepted ? 'cross_subject_fallback' : `fallback_similarity_too_low (${similarity.toFixed(3)})`,
        };
      });
    } catch (error) {
      this.logger.warn({ error }, '[Cache] Cross-subject fallback failed');
      return [];
    }
  }

  /**
   * Log comprehensive audit trail for cache lookup
   */
  logCacheLookupAudit(
    topicOriginal: string,
    topicNorm: string,
    requested: number,
    targetCached: number,
    actualCached: number,
    candidates: any[]
  ): void {
    const acceptedCount = candidates.filter((c) => c.accepted).length;
    const rejectedCount = candidates.filter((c) => !c.accepted).length;

    const auditData = {
      topic: topicOriginal,
      topicNorm,
      requested,
      targetCached,
      actualCached,
      acceptedCount,
      rejectedCount,
      candidatesTested: candidates.map((c) => ({
        questionId: c.id,
        topicNorm: c.topicNorm,
        subjectNorm: c.subjectNorm,
        similarity: c.similarity !== null ? Number(c.similarity.toFixed(3)) : null,
        accepted: c.accepted,
        reason: c.reason,
      })),
    };

    this.logger.debug(auditData, '[Cache] Lookup audit trail');
  }

}
