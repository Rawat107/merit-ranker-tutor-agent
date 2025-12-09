import pino from 'pino';
import { RedisCache } from '../../cache/RedisCache.js';
import { TopicRequest } from '../../types/index.js';
import { TopicValidator, NormalizedTopic } from './topicValidator.js';

export interface CachedQuestion {
  questionId: string;
  question: string;
  options: string[];
  correctAnswer: string;
  explanation?: string;
  marks?: number;
  negativeMarks?: number;
  section?: string;
  difficulty: 'basic' | 'intermediate' | 'advanced';
  topic: string;
  subject?: string;
  format?: 'standard' | 'math' | 'science' | 'reasoning' | 'diagram';
  cacheSource: 'exact' | 'semantic';
  cacheScore?: number;
}

export interface TopicWithCache extends TopicRequest {
  originalRequested: number;
  topicNorm: string; // Normalized topic for auditing
  cached: CachedQuestion[];
}

export interface CacheCandidate {
  id: string;
  question: CachedQuestion;
  topicNorm: string;
  subjectNorm: string;
  similarity: number | null; // null for exact/topic-set matches
  accepted: boolean;
  reason: string;
}

export interface CacheStageResult {
  topics: TopicWithCache[];
  cachedTotal: number;
  toGenerateTotal: number;
  cacheStats: {
    exactHits: number;
    semanticHits: number;
    misses: number;
    skipped: number;
  };
}

export interface CacheStageConfig {
  cacheRate: number; // Default 0.40 (40%)
  enableExactCache: boolean;
  enableSemanticCache: boolean;
  semanticThreshold: number; // Min similarity for same-topic semantic matches
  aliasBoostThreshold: number; // Min similarity for alias matches
  crossSubjectThreshold: number; // Min similarity for cross-subject fallback
  enableSessionDedupe: boolean;
  enableDebugLogs: boolean; // Log all candidates tested
}

const DEFAULT_CONFIG: CacheStageConfig = {
  cacheRate: 0.40,
  enableExactCache: true,
  enableSemanticCache: true,
  semanticThreshold: 0.78,
  aliasBoostThreshold: 0.75,
  crossSubjectThreshold: 0.85,
  enableSessionDedupe: true,
  enableDebugLogs: true,
};

/**
 * CACHE STAGE
 * Runs before blueprint, research, prompt refinement, and generation.
 * 
 * For each topic:
 * 1. Calculate targetCached = floor(requested * cacheRate)
 * 2. Fetch cached questions (exact match first, then semantic)
 * 3. Apply session dedupe to prevent duplicates
 * 4. Attach cached questions to topic
 * 5. Reduce topic.noOfQuestions by cached count
 * 
 * Returns modified topics with cached questions and reduced generation targets.
 */
export class CacheStage {
  private cache: RedisCache;
  private logger: pino.Logger;
  private config: CacheStageConfig;
  private validator: TopicValidator;
  private sessionQuestionIds: Set<string> = new Set();

  constructor(
    cache: RedisCache,
    logger: pino.Logger,
    config?: Partial<CacheStageConfig>
  ) {
    this.cache = cache;
    this.logger = logger;
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.validator = new TopicValidator(logger);
  }

  /**
   * Main entry point: process all topics and fetch cached questions
   */
  async process(
    topics: TopicRequest[],
    subject: string,
    examTags: string[]
  ): Promise<CacheStageResult> {
    this.logger.info(
      {
        action: 'cache_stage_start',
        topicsCount: topics.length,
        subject,
        examTags,
        cacheRate: this.config.cacheRate,
      },
      '[Cache Stage] Starting cache lookup for all topics'
    );

    const processedTopics: TopicWithCache[] = [];
    const stats = {
      exactHits: 0,
      semanticHits: 0,
      misses: 0,
      skipped: 0,
    };

    // Process each topic sequentially to maintain order
    for (const topic of topics) {
      const result = await this.processTopic(topic, subject, examTags);
      processedTopics.push(result.topic);
      
      // Update stats
      stats.exactHits += result.stats.exactHits;
      stats.semanticHits += result.stats.semanticHits;
      stats.misses += result.stats.misses;
      stats.skipped += result.stats.skipped;

      this.logger.info(
        {
          action: 'cache_stage',
          topic: topic.topicName,
          requested: topic.noOfQuestions,
          cachedCount: result.topic.cached.length,
          toGenerate: result.topic.noOfQuestions,
          exactHits: result.stats.exactHits,
          semanticHits: result.stats.semanticHits,
        },
        '[Cache Stage] Topic processed'
      );
    }

    const cachedTotal = processedTopics.reduce((sum, t) => sum + t.cached.length, 0);
    const toGenerateTotal = processedTopics.reduce((sum, t) => sum + t.noOfQuestions, 0);

    this.logger.info(
      {
        action: 'cache_stage_complete',
        cachedTotal,
        toGenerateTotal,
        cacheHitRate: cachedTotal / (cachedTotal + toGenerateTotal),
        stats,
      },
      '[Cache Stage] ✅ Cache stage complete'
    );

    return {
      topics: processedTopics,
      cachedTotal,
      toGenerateTotal,
      cacheStats: stats,
    };
  }

  /**
   * Process a single topic with 4-step ranking system:
   * 1. Exact fingerprint match
   * 2. Topic set exact fetch
   * 3. Semantic match with topic+subject context
   * 4. Cross-subject fallback (if needed)
   */
  private async processTopic(
    topic: TopicRequest,
    subject: string,
    examTags: string[]
  ): Promise<{
    topic: TopicWithCache;
    stats: { exactHits: number; semanticHits: number; misses: number; skipped: number };
  }> {
    const requested = topic.noOfQuestions;
    const targetCached = Math.floor(requested * this.config.cacheRate);

    const stats = {
      exactHits: 0,
      semanticHits: 0,
      misses: 0,
      skipped: 0,
    };

    // Normalize topic and subject
    const normalized = this.validator.normalize(topic.topicName);
    const subjectNorm = this.validator.normalizeSubject(subject);

    if (targetCached === 0) {
      this.logger.debug(
        { topic: topic.topicName, requested },
        '[Cache Stage] Skipping cache (targetCached = 0)'
      );
      
      return {
        topic: {
          ...topic,
          originalRequested: requested,
          topicNorm: normalized.stripped,
          cached: [],
        },
        stats: { ...stats, skipped: 1 },
      };
    }

    const cached: CachedQuestion[] = [];
    const candidates: CacheCandidate[] = [];

    this.logger.info(
      {
        action: 'cache_lookup_start',
        topic: topic.topicName,
        topicNorm: normalized.stripped,
        aliases: normalized.aliases,
        requested,
        targetCached,
        thresholds: {
          semantic: this.config.semanticThreshold,
          alias: this.config.aliasBoostThreshold,
          crossSubject: this.config.crossSubjectThreshold,
        },
      },
      '[Cache Stage] Starting 4-step cache lookup'
    );

    // STEP 1: Exact fingerprint match
    if (this.config.enableExactCache && cached.length < targetCached) {
      const exactResults = await this.cache.fetchExactFingerprint(
        topic.topicName,
        subject,
        examTags,
        targetCached - cached.length
      );

      for (const result of exactResults) {
        if (cached.length >= targetCached) break;
        if (this.config.enableSessionDedupe && this.sessionQuestionIds.has(result.question.questionId)) {
          candidates.push({ ...result, accepted: false, reason: 'session_dedupe' });
          continue;
        }

        cached.push(result.question);
        this.sessionQuestionIds.add(result.question.questionId);
        candidates.push({ ...result, accepted: true, reason: 'exact_fingerprint' });
        stats.exactHits++;
      }
    }

    // STEP 2: Topic set exact fetch
    if (cached.length < targetCached) {
      const topicSetResults = await this.cache.fetchTopicSet(
        normalized.stripped,
        subjectNorm,
        targetCached - cached.length
      );

      for (const result of topicSetResults) {
        if (cached.length >= targetCached) break;
        if (this.config.enableSessionDedupe && this.sessionQuestionIds.has(result.question.questionId)) {
          candidates.push({ ...result, accepted: false, reason: 'session_dedupe' });
          continue;
        }

        cached.push(result.question);
        this.sessionQuestionIds.add(result.question.questionId);
        candidates.push({ ...result, accepted: true, reason: 'topic_set' });
        stats.exactHits++;
      }
    }

    // STEP 3: Semantic match with context
    if (this.config.enableSemanticCache && cached.length < targetCached) {
      const semanticResults = await this.cache.fetchSemanticWithContext(
        normalized,
        subjectNorm,
        subject,
        examTags,
        targetCached - cached.length,
        this.config
      );

      for (const result of semanticResults) {
        if (cached.length >= targetCached) break;
        if (this.config.enableSessionDedupe && this.sessionQuestionIds.has(result.question.questionId)) {
          candidates.push({ ...result, accepted: false, reason: 'session_dedupe' });
          continue;
        }

        cached.push(result.question);
        this.sessionQuestionIds.add(result.question.questionId);
        candidates.push(result); // Already marked as accepted/rejected
        if (result.accepted) {
          stats.semanticHits++;
        }
      }
    }

    // STEP 4: Cross-subject fallback (if still short)
    if (this.config.enableSemanticCache && cached.length < targetCached) {
      const fallbackResults = await this.cache.fetchCrossSubjectFallback(
        normalized,
        subjectNorm,
        targetCached - cached.length,
        this.config
      );

      for (const result of fallbackResults) {
        if (cached.length >= targetCached) break;
        if (this.config.enableSessionDedupe && this.sessionQuestionIds.has(result.question.questionId)) {
          candidates.push({ ...result, accepted: false, reason: 'session_dedupe' });
          continue;
        }

        cached.push(result.question);
        this.sessionQuestionIds.add(result.question.questionId);
        candidates.push(result);
        if (result.accepted) {
          stats.semanticHits++;
        }
      }
    }

    // Calculate misses
    stats.misses = targetCached - cached.length;

    // Log audit trail
    this.cache.logCacheLookupAudit(
      topic.topicName,
      normalized.stripped,
      requested,
      targetCached,
      cached.length,
      candidates
    );

    // Build result
    const topicWithCache: TopicWithCache = {
      ...topic,
      originalRequested: requested,
      topicNorm: normalized.stripped,
      cached,
      noOfQuestions: requested - cached.length,
    };

    return { topic: topicWithCache, stats };
  }

  /**
   * Fetch exact cache matches (fingerprint-based)
   * Uses direct cache lookup with deterministic key generation
   */


  /**
   * Clear session dedupe set (call at the start of new requests)
   */
  resetSession(): void {
    this.sessionQuestionIds.clear();
    this.logger.debug('[Cache Stage] Session dedupe reset');
  }
}

/**
 * Factory function
 */
export function createCacheStage(
  cache: RedisCache,
  logger: pino.Logger,
  config?: Partial<CacheStageConfig>
): CacheStage {
  return new CacheStage(cache, logger, config);
}
