import Redis from 'ioredis';
import pino from 'pino';
import { CacheEntry, SemanticCacheResult } from '../types/index.js';
import { appConfig } from '../config/modelConfig.js';

export class SemanticCache {
  private redis: Redis;
  private logger: pino.Logger;
  private serveSim: number;
  private softSim: number;
  private ttl: number;

  constructor(logger: pino.Logger) {
    this.logger = logger;
    this.redis = new Redis(appConfig.redis.url, { password: appConfig.redis.password });
    this.serveSim = appConfig.cache.serveSimilarity; // 0.94 default
    this.softSim = appConfig.cache.softSimilarity;   // 0.82 default
    this.ttl = appConfig.cache.ttl;                  // 86400 default
  }

  async lookupByKey(key: string) {
    try {
      const raw = await this.redis.get(key);
      if (!raw) return null;
      return JSON.parse(raw) as CacheEntry;
    } catch (e) {
      this.logger.warn(e, 'Cache key lookup failed');
      return null;
    }
  }

  /**
   * Semantic cache lookup stub.
   * In production, replace with vector search (Upstash, Redis-Vector, etc).
   * Here, returns null for no hit, or a mock hit for demonstration.
   */
  async lookupByEmbedding(query: string, subject: string, language: string): Promise<SemanticCacheResult | null> {
    // TODO: Replace with real embedding + vector search logic
    // For now, always return null (no semantic hit)
    return null;
  }

  async lookupByQuery(query: string, subject: string, level: string) {
    const key = this.generateKey(query, subject, level);
    return this.lookupByKey(key);
  }

  async upsert(query: string, response: string, meta: Record<string, any>) {
    const key = this.generateKey(query, meta.subject, meta.language || meta.level);
    const entry: CacheEntry = { response, embedding: [], metadata: meta, timestamp: Date.now() };
    try {
      await this.redis.setex(key, this.ttl, JSON.stringify(entry));
      this.logger.info({ key }, 'Cached tutor response');
    } catch (e) {
      this.logger.error(e, 'Cache upsert failed');
    }
  }

  generateKey(query: string, subject: string, language: string) {
    const base = `${subject}:${language}:${query.toLowerCase().trim()}`;
    let hash = 0; for (let i = 0; i < base.length; i++) hash = (hash << 5) - hash + base.charCodeAt(i);
    return `tutor:cache:${Math.abs(hash)}`;
  }
}
