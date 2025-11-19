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

      await this.redis.setEx(key, this.ttl, JSON.stringify(entry));
      this.logger.info({ key }, 'Stored in direct cache (24h TTL)');
    } catch (error) {
      this.logger.error(error, 'Direct cache store failed');
    }
  }

  // --- SEMANTIC CACHE ---
  async checkSemanticCache(query: string, subject: string): Promise<SemanticCacheResult | null> {
    try {
      await this.connect();

      const queryEmbedding = await this.embeddings.embedQuery(query);
      const pattern = `cache:semantic:${subject}:*`;
      const keys = await this.redis.keys(pattern);

      if (keys.length === 0) {
        this.logger.debug('Semantic cache empty');
        return null;
      }

      let bestMatch: SemanticCacheResult | null = null;
      let bestScore = 0;

      for (const key of keys) {
        const raw = await this.redis.get(key);
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
      const key = `cache:semantic:${subject}:${timestamp}`;

      const entry = {
        response,
        embedding,
        metadata: { ...metadata, subject, query },
        timestamp,
      };

      await this.redis.setEx(key, this.ttl, JSON.stringify(entry));
      this.logger.info({ key }, 'Stored in semantic cache (24h TTL)');
    } catch (error) {
      this.logger.error(error, 'Semantic cache store failed');
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
}
