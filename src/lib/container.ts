import pino from 'pino';
import { ModelSelector } from '../llm/ModelSelector.js';
import { Classifier } from '../classifier/Classifier.js';
import { AWSKnowledgeBaseRetriever } from '../retriever/AwsKBRetriever.js';
import { Reranker } from '../reranker/Reranker.js';
import { EvaluatePrompt } from '../prompts/evaluatorPrompt.js';
import { createTutorChain } from '../chains/questionChat.js';
import { ChatMemory } from '../cache/ChatMemory.js';
import { createClient } from 'redis';
import { loadAllSecrets } from './secrets.js';
import { PresentationOutlineChain } from '../presentation/outlineChain.js';
import { PresentationContentChain } from '../presentation/contentChain.js';
import { RedisCache } from '../cache/RedisCache.js';

export function createContainer(logger: pino.Logger) {
  let secrets: any = null;
  let redisClient: any = null;

  return {
    // Initialize and load secrets (called once at startup)
    initialize: async () => {
      secrets = await loadAllSecrets(); // Handles dev/prod automatically
      logger.info('✅ Secrets loaded');

      redisClient = createClient({
        url: secrets.redisUrl,
        password: secrets.redisPassword,
      });

      redisClient.on('error', (err: any) => logger.error('Redis Client Error', err));

      await redisClient.connect();
      logger.info('✅ Redis client connected');
    },

    // Build Tutor Chain dependencies
    getTutorChain: () => {
      if (!secrets || !redisClient) throw new Error('Call initialize() first');

      const modelSelector = new ModelSelector(logger);
      const classifier = new Classifier(logger, undefined);
      const retriever = new AWSKnowledgeBaseRetriever(logger);
      const reranker = new Reranker(logger, secrets.cohereApiKey);
      const evaluatePrompt = new EvaluatePrompt(modelSelector, logger);
      const chatMemory = new ChatMemory(redisClient, logger);

      return createTutorChain(
        classifier,
        retriever,
        modelSelector,
        reranker,
        evaluatePrompt,
        chatMemory,
        logger,
        secrets
      );
    },

    // Build Presentation Chain dependencies
    // Returns: { outlineChain, contentChain }
    getPresentationChains: () => {
      if (!secrets || !redisClient) throw new Error('Call initialize() first');

      const redisCache = new RedisCache(logger);
      
      const outlineChain = new PresentationOutlineChain(
        redisCache,
        logger,
        secrets.tavilyApiKey
      );
      
      const contentChain = new PresentationContentChain(
        logger,
      );

      return {
        outlineChain,
        contentChain,
        redisCache,
      };
    },

    getReranker: () => {
      if (!secrets) throw new Error('Call initialize() first');
      return new Reranker(logger, secrets.cohereApiKey);
    },

    getLogger: () => logger,
  };
}
