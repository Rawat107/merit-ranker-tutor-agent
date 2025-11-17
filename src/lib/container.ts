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

export function createContainer(logger: pino.Logger) {
  let secrets: any = null;

  return {
    // Initialize and load secrets (called once at startup)
    initialize: async () => {
      secrets = await loadAllSecrets(); //This handles dev/prod automatically!
      logger.info('✅ Secrets loaded');
    },

    // Build dependencies with secrets
    getTutorChain: () => {
      if (!secrets) throw new Error('Call initialize() first');

      const modelSelector = new ModelSelector(logger);
      const classifier = new Classifier(logger, undefined);
      const retriever = new AWSKnowledgeBaseRetriever(logger);
      const reranker = new Reranker(logger, secrets.cohereApiKey); 
      const evaluatePrompt = new EvaluatePrompt(modelSelector, logger);

      const redisClient = createClient({
        url: secrets.redisUrl,        
        password: secrets.redisPassword, 
      });

      redisClient.connect().catch((err) =>
        logger.error(err, 'Failed to connect Redis')
      );

      const chatMemory = new ChatMemory(redisClient as any, logger);

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

    getReranker: () => {
      if (!secrets) throw new Error('Call initialize() first');
      return new Reranker(logger, secrets.cohereApiKey);
    },

    getLogger: () => logger,
  };
}
