import pino from 'pino';
import { RunnableLambda } from '@langchain/core/runnables';
import { ModelSelector } from '../llm/ModelSelector.js';
import { Classifier } from '../classifier/Classifier.js';
import { AWSKnowledgeBaseRetriever } from '../retriever/AwsKBRetriever.js';
import { Reranker } from '../reranker/Reranker.js';
import { EvaluatePrompt } from '../prompts/evaluatorPrompt.js';
import { createTutorChain } from '../chains/questionChat.js'; // ← Import the factory function
import { ChatMemory } from '../cache/ChatMemory.js';
import { createClient } from 'redis';
import { appConfig } from '../config/modelConfig.js';


export function createContainer(logger: pino.Logger) {
  // Initialize dependencies
  const modelSelector = new ModelSelector(logger);
  const classifier = new Classifier(logger, undefined);
  const retriever = new AWSKnowledgeBaseRetriever(logger);
  const reranker = new Reranker(logger);
  
  // Initialize EvaluatePrompt
  const evaluatePrompt = new EvaluatePrompt(modelSelector, logger);
  
  // Initialize ChatMemory with Redis
  const redisClient = createClient({
    url: appConfig.redis.url,
    password: appConfig.redis.password,
  });
  
  redisClient.connect().catch((err) =>
    logger.error(err, 'Failed to connect Redis client for ChatMemory')
  );
  
  const chatMemory = new ChatMemory(redisClient as any, logger);

  // Create TutorChain (now uses RunnableSequence internally)
  const tutorChain = createTutorChain(
    classifier,
    retriever,
    modelSelector,
    reranker,
    evaluatePrompt,
    chatMemory,
    logger
  );

  return {
    // Return the tutorChain instance (same as before, but now uses RunnableSequence)
    getTutorChain: () => tutorChain,
    
    // Utility getters
    getLogger: () => logger,
    getReranker: () => reranker,
    getChatMemory: () => chatMemory,
    getClassifier: () => classifier,
    getRetriever: () => retriever,
  };
}