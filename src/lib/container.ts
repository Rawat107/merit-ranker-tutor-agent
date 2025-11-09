import pino from 'pino';
import { ModelSelector } from '../llm/ModelSelector.js';
import { Classifier } from '../classifier/Classifier.js';
import { SemanticCache } from '../cache/SemanticCache.js';
import { UpstashRetriever } from '../retriever/UpstashRetriever.js';
import { Reranker } from '../reranker/Reranker.js';
import { TutorChain } from '../chains/questionChat.js';

export function createContainer(logger: pino.Logger) {
  const modelSelector = new ModelSelector(logger);
  const classifier = new Classifier(logger, undefined /* optionally await modelSelector.getClassifierLLM() */);
  const cache = new SemanticCache(logger);
  const retriever = new UpstashRetriever(logger);
  const reranker = new Reranker(modelSelector, logger, false /* use LLM reranker if available */);

  const tutorChain = new TutorChain(classifier, cache, retriever, reranker, modelSelector, logger);

  return {
    getTutorChain: () => tutorChain,
    getLogger: () => logger,
  };
}
