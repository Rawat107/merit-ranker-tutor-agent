import pino from 'pino';
import { ModelSelector } from '../llm/ModelSelector.js';
import { Classifier } from '../classifier/Classifier.js';
import { AWSKnowledgeBaseRetriever } from '../retriever/AwsKBRetriever.js';
import { Reranker } from '../reranker/Reranker.js';
import { TutorChain } from '../chains/questionChat.js';

export function createContainer(logger: pino.Logger) {
  const modelSelector = new ModelSelector(logger);
  const classifier = new Classifier(logger, undefined /* optionally await modelSelector.getClassifierLLM() */);
  const retriever = new AWSKnowledgeBaseRetriever(logger);
  const reranker = new Reranker(logger);

  const tutorChain = new TutorChain(classifier, retriever, modelSelector, reranker, logger);

   return {
    getTutorChain: () => tutorChain,
    getLogger: () => logger,
    getReranker: () => reranker,  
  };
}
