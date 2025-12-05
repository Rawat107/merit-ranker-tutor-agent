import { FastifyRequest, FastifyReply } from 'fastify';
import pino from 'pino';
import { EducatorAgent } from '../educatorAgent.js';
import { Classification } from '../../types/index.js';
import { Classifier } from '../../classifier/Classifier.js';
import { LinguaCompressor } from '../../compression/lingua_compressor.js';

const logger = pino({ level: 'info' });

/**
 * STREAMING HANDLER (SSE)
 * Real-time progress updates for educator content generation
 */
export async function educatorStreamHandler(
  request: FastifyRequest<{ Body: { query: string; userId?: string } }>,
  reply: FastifyReply
) {
  const { query, userId = 'anonymous' } = request.body;

  if (!query?.trim()) {
    return reply.status(400).send({ error: 'Query is required' });
  }

  // Set SSE headers
  reply.raw.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    'Connection': 'keep-alive',
    'X-Accel-Buffering': 'no',
  });

  const sendEvent = (data: any) => {
    reply.raw.write(`data: ${JSON.stringify(data)}\n\n`);
  };

  try {
    sendEvent({ type: 'start', message: 'Starting generation...' });

    // Create classifier and compress query for better classification
    const compressor = LinguaCompressor.getInstance(logger);
    const classifier = new Classifier(logger, compressor);
    
    // Classify the query to extract subject, level, intent
    const classification = await classifier.classify(query);
    
    logger.info(
      { subject: classification.subject, level: classification.level },
      'Query classified'
    );
    sendEvent({ 
      type: 'progress', 
      step: 'blueprint', 
      message: `Classified as ${classification.subject} (${classification.level} level)` 
    });

    // Create educator agent
    const agent = new EducatorAgent(logger);

    const startTime = Date.now();

    // Send progress updates for each step
    sendEvent({ type: 'progress', step: 'blueprint', message: 'Creating blueprint...' });
    
    // Execute the agent
    const result = await agent.execute(query, classification);

    const duration = Date.now() - startTime;

    // Send completion event
    sendEvent({
      type: 'complete',
      success: true,
      message: 'Generation complete!',
      content: {
        questions: result.generatedQuestions || [],
      },
      metadata: {
        duration,
        questionCount: result.generatedQuestions?.length || 0,
        successRate: 100,
      },
    });

    reply.raw.end();
  } catch (error) {
    logger.error({ error }, 'Stream handler error');
    sendEvent({
      type: 'error',
      error: error instanceof Error ? error.message : 'Unknown error',
    });
    reply.raw.end();
  }
}

/**
 * NON-STREAMING HANDLER (JSON response)
 * For backward compatibility and simple API calls
 */
export async function educatorLangGraphHandler(
  request: FastifyRequest<{ Body: { query: string; userId?: string } }>,
  reply: FastifyReply
) {
  const { query, userId = 'anonymous' } = request.body;

  if (!query?.trim()) {
    return reply.status(400).send({ error: 'Query is required' });
  }

  try {
    // Create classifier and compress query for better classification
    const compressor = LinguaCompressor.getInstance(logger);
    const classifier = new Classifier(logger, compressor);
    
    // Classify the query to extract subject, level, intent
    const classification = await classifier.classify(query);

    // Create educator agent
    const agent = new EducatorAgent(logger);

    const result = await agent.execute(query, classification);

    return reply.send({
      success: true,
      data: {
        questions: result.generatedQuestions || [],
        blueprint: result.blueprint,
        metadata: {
          questionCount: result.generatedQuestions?.length || 0,
        },
      },
    });
  } catch (error) {
    logger.error({ error }, 'LangGraph handler error');
    return reply.status(500).send({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
}

/**
 * QUIZ GENERATION HANDLER (backward compatible)
 */
export async function generateQuizHandler(
  request: FastifyRequest<{
    Body: { subject: string; topic: string; difficulty?: string; count?: number };
  }>,
  reply: FastifyReply
) {
  const { subject, topic, difficulty = 'MEDIUM', count = 10 } = request.body;

  const query = `Generate ${count} ${difficulty} difficulty ${subject} questions on ${topic}`;

  return educatorLangGraphHandler(
    { ...request, body: { query } } as any,
    reply
  );
}

/**
 * NOTES GENERATION HANDLER (backward compatible)
 */
export async function generateNotesHandler(
  request: FastifyRequest<{
    Body: { subject: string; topic: string; difficulty?: string; includeExamples?: boolean };
  }>,
  reply: FastifyReply
) {
  const { subject, topic, difficulty = 'MEDIUM', includeExamples = true } = request.body;

  const query = `Create comprehensive study notes on ${topic} (${subject}) at ${difficulty} level${
    includeExamples ? ' with examples' : ''
  }`;

  return educatorLangGraphHandler(
    { ...request, body: { query } } as any,
    reply
  );
}

/**
 * MOCK TEST GENERATION HANDLER (backward compatible)
 */
export async function generateMockTestHandler(
  request: FastifyRequest<{
    Body: { subject: string; topic: string; difficulty?: string; count?: number; duration?: number };
  }>,
  reply: FastifyReply
) {
  const { subject, topic, difficulty = 'MEDIUM', count = 20, duration = 60 } = request.body;

  const query = `Create a mock test on ${topic} with ${count} ${difficulty} questions (${duration} minutes duration)`;

  return educatorLangGraphHandler(
    { ...request, body: { query } } as any,
    reply
  );
}

/**
 * GRAPH VISUALIZATION HANDLER
 */
export async function getGraphVisualizationHandler(
  request: FastifyRequest,
  reply: FastifyReply
) {
  try {
    // Return graph flow description
    return reply.send({
      success: true,
      graph: {
        nodes: ['blueprint', 'research', 'refine', 'generate'],
        flow: 'START → blueprint → research → refine → generate → END',
        description: 'Educator agent pipeline with 4 sequential steps',
      },
    });
  } catch (error) {
    logger.error({ error }, 'Graph visualization error');
    return reply.status(500).send({
      success: false,
      error: error instanceof Error ? error.message : 'Unknown error',
    });
  }
}

/**
 * CONTENT RETRIEVAL HANDLER
 */
export async function getContentHandler(
  request: FastifyRequest<{ Params: { contentId: string } }>,
  reply: FastifyReply
) {
  const { contentId } = request.params;

  // TODO: Implement content storage and retrieval
  return reply.status(404).send({
    success: false,
    error: 'Content storage not yet implemented',
  });
}

/**
 * CONTENT UPDATE HANDLER
 */
export async function updateContentHandler(
  request: FastifyRequest<{ Params: { contentId: string }; Body: { updates: any } }>,
  reply: FastifyReply
) {
  const { contentId } = request.params;

  // TODO: Implement content updates
  return reply.status(404).send({
    success: false,
    error: 'Content updates not yet implemented',
  });
}

/**
 * CONTENT DELETE HANDLER
 */
export async function deleteContentHandler(
  request: FastifyRequest<{ Params: { contentId: string } }>,
  reply: FastifyReply
) {
  const { contentId } = request.params;

  // TODO: Implement content deletion
  return reply.status(404).send({
    success: false,
    error: 'Content deletion not yet implemented',
  });
}
