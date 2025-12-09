import { FastifyRequest, FastifyReply } from 'fastify';
import pino from 'pino';
import { EducatorAgent } from '../educatorAgent.js';
import { Classification, AssessmentCategory, AssessmentRequest } from '../../types/index.js';
import { Classifier } from '../../classifier/Classifier.js';
import { LinguaCompressor } from '../../compression/lingua_compressor.js';

const logger = pino({ level: 'info' });

/**
 * STREAMING HANDLER (SSE)
 * Real-time progress updates for educator content generation
 */
export async function educatorStreamHandler(
  request: FastifyRequest<{ Body: AssessmentRequest }>,
  reply: FastifyReply
) {
  const { examTags, subject, totalQuestions, topics, userId = 'anonymous' } = request.body;

  if (!subject?.trim()) {
    return reply.status(400).send({ error: 'Subject is required' });
  }

  if (!totalQuestions || totalQuestions < 1 || totalQuestions > 150) {
    return reply.status(400).send({ error: 'Total questions must be between 1 and 150' });
  }

  // Set SSE headers with CORS support
  reply.raw.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache, no-transform',
    'Connection': 'keep-alive',
    'X-Accel-Buffering': 'no',
    'Access-Control-Allow-Origin': request.headers.origin || '*',
    'Access-Control-Allow-Credentials': 'true',
    'Access-Control-Allow-Headers': 'Content-Type, Authorization',
  });

  const sendEvent = (data: any) => {
    reply.raw.write(`data: ${JSON.stringify(data)}\n\n`);
  };

  try {
    sendEvent({ type: 'start', message: 'Starting generation...' });

    // Build a query string for classification if topics are not provided
    let classification: Classification;
    
    if (!topics || topics.length === 0) {
      // User didn't specify topics - use classifier to generate them
      const query = `Generate ${totalQuestions} questions on ${subject} for ${examTags.join(', ')}`;
      
      const compressor = LinguaCompressor.getInstance(logger);
      const classifier = new Classifier(logger, compressor);
      classification = await classifier.classify(query);
      
      logger.info({ subject: classification.subject, level: classification.level }, 'Query classified');
      sendEvent({ 
        type: 'progress', 
        step: 'blueprint', 
        message: `Classified as ${classification.subject} (${classification.level} level)` 
      });
    } else {
      // User provided topics - create classification from request
      classification = {
        subject: subject,
        level: 'intermediate', // Default level when topics are provided
        confidence: 1.0,
      };
      
      logger.info({ subject, topicsProvided: topics.length }, 'Using provided topics');
      sendEvent({ 
        type: 'progress', 
        step: 'blueprint', 
        message: `Using ${topics.length} provided topics` 
      });
    }

    // Create educator agent
    const agent = new EducatorAgent(logger);

    const startTime = Date.now();

    sendEvent({ type: 'progress', step: 'blueprint', message: 'Creating blueprint...' });
    
    // Execute the agent with the new request format
    const result = await agent.executeFromRequest(request.body, classification);

    const duration = Date.now() - startTime;

    // Transform questions to frontend format
    const transformedQuestions = (result.generatedQuestions || []).map((q: any) => ({
      question: q.q,
      options: q.options || [],
      correctAnswer: q.options?.[q.answer - 1] || q.options?.[0], // Convert 1-based index to actual answer
      explanation: q.explanation,
      difficulty: 'intermediate', // Default since we don't have this in new format
      topic: 'General', // We don't track topic per question anymore
      format: 'standard',
    }));

    logger.info(
      {
        generatedQuestionsCount: result.generatedQuestions?.length || 0,
        transformedQuestionsCount: transformedQuestions.length,
        cachedTotal: result.cachedTotal,
        toGenerateTotal: result.toGenerateTotal,
      },
      '[Handler] Sending completion event with all questions'
    );

    // Send completion event
    sendEvent({
      type: 'complete',
      success: true,
      message: 'Generation complete!',
      content: {
        questions: transformedQuestions,
      },
      metadata: {
        duration,
        questionCount: transformedQuestions.length,
        cachedTotal: result.cachedTotal || 0,
        generatedTotal: result.toGenerateTotal || 0,
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
  request: FastifyRequest<{ Body: { query: string; userId?: string; assessmentCategory?: AssessmentCategory } }>,
  reply: FastifyReply
) {
  const { query, userId = 'anonymous', assessmentCategory = 'quiz' } = request.body;

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

    const result = await agent.execute(query, classification, assessmentCategory);

    // Transform questions to frontend format
    const transformedQuestions = (result.generatedQuestions || []).map((q: any) => ({
      question: q.q,
      options: q.options || [],
      correctAnswer: q.options?.[q.answer - 1] || q.options?.[0],
      explanation: q.explanation,
      difficulty: 'intermediate',
      topic: 'General',
      format: 'standard',
    }));

    return reply.send({
      success: true,
      data: {
        questions: transformedQuestions,
        blueprint: result.blueprint,
        metadata: {
          questionCount: transformedQuestions.length,
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
