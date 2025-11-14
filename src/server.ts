import Fastify, { FastifyInstance } from 'fastify';
import cors from '@fastify/cors';
import { appConfig, modelConfigService } from './config/modelConfig.js';
import { createContainer } from './lib/container.js';
import { ChatRequest, AITutorResponse, Classification, Document } from './types/index.js';
import { Classifier } from './classifier/Classifier.js';
import pino from 'pino';

/**
 * Main Fastify server with LangChain AI Tutor orchestration + Evaluate endpoint
 */
export async function createServer(): Promise<FastifyInstance> {
  const server = Fastify({
    logger: {
      level: appConfig.logLevel,
      transport:
        appConfig.nodeEnv === 'development' ? { target: 'pino-pretty' } : undefined,
    },
    keepAliveTimeout: 60_000,
    requestTimeout: 30_000,
  });

  const logger = pino({ level: appConfig.logLevel });

  // Register plugins
  await server.register(cors, {
    origin: appConfig.nodeEnv === 'development' ? true : ['http://localhost:3000'],
    credentials: true,
  });

  // Initialize DI container
  const container = createContainer(logger);
  const classifier = new Classifier(logger);

  // Health check endpoints
  server.get('/health', async () => {
    return { status: 'ok', timestamp: new Date().toISOString() };
  });

  server.get('/ready', async () => {
    try {
      return { status: 'ready', services: { bedrock: 'ok' } };
    } catch (error) {
      server.log.error(error, 'Readiness check failed');
      const msg = error instanceof Error ? error.message : 'unknown';
      return { status: 'not ready', error: msg };
    }
  });

  /**
   * POST /chat - Initial chat request (returns classification + retrieval, ready for evaluation)
   */
  server.post<{ Body: ChatRequest }>('/chat', {
    schema: {
      body: {
        type: 'object',
        required: ['message'],
        properties: {
          message: { type: 'string' },
          subject: { type: 'string' },
          level: { type: 'string' },
          userSubscription: { type: 'string' },
          sessionId: { type: 'string' },
          language: { type: 'string' },
          examPrep: { type: 'boolean' },
        },
      },
    }
  }, async (request, reply) => {
    try {
      const tutorChain = container.getTutorChain();

      server.log.info(
        {
          message: request.body.message.substring(0, 100),
          subject: request.body.subject || 'auto-classify',
          level: request.body.level || 'auto-classify',
        },
        '[Chat] 📝 Incoming request'
      );

      // Run tutor chain (classifier + retrieval) with sessionId
      const result = await tutorChain.run(request.body, request.body.sessionId);

      server.log.info(
        {
          subject: result.classification.subject,
          sourceCount: result.sources?.length || 0,
          confidence: result.classification.confidence,
                  sessionId: request.body.sessionId,
        },
        '[Chat] ✅ Response ready'
      );

      reply.type('application/json');
      return result;
    } catch (error) {
      server.log.error(error, '[Chat] ❌ Chat request failed');
      reply.status(500);
      const msg = error instanceof Error ? error.message : 'unknown error';
      return {
        error: 'Internal server error',
        message: msg,
      };
    }
  });

  /**
   * POST /evaluate/stream - Streaming evaluate prompt and generate final response
   * Called when user clicks "Evaluate" button with streaming enabled
   */
  server.post<{
    Body: {
      userQuery: string;
      classification: Classification;
      documents: Document[];
      userSubscription?: string;
      sessionId?: string;
    };
  }>('/evaluate/stream', {
    schema: {
      body: {
        type: 'object',
        required: ['userQuery', 'classification', 'documents'],
        properties: {
          userQuery: { type: 'string' },
          classification: {
            type: 'object',
            properties: {
              subject: { type: 'string' },
              level: { type: 'string' },
              confidence: { type: 'number' },
              intent: { type: 'string' },
            },
          },
          documents: {
            type: 'array',
            items: { type: 'object' },
          },
          userSubscription: { type: 'string' },
          sessionId: { type: 'string' },
        },
      },
    }
  }, async (request, reply) => {
    try {
      const { userQuery, classification, documents, userSubscription, sessionId } = request.body;

      if (!userQuery || userQuery.trim() === '') {
        reply.status(400);
        return { error: 'userQuery is required and cannot be empty' };
      }

      if (!classification) {
        reply.status(400);
        return { error: 'classification is required' };
      }

      if (!Array.isArray(documents)) {
        reply.status(400);
        return { error: 'documents must be an array' };
      }

      server.log.info(
        {
          query: userQuery.substring(0, 80),
          subject: classification.subject,
          confidence: classification.confidence,
          docCount: documents.length,
                  sessionId,
        },
        '[Evaluate Stream] 🌊 Streaming evaluation request received'
      );

      // Set up Server-Sent Events headers
      reply.raw.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Cache-Control',
      });

      const tutorChain = container.getTutorChain();

      // Start streaming evaluation
      await tutorChain.evaluateStreaming(
        userQuery,
        classification,
        documents,
        userSubscription || 'free',
        {
          onToken: (token: string) => {
            // Send token as SSE event
            reply.raw.write(`data: ${JSON.stringify({ type: 'token', content: token })}\n\n`);
          },
          onMetadata: (metadata: any) => {
            // Send metadata as SSE event
            reply.raw.write(`data: ${JSON.stringify({ type: 'metadata', content: metadata })}\n\n`);
          },
          onComplete: (result) => {
            // Send final result
            reply.raw.write(`data: ${JSON.stringify({ 
              type: 'complete', 
              content: {
                success: true,
                data: {
                  answer: result.answer,
                  modelUsed: result.modelUsed,
                  levelUsed: result.levelUsed,
                  latency: result.latency,
                  classification,
                  sources: documents,
                                  sessionId,
                },
              }
            })}\n\n`);

            server.log.info(
              {
                modelUsed: result.modelUsed,
                levelUsed: result.levelUsed,
                latency: result.latency,
                answerLength: result.answer.length,
                              sessionId,
              },
              '[Evaluate Stream] ✅ Streaming evaluation complete'
            );

            reply.raw.end();
          },
          onError: (error) => {
            server.log.error(error, '[Evaluate Stream] ❌ Streaming evaluation failed');
            
            reply.raw.write(`data: ${JSON.stringify({ 
              type: 'error', 
              content: {
                success: false,
                error: 'Streaming evaluation failed',
                message: error.message,
              }
            })}\n\n`);
            
            reply.raw.end();
          },
        },
        sessionId
      );
    } catch (error) {
      server.log.error(error, '[Evaluate Stream] ❌ Streaming setup failed');
      
      if (!reply.sent) {
        reply.status(500);
        return {
          success: false,
          error: 'Streaming setup failed',
          message: error instanceof Error ? error.message : 'unknown error',
        };
      }
    }
  });

  /**
   * POST /evaluate - Evaluate prompt and generate final response
   * Called when user clicks "Evaluate" button on frontend
   */
  server.post<{
    Body: {
      userQuery: string;
      classification: Classification;
      documents: Document[];
      userSubscription?: string;
      sessionId?: string;
    };
  }>('/evaluate', {
    schema: {
      body: {
        type: 'object',
        required: ['userQuery', 'classification', 'documents'],
        properties: {
          userQuery: { type: 'string' },
          classification: {
            type: 'object',
            properties: {
              subject: { type: 'string' },
              level: { type: 'string' },
              confidence: { type: 'number' },
              intent: { type: 'string' },
            },
          },
          documents: {
            type: 'array',
            items: { type: 'object' },
          },
          userSubscription: { type: 'string' },
          sessionId: { type: 'string' },
        },
      },
    }
  }, async (request, reply) => {
    try {
      const { userQuery, classification, documents, userSubscription, sessionId } = request.body;

      if (!userQuery || userQuery.trim() === '') {
        reply.status(400);
        return { error: 'userQuery is required and cannot be empty' };
      }

      if (!classification) {
        reply.status(400);
        return { error: 'classification is required' };
      }

      if (!Array.isArray(documents)) {
        reply.status(400);
        return { error: 'documents must be an array' };
      }

      server.log.info(
        {
          query: userQuery.substring(0, 80),
          subject: classification.subject,
          confidence: classification.confidence,
          docCount: documents.length,
        },
        '[Evaluate] 🔬 Evaluation request received'
      );

      const tutorChain = container.getTutorChain();

      // Call evaluate on TutorChain
      const evaluateResult = await tutorChain.evaluate(
        userQuery,
        classification,
        documents,
        userSubscription || 'free',
        sessionId // Pass sessionId to enable chat history
      );

      server.log.info(
        {
          modelUsed: evaluateResult.modelUsed,
          levelUsed: evaluateResult.levelUsed,
          latency: evaluateResult.latency,
          answerLength: evaluateResult.answer.length,
                  sessionId,
        },
        '[Evaluate] ✅ Evaluation complete'
      );

      reply.type('application/json');
      return {
        success: true,
        data: {
          answer: evaluateResult.answer,
          modelUsed: evaluateResult.modelUsed,
          levelUsed: evaluateResult.levelUsed,
          latency: evaluateResult.latency,
          classification,
          sources: documents,
          sessionId,
        },
      };
    } catch (error) {
      server.log.error(error, '[Evaluate] ❌ Evaluation failed');
      reply.status(500);
      const msg = error instanceof Error ? error.message : 'unknown error';
      return {
        success: false,
        error: 'Evaluation failed',
        message: msg,
      };
    }
  });

  /**
   * POST /classify - Query classification endpoint
   */
  server.post<{ Body: { query: string } }>('/classify', {
    schema: {
      body: {
        type: 'object',
        required: ['query'],
        properties: {
          query: { type: 'string' },
        },
      },
    }
  }, async (request, reply) => {
    try {
      const { query } = request.body;
      const result = await classifier.classify(query);

      server.log.info(
        {
          query,
          subject: result.subject,
          level: result.level,
          confidence: result.confidence,
          intent: (result as any).intent,
        },
        '✓ Classification result with intent'
      );

      reply.type('application/json');
      return {
        success: true,
        query,
        classification: {
          subject: result.subject,
          level: result.level,
          confidence: result.confidence,
          intent: (result as any).intent,
          expectedFormat: (result as any).expectedFormat,
        },
      };
    } catch (error) {
      server.log.error(error, 'Classification failed');
      reply.status(500);
      const msg = error instanceof Error ? error.message : 'unknown';
      return { success: false, error: msg };
    }
  });

  /**
   * POST /rerank - Reranker endpoint
   */
  server.post<{
    Body: {
      documents: Document[];
      query: string;
      topK?: number;
    };
  }>('/rerank', async (request, reply) => {
    try {
      const { documents, query, topK } = request.body;

      if (!documents || documents.length === 0) {
        reply.status(400);
        return { error: 'Documents array is required and cannot be empty' };
      }

      if (!query || query.trim() === '') {
        reply.status(400);
        return { error: 'Query string is required and cannot be empty' };
      }

      server.log.info(
        { docCount: documents.length, query: query.substring(0, 50), topK },
        '[Server] Reranking request received'
      );

      const rerankerConfig = modelConfigService.getRerankerConfig();
      const reranker = container.getReranker();

      const rerankedResults = await reranker.rerank(documents, query, topK);

      server.log.info(
        { originalCount: documents.length, rerankedCount: rerankedResults.length },
        '[Server] Reranking completed'
      );

      reply.type('application/json');
      return {
        success: true,
        original_count: documents.length,
        reranked_count: rerankedResults.length,
        model_used: rerankerConfig.modelId,
        results: rerankedResults.map(r => ({
          id: r.document.id,
          text: r.document.text,
          metadata: r.document.metadata,
          original_score: r.document.score || 0,
          reranked_score: r.score,
          reason: r.reason,
        })),
      };
    } catch (error) {
      server.log.error(error, '[Server] Reranking failed');
      reply.status(500);
      const msg = error instanceof Error ? error.message : 'unknown';
      return { success: false, error: msg };
    }
  });

  /**
   * Graceful shutdown
   */
  const gracefulShutdown = () => {
    server.log.info('Received shutdown signal, starting graceful shutdown...');
    server.close(() => {
      server.log.info('Server closed successfully');
      process.exit(0);
    });
  };

  process.on('SIGTERM', gracefulShutdown);
  process.on('SIGINT', gracefulShutdown);

  return server;
}

/**
 * Start the server
 */
async function startServer() {
  try {
    const server = await createServer();

    // Start listening
    await server.listen({
      port: appConfig.port,
      host: '0.0.0.0',
    });

    server.log.info(`🚀 AI Tutor Service running on http://localhost:${appConfig.port}`);
    server.log.info(`📊 Health check: http://localhost:${appConfig.port}/health`);
    server.log.info(`🔄 Ready check: http://localhost:${appConfig.port}/ready`);

    return server;
  } catch (error) {
    console.error('Failed to start server:', error);
    process.exit(1);
  }
}

// Start server if this file is run directly
if (import.meta.url.endsWith('server.ts')) {
  startServer();
}

export { startServer };