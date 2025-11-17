import Fastify, { FastifyInstance } from 'fastify';
import cors from '@fastify/cors';
import { Client } from 'langsmith';
import { appConfig, modelConfigService } from './config/modelConfig.js';
import { createContainer } from './lib/container.js';
import { ChatRequest, AITutorResponse, Classification, Document } from './types/index.js';
import { Classifier } from './classifier/Classifier.js';
import pino from 'pino';


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

  if (process.env.LANGCHAIN_TRACING_V2 === 'true') {
    server.log.info('LangSmith tracing enabled');
  } else {
    server.log.info('LangSmith tracing disabled');
  }

  const logger = pino({ 
    level: appConfig.logLevel,
    transport: appConfig.nodeEnv === 'development' ? { target: 'pino-pretty' } : undefined,
  });

  await server.register(cors, {
    origin: appConfig.nodeEnv === 'development' ? true : ['http://localhost:3000'],
    credentials: true,
  });

  server.log.info({ env: appConfig.nodeEnv, logLevel: appConfig.logLevel }, 'Initializing server');

  const container = createContainer(logger);
  const classifier = new Classifier(logger);

  server.log.info('Container and classifier initialized');

  server.get('/health', async () => {
    return { status: 'ok', timestamp: new Date().toISOString() };
  });

  server.get('/ready', async (request, reply) => {
    try {
      return { status: 'ready', services: { bedrock: 'ok' } };
    } catch (error) {
      server.log.error({ error, path: request.url }, 'Readiness check failed');
      const msg = error instanceof Error ? error.message : 'unknown';
      return { status: 'not ready', error: msg };
    }
  });

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
    const startTime = Date.now();
    const { sessionId, message, subject, level } = request.body;
    
    try {
      server.log.info(
        {
          endpoint: '/chat',
          sessionId,
          messagePreview: message.substring(0, 100),
          subject: subject || 'auto',
          level: level || 'auto',
          messageLength: message.length,
        },
        'Chat request received'
      );

      const tutorChain = container.getTutorChain();
      const result = await tutorChain.run(request.body, sessionId);

      const duration = Date.now() - startTime;

      server.log.info(
        {
          endpoint: '/chat',
          sessionId,
          subject: result.classification.subject,
          level: result.classification.level,
          confidence: result.classification.confidence,
          sourceCount: result.sources?.length || 0,
          cached: result.cached || false,
          duration,
        },
        'Chat request completed'
      );

      reply.type('application/json');
      return result;
    } catch (error) {
      const duration = Date.now() - startTime;
      server.log.error(
        { 
          endpoint: '/chat',
          sessionId,
          error: error instanceof Error ? error.message : String(error),
          stack: error instanceof Error ? error.stack : undefined,
          duration,
        }, 
        'Chat request failed'
      );
      
      reply.status(500);
      return {
        error: 'Internal server error',
        message: error instanceof Error ? error.message : 'unknown error',
      };
    }
  });

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
    const startTime = Date.now();
    const { userQuery, classification, documents, userSubscription, sessionId } = request.body;

    if (!userQuery || userQuery.trim() === '') {
      server.log.warn({ endpoint: '/evaluate/stream' }, 'Empty userQuery received');
      reply.status(400);
      return { error: 'userQuery is required and cannot be empty' };
    }

    if (!classification) {
      server.log.warn({ endpoint: '/evaluate/stream' }, 'Missing classification');
      reply.status(400);
      return { error: 'classification is required' };
    }

    if (!Array.isArray(documents)) {
      server.log.warn({ endpoint: '/evaluate/stream' }, 'Invalid documents format');
      reply.status(400);
      return { error: 'documents must be an array' };
    }

    try {
      server.log.info(
        {
          endpoint: '/evaluate/stream',
          sessionId,
          queryPreview: userQuery.substring(0, 80),
          subject: classification.subject,
          level: classification.level,
          confidence: classification.confidence,
          intent: (classification as any).intent,
          docCount: documents.length,
          subscription: userSubscription || 'free',
        },
        'Streaming evaluation started'
      );

      reply.raw.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Cache-Control',
      });

      const tutorChain = container.getTutorChain();
      let tokenCount = 0;

      await tutorChain.evaluateStreaming(
        userQuery,
        classification,
        documents,
        userSubscription || 'free',
        {
          onToken: (token: string) => {
            tokenCount++;
            reply.raw.write(`data: ${JSON.stringify({ type: 'token', content: token })}\n\n`);
          },
          onMetadata: (metadata: any) => {
            server.log.debug({ 
              endpoint: '/evaluate/stream',
              sessionId,
              metadata 
            }, 'Metadata sent to client');
            reply.raw.write(`data: ${JSON.stringify({ type: 'metadata', content: metadata })}\n\n`);
          },
          onComplete: (result) => {
            const duration = Date.now() - startTime;
            
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
                endpoint: '/evaluate/stream',
                sessionId,
                modelUsed: result.modelUsed,
                levelUsed: result.levelUsed,
                streamLatency: result.latency,
                totalDuration: duration,
                answerLength: result.answer.length,
                tokensStreamed: tokenCount,
              },
              'Streaming evaluation completed'
            );

            reply.raw.end();
          },
          onError: (error) => {
            const duration = Date.now() - startTime;
            
            server.log.error({ 
              endpoint: '/evaluate/stream',
              sessionId,
              error: error.message,
              stack: error.stack,
              duration,
            }, 'Streaming evaluation failed');
            
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
      const duration = Date.now() - startTime;
      
      server.log.error({ 
        endpoint: '/evaluate/stream',
        sessionId,
        error: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined,
        duration,
      }, 'Streaming setup failed');
      
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
    const startTime = Date.now();
    const { userQuery, classification, documents, userSubscription, sessionId } = request.body;

    if (!userQuery || userQuery.trim() === '') {
      server.log.warn({ endpoint: '/evaluate' }, 'Empty userQuery received');
      reply.status(400);
      return { error: 'userQuery is required and cannot be empty' };
    }

    if (!classification) {
      server.log.warn({ endpoint: '/evaluate' }, 'Missing classification');
      reply.status(400);
      return { error: 'classification is required' };
    }

    if (!Array.isArray(documents)) {
      server.log.warn({ endpoint: '/evaluate' }, 'Invalid documents format');
      reply.status(400);
      return { error: 'documents must be an array' };
    }

    try {
      server.log.info(
        {
          endpoint: '/evaluate',
          sessionId,
          queryPreview: userQuery.substring(0, 80),
          subject: classification.subject,
          level: classification.level,
          confidence: classification.confidence,
          intent: (classification as any).intent,
          docCount: documents.length,
          subscription: userSubscription || 'free',
        },
        'Evaluation started'
      );

      const tutorChain = container.getTutorChain();
      const evaluateResult = await (tutorChain as any).evaluate(
        userQuery,
        classification,
        documents,
        userSubscription || 'free',
        sessionId
      );

      const duration = Date.now() - startTime;

      server.log.info(
        {
          endpoint: '/evaluate',
          sessionId,
          modelUsed: evaluateResult.modelUsed,
          levelUsed: evaluateResult.levelUsed,
          evaluateLatency: evaluateResult.latency,
          totalDuration: duration,
          answerLength: evaluateResult.answer.length,
        },
        'Evaluation completed'
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
      const duration = Date.now() - startTime;
      
      server.log.error({ 
        endpoint: '/evaluate',
        sessionId,
        error: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined,
        duration,
      }, 'Evaluation failed');
      
      reply.status(500);
      return {
        success: false,
        error: 'Evaluation failed',
        message: error instanceof Error ? error.message : 'unknown error',
      };
    }
  });

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
    const startTime = Date.now();
    const { query } = request.body;

    try {
      server.log.info(
        { 
          endpoint: '/classify',
          queryPreview: query.substring(0, 100),
          queryLength: query.length,
        },
        'Classification request received'
      );

      const result = await classifier.classify(query);
      const duration = Date.now() - startTime;

      server.log.info(
        {
          endpoint: '/classify',
          subject: result.subject,
          level: result.level,
          confidence: result.confidence,
          intent: (result as any).intent,
          duration,
        },
        'Classification completed'
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
      const duration = Date.now() - startTime;
      
      server.log.error({ 
        endpoint: '/classify',
        error: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined,
        duration,
      }, 'Classification failed');
      
      reply.status(500);
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'unknown' 
      };
    }
  });

  server.post<{
    Body: {
      documents: Document[];
      query: string;
      topK?: number;
    };
  }>('/rerank', async (request, reply) => {
    const startTime = Date.now();
    const { documents, query, topK } = request.body;

    if (!documents || documents.length === 0) {
      server.log.warn({ endpoint: '/rerank' }, 'Empty documents array received');
      reply.status(400);
      return { error: 'Documents array is required and cannot be empty' };
    }

    if (!query || query.trim() === '') {
      server.log.warn({ endpoint: '/rerank' }, 'Empty query received');
      reply.status(400);
      return { error: 'Query string is required and cannot be empty' };
    }

    try {
      server.log.info(
        { 
          endpoint: '/rerank',
          docCount: documents.length,
          queryPreview: query.substring(0, 50),
          topK: topK || documents.length,
        },
        'Reranking request received'
      );

      const rerankerConfig = modelConfigService.getRerankerConfig();
      const reranker = container.getReranker();
      const rerankedResults = await reranker.rerank(documents, query, topK);

      const duration = Date.now() - startTime;

      server.log.info(
        { 
          endpoint: '/rerank',
          originalCount: documents.length,
          rerankedCount: rerankedResults.length,
          model: rerankerConfig.modelId,
          duration,
        },
        'Reranking completed'
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
      const duration = Date.now() - startTime;
      
      server.log.error({ 
        endpoint: '/rerank',
        error: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined,
        duration,
      }, 'Reranking failed');
      
      reply.status(500);
      return { 
        success: false, 
        error: error instanceof Error ? error.message : 'unknown' 
      };
    }
  });

  const gracefulShutdown = async () => {
    server.log.info({ signal: 'SIGTERM/SIGINT' }, 'Shutdown signal received');
    
    try {
      await server.close();
      server.log.info('Server closed gracefully');
      process.exit(0);
    } catch (error) {
      server.log.error({ error }, 'Error during shutdown');
      process.exit(1);
    }
  };

  process.on('SIGTERM', gracefulShutdown);
  process.on('SIGINT', gracefulShutdown);

  return server;
}

async function startServer() {
  try {
    const server = await createServer();

    await server.listen({
      port: appConfig.port,
      host: '0.0.0.0',
    });

    server.log.info({ 
      port: appConfig.port,
      host: '0.0.0.0',
      env: appConfig.nodeEnv,
      endpoints: ['/chat', '/evaluate', '/evaluate/stream', '/classify', '/rerank', '/health', '/ready'],
    }, 'Server started successfully');

    return server;
  } catch (error) {
    const logger = pino({ level: 'error' });
    logger.error({ 
      error: error instanceof Error ? error.message : String(error),
      stack: error instanceof Error ? error.stack : undefined,
    }, 'Failed to start server');
    process.exit(1);
  }
}

if (import.meta.url.endsWith('server.ts')) {
  startServer();
}

export { startServer };