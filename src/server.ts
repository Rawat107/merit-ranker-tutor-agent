import Fastify, { FastifyInstance } from 'fastify';
import cors from '@fastify/cors';
import { appConfig } from './config/modelConfig.js';
import { createContainer } from './lib/container.js';
import { ChatRequest, AITutorResponse } from './types/index.js';
import { Classifier } from './classifier/Classifier.js';
import pino from 'pino';

/**
 * Main Fastify server with LangChain AI Tutor orchestration
 */
export async function createServer(): Promise<FastifyInstance> {
  const server = Fastify({
    logger: {
      level: appConfig.logLevel,
      transport: appConfig.nodeEnv === 'development' ? { target: 'pino-pretty' } : undefined
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
      // TODO: Check dependencies (AWS, etc.)
      return { status: 'ready', services: { bedrock: 'ok' } };
    } catch (error) {
      server.log.error(error, 'Readiness check failed');
      const msg = error instanceof Error ? error.message : 'unknown';
      return { status: 'not ready', error: msg };
    }
  });

  // Non-streaming chat endpoint
  // Non-streaming chat endpoint
// POST /chat with { message, subject?, level? }
server.post<{ Body: ChatRequest }>('/chat', {
  schema: {
    body: {
      type: 'object',
      required: ['message'],
      properties: {
        message: { type: 'string' },
        subject: { type: 'string' },           // Optional: from classifier
        level: { type: 'string' },              // Optional: from classifier
        userSubscription: { type: 'string' },
        sessionId: { type: 'string' },
        language: { type: 'string' },
        examPrep: { type: 'boolean' }
      }
    }
  }
}, async (request, reply) => {
  try {
    const tutorChain = container.getTutorChain();
    
    // Log incoming request
    server.log.info(
      {
        message: request.body.message.substring(0, 100),
        subject: request.body.subject || 'auto-classify',
        level: request.body.level || 'auto-classify'
      },
      '[Chat] 📝 Incoming request'
    );

    // Run tutor chain
    const result = await tutorChain.run(request.body);

    // Log response
    server.log.info(
      {
        subject: result.classification.subject,
        sourceCount: result.sources?.length || 0,
        confidence: result.classification.confidence
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
      message: msg 
    };
  }
});

  // Query classification endpoint
server.post<{ Body: { query: string } }>('/classify', {
  schema: {
    body: {
      type: 'object',
      required: ['query'],
      properties: {
        query: { type: 'string' }
      }
    }
  }
}, async (request, reply) => {
  try {
    const { query } = request.body;
    const result = await classifier.classify(query);

    server.log.info({ 
      query, 
      subject: result.subject, 
      level: result.level, 
      confidence: result.confidence 
    }, '✓ Classification result');

    reply.type('application/json');
    return {
      success: true,
      query,
      classification: result
    };
  } catch (error) {
    server.log.error(error, 'Classification failed');
    reply.status(500);
    const msg = error instanceof Error ? error.message : 'unknown';
    return { success: false, error: msg };
  }
});


  // Streaming chat endpoint using Server-Sent Events
  server.get<{ Querystring: { message: string; subject?: string; level?: string; userSubscription?: string } }>(
    '/chat/stream',
    async (request, reply) => {
      const { message, subject, level, userSubscription } = request.query;
      
      if (!message) {
        reply.status(400);
        return { error: 'Message parameter is required' };
      }

      // Set SSE headers
      reply.type('text/event-stream');
      reply.header('Cache-Control', 'no-cache');
      reply.header('Connection', 'keep-alive');
      reply.header('Access-Control-Allow-Origin', '*');
      reply.header('Access-Control-Allow-Headers', 'Cache-Control');

      const tutorChain = container.getTutorChain();
      const chatRequest: ChatRequest = {
        message,
        subject,
        level,
        userSubscription
      };

      try {
        await tutorChain.runStreaming(chatRequest, {
          onToken: (token: string) => {
            reply.raw.write(`data: ${JSON.stringify({ type: 'token', content: token })}\n\n`);
          },
          onMetadata: (metadata: any) => {
            reply.raw.write(`data: ${JSON.stringify({ type: 'metadata', content: metadata })}\n\n`);
          },
          onComplete: (result: AITutorResponse) => {
            reply.raw.write(`data: ${JSON.stringify({ type: 'complete', content: result })}\n\n`);
            reply.raw.write('data: [DONE]\n\n');
            reply.raw.end();
          },
          onError: (error: Error) => {
            reply.raw.write(`data: ${JSON.stringify({ type: 'error', content: error.message })}\n\n`);
            reply.raw.end();
          }
        });
      } catch (error) {
        server.log.error(error, 'Streaming chat failed');
        reply.raw.write(`data: ${JSON.stringify({ type: 'error', content: 'Streaming failed' })}\n\n`);
        reply.raw.end();
      }
    }
  );

  // Graceful shutdown
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
      host: '0.0.0.0' 
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