import Fastify, { FastifyInstance } from 'fastify';
import cors from '@fastify/cors';
import rateLimit from '@fastify/rate-limit';
import FastifyAwsJwtVerify from 'fastify-aws-jwt-verify';
import { appConfig, modelConfigService } from './config/modelConfig.js';
import { createContainer } from './lib/container.js';
import { ChatRequest, Classification, Document } from './types/index.js';
import { Classifier } from './classifier/Classifier.js';
import { loadAllSecrets } from './lib/secrets.js';
import pino from 'pino';
import { RedisCache } from './cache/RedisCache.js';
import { NodeHttpHandler } from '@smithy/node-http-handler';
import { Agent } from 'https';
import { ModelSelector } from './llm/ModelSelector.js';
import { PresentationOutlineChain } from './presentation/outlineChain.js';
import { PresentationContentChain } from './presentation/contentChain.js';
import { SlideOutlineRequest } from './types/index.js';
import { linguaCompressor } from './compression/lingua_compressor.js';


export async function createServer(): Promise<FastifyInstance> {

  const httpsAgent = new Agent({
    keepAlive: true,
    maxSockets: 50,
    maxFreeSockets: 10,
    timeout: 60000,
  });



  // Enable connection reuse globally
  process.env.AWS_NODEJS_CONNECTION_REUSE_ENABLED = '1';

  const server = Fastify({
    logger: {
      level: appConfig.logLevel,
      transport:
        appConfig.nodeEnv === 'development' ? { target: 'pino-pretty' } : undefined,
    },
    keepAliveTimeout: 60_000,
    requestTimeout: 30_000,
  });

  server.decorate('awsConfig', {
    region: process.env.AWS_REGION || 'ap-south-1',
    requestHandler: new NodeHttpHandler({
      httpsAgent: httpsAgent, // ← USE the agent you created
      requestTimeout: 30000,
    }),
    maxAttempts: 3,
  });

  if (appConfig.nodeEnv === 'development') {
    server.addHook('onRequest', (request, reply, done) => {
      if (!request.headers.authorization && process.env.DEV_JWT_TOKEN) {
        request.headers.authorization = `Bearer ${process.env.DEV_JWT_TOKEN}`;
      }
      done();
    });
  }

  if (process.env.LANGCHAIN_TRACING === 'true') {
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

  // Rate limiting
  await server.register(rateLimit, {
    max: 100,
    timeWindow: '10 minute',
    cache: 10000,
    allowList: ['127.0.0.1'],
    errorResponseBuilder: (req, context) => ({
      statusCode: 429,
      error: 'Too Many Requests',
      message: `You have exceeded the maximum request limit. Retry in ${Math.ceil(context.ttl / 1000)}s`,
    })
  });

  server.log.info('Rate limiting configured');

  // Load secrets from AWS Secrets Manager (production) or .env (development)
  server.log.info('Loading secrets...');
  const secrets = await loadAllSecrets();
  server.decorate('secrets', secrets);
  server.log.info('✅ Secrets loaded successfully');

  const cognitoClientId = process.env.COGNITO_CLIENT_ID;
  const cognitoUserPoolId = process.env.COGNITO_USER_POOL_ID;
  const cognitoRegion = process.env.COGNITO_REGION || 'ap-southeast-2';
  if (!cognitoClientId || !cognitoUserPoolId) {
    server.log.error(
      { clientId: cognitoClientId, userPoolId: cognitoUserPoolId },
      'Missing AWS Cognito configuration in environment variables'
    );
    throw new Error('COGNITO_CLIENT_ID and COGNITO_USER_POOL_ID must be set in environment variables');
  }

  await server.register(FastifyAwsJwtVerify as any, {
    clientId: cognitoClientId,
    region: cognitoRegion,
    tokenProvider: 'Bearer',
    tokenUse: 'access',
    userPoolId: cognitoUserPoolId,
  });

  server.log.info({
    userPoolId: process.env.COGNITO_USER_POOL_ID,
    clientId: process.env.COGNITO_CLIENT_ID,
  }, 'AWS Cognito JWT verification configured');

  server.log.info({ env: appConfig.nodeEnv, logLevel: appConfig.logLevel }, 'Initializing server');

  const container = createContainer(logger);
  await container.initialize();
  const classifier = new Classifier(logger, linguaCompressor);
  // PRE-WARM CONNECTIONS
  const redisCache = new RedisCache(logger);
  await redisCache.connect(); // Connect once at startup

  const modelSelector = new ModelSelector(logger);
  // Pre-initialize classifier LLM (and log the exact model we'll use for small tasks)
  const classifierLLM = await modelSelector.getClassifierLLM();
  try {
    server.log.info({ classifierModel: classifierLLM.getModelInfo().modelId }, 'Classifier LLM pre-initialized');
  } catch (_) {
    server.log.info('Classifier LLM pre-initialized');
  }

  server.decorate('redisCache', redisCache);
  server.decorate('modelSelector', modelSelector);

  // Initialize Presentation Chains with corrected Redis methods
  const presentationOutlineChain = new PresentationOutlineChain(
    redisCache,
    logger,
    secrets.tavilyApiKey
  );

  const presentationContentChain = new PresentationContentChain(logger);

  server.decorate('presentationOutlineChain', presentationOutlineChain);
  server.decorate('presentationContentChain', presentationContentChain);


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
    onRequest: server.auth.require(),
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

  server.post<{ Body: ChatRequest }>('/chat/stream', {
    onRequest: server.auth.require(),
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
    const { sessionId } = request.body;

    try {
      server.log.info(
        {
          endpoint: '/chat/stream',
          sessionId,
          messagePreview: request.body.message.substring(0, 100),
        },
        'Streaming chat request received'
      );

      reply.raw.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive',
        'Transfer-Encoding': 'chunked',
        'X-Accel-Buffering': 'no',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Cache-Control, Content-Type',
      });

      const tutorChain = container.getTutorChain();
      let tokenCount = 0;

      await tutorChain.chatStream(
        request.body,
        {
          onToken: (token: string) => {
            tokenCount++;
            reply.raw.write(`data: ${JSON.stringify({ type: 'token', content: token })}\n\n`);
          },
          onMetadata: (metadata: any) => {
            server.log.debug({
              endpoint: '/chat/stream',
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
                  sessionId,
                },
              }
            })}\n\n`);

            server.log.info(
              {
                endpoint: '/chat/stream',
                sessionId,
                modelUsed: result.modelUsed,
                levelUsed: result.levelUsed,
                streamLatency: result.latency,
                totalDuration: duration,
                answerLength: result.answer.length,
                tokensStreamed: tokenCount,
              },
              'Streaming chat completed'
            );

            reply.raw.end();
          },
          onError: (error) => {
            const duration = Date.now() - startTime;

            server.log.error({
              endpoint: '/chat/stream',
              sessionId,
              error: error.message,
              stack: error.stack,
              duration,
            }, 'Streaming chat failed');

            reply.raw.write(`data: ${JSON.stringify({
              type: 'error',
              content: {
                success: false,
                error: 'Streaming chat failed',
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
        endpoint: '/chat/stream',
        sessionId,
        error: error instanceof Error ? error.message : String(error),
        stack: error instanceof Error ? error.stack : undefined,
        duration,
      }, 'Streaming chat setup failed');

      if (!reply.sent) {
        reply.status(500);
        return {
          success: false,
          error: 'Streaming chat setup failed',
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
  }>('/evaluate/stream', {
    onRequest: server.auth.require(),
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
        'Cache-Control': 'no-cache, no-transform',
        'Connection': 'keep-alive',
        'Transfer-Encoding': 'chunked',
        'X-Accel-Buffering': 'no',
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Headers': 'Cache-Control, Content-Type',
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

  // server.post<{
  //   Body: {
  //     userQuery: string;
  //     classification: Classification;
  //     documents: Document[];
  //     userSubscription?: string;
  //     sessionId?: string;
  //   };
  // }>('/evaluate', {
  //   onRequest: server.auth.require(),
  //   schema: {
  //     body: {
  //       type: 'object',
  //       required: ['userQuery', 'classification', 'documents'],
  //       properties: {
  //         userQuery: { type: 'string' },
  //         classification: {
  //           type: 'object',
  //           properties: {
  //             subject: { type: 'string' },
  //             level: { type: 'string' },
  //             confidence: { type: 'number' },
  //             intent: { type: 'string' },
  //           },
  //         },
  //         documents: {
  //           type: 'array',
  //           items: { type: 'object' },
  //         },
  //         userSubscription: { type: 'string' },
  //         sessionId: { type: 'string' },
  //       },
  //     },
  //   }
  // }, async (request, reply) => {
  //   const startTime = Date.now();
  //   const { userQuery, classification, documents, userSubscription, sessionId } = request.body;

  //   if (!userQuery || userQuery.trim() === '') {
  //     server.log.warn({ endpoint: '/evaluate' }, 'Empty userQuery received');
  //     reply.status(400);
  //     return { error: 'userQuery is required and cannot be empty' };
  //   }

  //   if (!classification) {
  //     server.log.warn({ endpoint: '/evaluate' }, 'Missing classification');
  //     reply.status(400);
  //     return { error: 'classification is required' };
  //   }

  //   if (!Array.isArray(documents)) {
  //     server.log.warn({ endpoint: '/evaluate' }, 'Invalid documents format');
  //     reply.status(400);
  //     return { error: 'documents must be an array' };
  //   }

  //   try {
  //     server.log.info(
  //       {
  //         endpoint: '/evaluate',
  //         sessionId,
  //         queryPreview: userQuery.substring(0, 80),
  //         subject: classification.subject,
  //         level: classification.level,
  //         confidence: classification.confidence,
  //         intent: (classification as any).intent,
  //         docCount: documents.length,
  //         subscription: userSubscription || 'free',
  //       },
  //       'Evaluation started'
  //     );

  //     const tutorChain = container.getTutorChain();
  //     const evaluateResult = await (tutorChain as any).evaluate(
  //       userQuery,
  //       classification,
  //       documents,
  //       userSubscription || 'free',
  //       sessionId
  //     );

  //     const duration = Date.now() - startTime;

  //     server.log.info(
  //       {
  //         endpoint: '/evaluate',
  //         sessionId,
  //         modelUsed: evaluateResult.modelUsed,
  //         levelUsed: evaluateResult.levelUsed,
  //         evaluateLatency: evaluateResult.latency,
  //         totalDuration: duration,
  //         answerLength: evaluateResult.answer.length,
  //       },
  //       'Evaluation completed'
  //     );

  //     reply.type('application/json');
  //     return {
  //       success: true,
  //       data: {
  //         answer: evaluateResult.answer,
  //         modelUsed: evaluateResult.modelUsed,
  //         levelUsed: evaluateResult.levelUsed,
  //         latency: evaluateResult.latency,
  //         classification,
  //         sources: documents,
  //         sessionId,
  //       },
  //     };
  //   } catch (error) {
  //     const duration = Date.now() - startTime;

  //     server.log.error({ 
  //       endpoint: '/evaluate',
  //       sessionId,
  //       error: error instanceof Error ? error.message : String(error),
  //       stack: error instanceof Error ? error.stack : undefined,
  //       duration,
  //     }, 'Evaluation failed');

  //     reply.status(500);
  //     return {
  //       success: false,
  //       error: 'Evaluation failed',
  //       message: error instanceof Error ? error.message : 'unknown error',
  //     };
  //   }
  // });

  server.post<{ Body: SlideOutlineRequest }>(
    '/presentations/outline',
    {
      onRequest: server.auth.require(),
    },
    async (request, reply) => {
      const startTime = Date.now();
      try {
        const userId = (request as any).user?.sub || 'unknown-user';

        const result = await presentationOutlineChain.generateOutline({
          ...request.body,
          userId,
        });

        const duration = Date.now() - startTime;
        server.log.info(
          { slideId: result.slideId, duration },
          'Outline generated'
        );

        return { success: true, data: result };
      } catch (error) {
        const duration = Date.now() - startTime;
        server.log.error({ error, duration }, 'Outline generation failed');
        reply.status(500);
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error',
        };
      }
    }
  );

  server.get<{ Params: { slideId: string } }>(
    '/presentations/:slideId/outline',
    {
      onRequest: server.auth.require(),
    },
    async (request, reply) => {
      try {
        const result = await presentationOutlineChain.getOutline(
          request.params.slideId
        );
        return { success: true, data: result };
      } catch (error) {
        server.log.warn(
          { slideId: request.params.slideId },
          'Outline not found'
        );
        reply.status(404);
        return { success: false, error: 'Outline not found' };
      }
    }
  );

  server.put<{
    Params: { slideId: string };
    Body: { updates: any[] };
  }>(
    '/presentations/:slideId/outline',
    {
      onRequest: server.auth.require(),
    },
    async (request, reply) => {
      try {
        const result = await presentationOutlineChain.updateOutline(
          request.params.slideId,
          request.body.updates
        );

        server.log.info(
          { slideId: request.params.slideId },
          'Outline updated'
        );
        return { success: true, data: result };
      } catch (error) {
        server.log.error(
          { slideId: request.params.slideId, error },
          'Outline update failed'
        );
        reply.status(500);
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error',
        };
      }
    }
  );

  server.post<{ Params: { slideId: string } }>(
    '/presentations/:slideId/generate',
    {
      onRequest: server.auth.require(),
    },
    async (request, reply) => {
      const startTime = Date.now();
      try {
        // Get outline from Redis using checkDirectCache
        const outlineData = await presentationOutlineChain.getOutline(
          request.params.slideId
        );

        server.log.info(
          { slideId: request.params.slideId },
          'Generating presentation content'
        );

        // Generate content for all slides (fetches images from Unsplash)
        const slidesContent = await presentationContentChain.generateSlideContent(
          outlineData.outline,
          outlineData.webSearchResults || ''
        );

        const response = {
          slideId: request.params.slideId,
          userId: outlineData.userId,
          title: outlineData.title,
          status: 'READY' as const,
          slidesContent,
          totalSlides: outlineData.noOfSlides,
          createdAt: outlineData.createdAt,
          updatedAt: new Date().toISOString(),
        };

        // Store final presentation in Redis using storeDirectCache
        const cacheKey = `presentation:final:${request.params.slideId}`;
        await redisCache.storeDirectCache(
          cacheKey,
          JSON.stringify(response),
          'presentation',
          { slideId: request.params.slideId, userId: outlineData.userId }
        );

        const duration = Date.now() - startTime;
        server.log.info(
          { slideId: request.params.slideId, duration },
          'Presentation generated'
        );

        return { success: true, data: response };
      } catch (error) {
        const duration = Date.now() - startTime;
        server.log.error(
          { slideId: request.params.slideId, error, duration },
          'Presentation generation failed'
        );
        reply.status(500);
        return {
          success: false,
          error: error instanceof Error ? error.message : 'Unknown error',
        };
      }
    }
  );

  server.get<{ Params: { slideId: string } }>(
    '/presentations/:slideId',
    {
      onRequest: server.auth.require(),
    },
    async (request, reply) => {
      try {
        const cacheKey = `presentation:final:${request.params.slideId}`;
        const cached = await redisCache.checkDirectCache(
          cacheKey,
          'presentation'
        );

        if (!cached) {
          reply.status(404);
          return { success: false, error: 'Presentation not found' };
        }

        return { success: true, data: JSON.parse(cached.response) };
      } catch (error) {
        server.log.warn(
          { slideId: request.params.slideId },
          'Presentation not found'
        );
        reply.status(404);
        return { success: false, error: 'Presentation not found' };
      }
    }
  );



  server.post<{ Body: { query: string } }>('/classify', {
    onRequest: server.auth.require(),
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
  }>('/rerank', {
    onRequest: server.auth.require(),

  }, async (request, reply) => {
    const startTime = Date.now();
    const { documents, query, topK } = request.body;

    if (!documents || documents.length === 0) {
      server.log.warn({ endpoint: '/rerank' }, 'No documents to rerank');
      reply.status(400);
      return {
        error: 'No documents provided for reranking',
        success: false
      };
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
      endpoints: ['/chat', '/evaluate/stream', '/classify', '/rerank', '/health', '/ready'],
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

// Type augmentation for secrets
declare module 'fastify' {
  interface FastifyInstance {
    secrets: {
      cohereApiKey: string;
      tavilyApiKey: string;
      langsmithApiKey: string;
      bedrockApiKey: string;
      redisUrl: string;
      redisPassword: string;
      awsAccessKeyId: string;
      awsSecretAccessKey: string;
    };
  }
}