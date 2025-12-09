import pino from 'pino';
import { START, END, StateGraph, Annotation } from '@langchain/langgraph';
import { 
  Classification, 
  Document, 
  AssessmentCategory, 
  AssessmentRequest,
  EnrichedBlueprint,
  GeneratedQuestionOutput,
  QuestionBatchResultItem
} from '../types/index.js';
import { createBlueprintGenerator } from './generator/blueprintGenerator.ts';
import { createResearchBatchProcessor } from './generator/research.js';
import { PromptRefiner } from './generator/promptRefiner.ts';
import { QuestionBatchRunner } from './generator/batchRunner.ts';
import { CacheStage, createCacheStage, TopicWithCache } from './generator/cacheStage.js';
import { RedisCache } from '../cache/RedisCache.js';

/**
 * Pipeline:
 * START → blueprintNode → researchNode → promptRefinementNode → questionGenerationNode → END
 */

// Define state using LangGraph Annotation
const EducatorAgentAnnotation = Annotation.Root({
  userQuery: Annotation<string>({
    reducer: (x, y) => y ?? x,
    default: () => '',
  }),
  classification: Annotation<Classification>({
    reducer: (x, y) => y ?? x,
  }),
  assessmentCategory: Annotation<AssessmentCategory>({
    reducer: (x, y) => y ?? x,
    default: () => 'quiz' as AssessmentCategory,
  }),
  assessmentRequest: Annotation<AssessmentRequest | null>({
    reducer: (x, y) => y ?? x,
    default: () => null,
  }),
  cachedTopics: Annotation<TopicWithCache[]>({
    reducer: (x, y) => y ?? x,
    default: () => [],
  }),
  cachedTotal: Annotation<number>({
    reducer: (x, y) => y ?? x,
    default: () => 0,
  }),
  toGenerateTotal: Annotation<number>({
    reducer: (x, y) => y ?? x,
    default: () => 0,
  }),
  blueprint: Annotation<any>({
    reducer: (x, y) => y ?? x,
    default: () => null,
  }),
  webSearchResults: Annotation<Document[]>({
    reducer: (x, y) => y ?? x,
    default: () => [],
  }),
  kbResults: Annotation<Document[]>({
    reducer: (x, y) => y ?? x,
    default: () => [],
  }),
  refinedPrompts: Annotation<any[]>({
    reducer: (x, y) => y ?? x,
    default: () => [],
  }),
  generatedQuestions: Annotation<GeneratedQuestionOutput[]>({
    reducer: (x, y) => y ?? x,
    default: () => [],
  }),
  stepLogs: Annotation<Array<{
    step: string;
    status: 'pending' | 'running' | 'completed' | 'failed';
    metadata?: Record<string, any>;
    error?: string;
  }>>({
    reducer: (x, y) => (y ? [...x, ...y] : x),
    default: () => [],
  }),
});

export type EducatorAgentState = typeof EducatorAgentAnnotation.State;

export class EducatorAgent {
  private blueprintGenerator: any;
  private researchProcessor: any;
  private promptRefiner: PromptRefiner;
  private questionBatchRunner: QuestionBatchRunner;
  private cacheStage: CacheStage;
  private cache: RedisCache;
  private graph: any;
  private logger: pino.Logger;

  constructor(logger: pino.Logger) {
    this.logger = logger;
    this.cache = new RedisCache(logger);
    this.cacheStage = createCacheStage(this.cache, logger);
    this.blueprintGenerator = createBlueprintGenerator(logger);
    this.researchProcessor = createResearchBatchProcessor(logger);
    this.promptRefiner = new PromptRefiner(logger);
    this.questionBatchRunner = new QuestionBatchRunner(logger);
    this.graph = this.buildGraph();
    
    // Test Redis connection on startup
    this.testRedisConnection();
  }

  /**
   * Test Redis connection and log status
   */
  private async testRedisConnection(): Promise<void> {
    try {
      await this.cache.connect();
      this.logger.info('[Educator Agent] ✅ Redis cache connected and ready');
    } catch (error) {
      this.logger.error(
        { error },
        '[Educator Agent] ❌ Redis cache connection failed - caching will not work'
      );
    }
  }

  /**
   * Build LangGraph with nodes and edges
   * Pipeline: START → cacheNode → blueprintNode → researchNode → promptRefinementNode → questionGenerationNode → END
   */
  private buildGraph() {
    const workflow = new StateGraph(EducatorAgentAnnotation)
      // Add nodes
      .addNode('cacheNode', this.cacheNode.bind(this))
      .addNode('blueprintNode', this.blueprintNode.bind(this))
      .addNode('researchNode', this.researchNode.bind(this))
      .addNode('promptRefinementNode', this.promptRefinementNode.bind(this))
      .addNode('questionGenerationNode', this.questionGenerationNode.bind(this))
      // Add edges
      .addEdge(START, 'cacheNode')
      .addEdge('cacheNode', 'blueprintNode')
      .addEdge('blueprintNode', 'researchNode')
      .addEdge('researchNode', 'promptRefinementNode')
      .addEdge('promptRefinementNode', 'questionGenerationNode')
      .addEdge('questionGenerationNode', END);

    return workflow.compile();
  }

  /**
   * NODE: Cache Stage
   * Runs FIRST before any other processing
   * Input: assessmentRequest with topics
   * Output: cachedTopics with cached questions attached, modified noOfQuestions
   */
  private async cacheNode(
    state: EducatorAgentState
  ): Promise<Partial<EducatorAgentState>> {
    const startTime = Date.now();

    this.logger.info(
      { step: 'cache' },
      '[Educator Agent] Starting cache stage'
    );

    try {
      // Ensure cache is connected
      await this.cache.connect();

      // Reset session dedupe for new request
      this.cacheStage.resetSession();

      // Extract request
      let request = state.assessmentRequest;
      
      if (!request && state.userQuery) {
        // Fallback: create basic request from userQuery
        const match = state.userQuery.match(/(\d+)\s+questions?/i);
        const questionCount = match ? parseInt(match[1], 10) : 20;
        
        request = {
          examTags: ['General'],
          subject: state.classification.subject || 'General',
          totalQuestions: questionCount,
          topics: [], // Will be generated by blueprint
        };
      }

      // If no topics yet, skip cache (will be handled after blueprint)
      if (!request?.topics || request.topics.length === 0) {
        this.logger.info(
          { step: 'cache' },
          '[Educator Agent] No topics provided, skipping cache stage'
        );

        const duration = Date.now() - startTime;

        return {
          cachedTopics: [],
          cachedTotal: 0,
          toGenerateTotal: request?.totalQuestions || 0,
          stepLogs: [
            ...state.stepLogs,
            {
              step: 'cache',
              status: 'completed',
              metadata: {
                duration,
                skipped: true,
                reason: 'no_topics_provided',
              },
            },
          ],
        };
      }

      // Run cache stage
      const cacheResult = await this.cacheStage.process(
        request.topics,
        request.subject,
        request.examTags
      );

      const duration = Date.now() - startTime;

      this.logger.info(
        {
          step: 'cache',
          cachedTotal: cacheResult.cachedTotal,
          toGenerateTotal: cacheResult.toGenerateTotal,
          cacheHitRate: cacheResult.cachedTotal / (cacheResult.cachedTotal + cacheResult.toGenerateTotal),
          stats: cacheResult.cacheStats,
        },
        '[Educator Agent] ✅ Cache stage complete'
      );

      return {
        cachedTopics: cacheResult.topics,
        cachedTotal: cacheResult.cachedTotal,
        toGenerateTotal: cacheResult.toGenerateTotal,
        stepLogs: [
          ...state.stepLogs,
          {
            step: 'cache',
            status: 'completed',
            metadata: {
              duration,
              cachedTotal: cacheResult.cachedTotal,
              toGenerateTotal: cacheResult.toGenerateTotal,
              cacheHitRate: cacheResult.cachedTotal / (cacheResult.cachedTotal + cacheResult.toGenerateTotal),
              stats: cacheResult.cacheStats,
            },
          },
        ],
      };
    } catch (error) {
      const duration = Date.now() - startTime;
      this.logger.error({ error }, '[Educator Agent] Cache stage failed');

      return {
        cachedTopics: [],
        cachedTotal: 0,
        toGenerateTotal: state.assessmentRequest?.totalQuestions || 0,
        stepLogs: [
          ...state.stepLogs,
          {
            step: 'cache',
            status: 'failed',
            error: error instanceof Error ? error.message : String(error),
            metadata: { duration },
          },
        ],
      };
    }
  }

  /**
   * NODE: Blueprint Generation
   * Input: assessmentRequest + classification
   * Output: blueprint in standardized format
   * 
   * RULE: Only runs if topics are missing/empty from the request
   */
  private async blueprintNode(
    state: EducatorAgentState
  ): Promise<Partial<EducatorAgentState>> {
    const startTime = Date.now();

    this.logger.info(
      { step: 'blueprint' },
      '[Educator Agent] Starting blueprint generation'
    );

    try {
      // Extract request from state (added in executeFromRequest) or create from userQuery
      let request = state.assessmentRequest;
      
      if (!request && state.userQuery) {
        // Fallback: create basic request from userQuery
        const match = state.userQuery.match(/(\d+)\s+questions?/i);
        const questionCount = match ? parseInt(match[1], 10) : 20;
        
        request = {
          examTags: ['General'],
          subject: state.classification.subject || 'General',
          totalQuestions: questionCount,
        };
      }

      // Check if topics were provided by user (and already cached)
      if (state.cachedTopics && state.cachedTopics.length > 0) {
        this.logger.info(
          { 
            action: 'bypass_blueprint', 
            reason: 'topics_provided',
            topicsCount: state.cachedTopics.length,
          },
          '[Educator Agent] Skipping blueprint - using cached topics'
        );

        const duration = Date.now() - startTime;

        // Build blueprint from cached topics
        const blueprint = {
          examTags: request?.examTags || [],
          subject: request?.subject || 'General',
          totalQuestions: state.toGenerateTotal, // Only generate what's not cached
          topics: state.cachedTopics.map(t => ({
            topicName: t.topicName,
            level: t.level,
            noOfQuestions: t.noOfQuestions, // Already reduced by cache stage
          })),
        };

        return {
          blueprint,
          stepLogs: [
            ...state.stepLogs,
            {
              step: 'blueprint',
              status: 'completed',
              metadata: {
                duration,
                topics: blueprint.topics.length,
                totalQuestions: blueprint.totalQuestions,
                bypassReason: 'topics_provided',
              },
            },
          ],
        };
      }
      
      // Generate blueprint with LLM
      const blueprint = await this.blueprintGenerator.generateBlueprint(
        request,
        state.classification,
        state.assessmentCategory
      );

      const duration = Date.now() - startTime;

      return {
        blueprint,
        stepLogs: [
          ...state.stepLogs,
          {
            step: 'blueprint',
            status: 'completed',
            metadata: {
              duration,
              topics: blueprint.topics.length,
              totalQuestions: blueprint.totalQuestions,
            },
          },
        ],
      };
    } catch (error) {
      const duration = Date.now() - startTime;
      this.logger.error({ error }, '[Educator Agent] Blueprint generation failed');

      return {
        stepLogs: [
          ...state.stepLogs,
          {
            step: 'blueprint',
            status: 'failed',
            error: error instanceof Error ? error.message : String(error),
            metadata: { duration },
          },
        ],
      };
    }
  }

  /**
   * NODE: Research (Parallel batch execution per topic)
   * Input: blueprint with topics
   * Output: enriched blueprint with web1-3 and kb1-3 per topic
   */
  private async researchNode(
    state: EducatorAgentState
  ): Promise<Partial<EducatorAgentState>> {
    const startTime = Date.now();

    if (!state.blueprint) {
      this.logger.error('[Educator Agent] Blueprint is missing for research');
      return {
        stepLogs: [
          ...state.stepLogs,
          {
            step: 'research',
            status: 'failed',
            error: 'Blueprint not found',
          },
        ],
      };
    }

    // Filter out topics with noOfQuestions == 0 (fully cached)
    const topicsToResearch = state.blueprint.topics.filter((t: any) => t.noOfQuestions > 0);
    const skippedTopics = state.blueprint.topics.filter((t: any) => t.noOfQuestions === 0);

    this.logger.info(
      { 
        step: 'research', 
        totalTopics: state.blueprint.topics.length,
        topicsToResearch: topicsToResearch.length,
        skippedTopics: skippedTopics.length,
      },
      '[Educator Agent] Starting parallel research (skipping fully cached topics)'
    );

    try {
      // If no topics need research, return early
      if (topicsToResearch.length === 0) {
        this.logger.info(
          { step: 'research' },
          '[Educator Agent] All topics fully cached, skipping research'
        );

        const duration = Date.now() - startTime;

        return {
          blueprint: state.blueprint, // Keep blueprint as-is
          stepLogs: [
            ...state.stepLogs,
            {
              step: 'research',
              status: 'completed',
              metadata: {
                duration,
                skipped: true,
                reason: 'all_topics_fully_cached',
              },
            },
          ],
        };
      }

      // Enrich only topics that need generation
      const blueprintForResearch = {
        ...state.blueprint,
        topics: topicsToResearch,
      };

      const enrichedBlueprint = await this.researchProcessor.enrichBlueprint(
        blueprintForResearch,
        state.classification,
        5 // maxConcurrency for parallel research
      );

      // Merge back skipped topics (no research results)
      const finalBlueprint = {
        ...enrichedBlueprint,
        topics: [
          ...enrichedBlueprint.topics,
          ...skippedTopics.map((t: any) => ({ ...t, skipResearch: true })),
        ],
      };

      const duration = Date.now() - startTime;

      // Count successful research results
      const topicsWithWeb = enrichedBlueprint.topics.filter((t: any) => t.web1).length;
      const topicsWithKB = enrichedBlueprint.topics.filter((t: any) => t.kb1).length;

      this.logger.info(
        {
          step: 'research',
          topicsCount: finalBlueprint.topics.length,
          topicsResearched: enrichedBlueprint.topics.length,
          topicsSkipped: skippedTopics.length,
          topicsWithWeb,
          topicsWithKB,
          duration,
        },
        '[Educator Agent] ✅ Research complete'
      );

      return {
        blueprint: finalBlueprint, // Replace blueprint with enriched version
        stepLogs: [
          ...state.stepLogs,
          {
            step: 'research',
            status: 'completed',
            metadata: {
              duration,
              topicsCount: finalBlueprint.topics.length,
              topicsResearched: enrichedBlueprint.topics.length,
              topicsSkipped: skippedTopics.length,
              topicsWithWeb,
              topicsWithKB,
            },
          },
        ],
      };
    } catch (error) {
      const duration = Date.now() - startTime;
      this.logger.error({ error }, '[Educator Agent] Research failed');

      return {
        stepLogs: [
          ...state.stepLogs,
          {
            step: 'research',
            status: 'failed',
            error: error instanceof Error ? error.message : String(error),
            metadata: { duration },
          },
        ],
      };
    }
  }

  /**
   * NODE: Prompt Refinement
   * Input: blueprint topics + research results + userQuery
   * Output: refined prompts array (one per topic with embedded research)
   */
  private async promptRefinementNode(
    state: EducatorAgentState
  ): Promise<Partial<EducatorAgentState>> {
    const startTime = Date.now();

    if (!state.blueprint) {
      this.logger.error('[Educator Agent] Blueprint is missing for prompt refinement');
      return {
        stepLogs: [
          ...state.stepLogs,
          {
            step: 'promptRefinement',
            status: 'failed',
            error: 'Blueprint not found',
          },
        ],
      };
    }

    // Filter out topics with noOfQuestions == 0 (fully cached, skip prompt refinement)
    const topicsToRefine = state.blueprint.topics.filter((t: any) => t.noOfQuestions > 0);
    const skippedTopics = state.blueprint.topics.filter((t: any) => t.noOfQuestions === 0);

    this.logger.info(
      {
        step: 'promptRefinement',
        totalTopics: state.blueprint.topics.length,
        topicsToRefine: topicsToRefine.length,
        skippedTopics: skippedTopics.length,
      },
      '[Educator Agent] Starting prompt refinement (skipping fully cached topics)'
    );

    try {
      // If no topics need refinement, return early
      if (topicsToRefine.length === 0) {
        this.logger.info(
          { step: 'promptRefinement' },
          '[Educator Agent] All topics fully cached, skipping prompt refinement'
        );

        const duration = Date.now() - startTime;

        return {
          refinedPrompts: [],
          stepLogs: [
            ...state.stepLogs,
            {
              step: 'promptRefinement',
              status: 'completed',
              metadata: {
                duration,
                skipped: true,
                reason: 'all_topics_fully_cached',
              },
            },
          ],
        };
      }

      const refinedPrompts = await this.promptRefiner.refinePrompts(
        topicsToRefine, // Only refine topics that need generation
        state.blueprint.examTags,
        state.blueprint.subject,
        state.classification,
        5 // maxConcurrency for parallel prompt refinement
      );

      const duration = Date.now() - startTime;

      this.logger.info(
        {
          step: 'promptRefinement',
          promptCount: refinedPrompts.length,
          duration,
        },
        '[Educator Agent] Prompt refinement complete'
      );

      return {
        refinedPrompts,
        stepLogs: [
          ...state.stepLogs,
          {
            step: 'promptRefinement',
            status: 'completed',
            metadata: {
              duration,
              prompts: refinedPrompts.length,
            },
          },
        ],
      };
    } catch (error) {
      const duration = Date.now() - startTime;
      this.logger.error({ error }, '[Educator Agent] Prompt refinement failed');

      return {
        stepLogs: [
          ...state.stepLogs,
          {
            step: 'promptRefinement',
            status: 'failed',
            error: error instanceof Error ? error.message : String(error),
            metadata: { duration },
          },
        ],
      };
    }
  }

  /**
   * NODE: Question Generation (Batch)
   * Input: refinedPrompts + blueprint.selectedModel
   * Output: generatedQuestions array (final quiz with answers)
   */
  private async questionGenerationNode(
    state: EducatorAgentState
  ): Promise<Partial<EducatorAgentState>> {
    const startTime = Date.now();

    if (!state.blueprint || !state.refinedPrompts.length) {
      this.logger.error('[Educator Agent] Blueprint or refined prompts missing for question generation');
      return {
        stepLogs: [
          ...state.stepLogs,
          {
            step: 'questionGeneration',
            status: 'failed',
            error: 'Blueprint or refined prompts not found',
          },
        ],
      };
    }

    this.logger.info(
      {
        step: 'questionGeneration',
        prompts: state.refinedPrompts.length,
      },
      '[Educator Agent] Starting batch question generation'
    );

    try {
      const result = await this.questionBatchRunner.runBatch(
        state.refinedPrompts,
        state.assessmentCategory,
        state.classification,
        5 // max concurrency
      );

      const duration = Date.now() - startTime;

      // Flatten all generated questions from all topics
      const generatedQuestions = result.items.flatMap((item) => item.questions);

      // Merge cached questions with generated questions
      const cachedQuestions: GeneratedQuestionOutput[] = state.cachedTopics.flatMap((t, topicIndex) => 
        t.cached.map((q, qIndex) => ({
          slotId: topicIndex * 1000 + qIndex, // Generate unique slotId for cached questions
          q: q.question,
          options: q.options,
          answer: q.options.indexOf(q.correctAnswer) + 1, // Convert to 1-based index
          explanation: q.explanation || 'N/A',
        }))
      );

      const allQuestions = [...cachedQuestions, ...generatedQuestions];

      // Store newly generated questions to cache for future use
      await this.storeGeneratedQuestionsToCache(
        result.items,
        state.blueprint.subject,
        state.blueprint.examTags
      );

      this.logger.info(
        {
          step: 'questionGeneration',
          totalQuestions: allQuestions.length,
          generatedCount: generatedQuestions.length,
          cachedCount: cachedQuestions.length,
          successCount: result.successCount,
          errorCount: result.errorCount,
          storedToCache: generatedQuestions.length,
          duration,
        },
        '[Educator Agent] ✅ Question generation complete (merged with cache + stored new questions)'
      );

      return {
        generatedQuestions: allQuestions,
        stepLogs: [
          ...state.stepLogs,
          {
            step: 'questionGeneration',
            status: 'completed',
            metadata: {
              duration,
              totalQuestions: allQuestions.length,
              generatedCount: generatedQuestions.length,
              cachedCount: cachedQuestions.length,
              successCount: result.successCount,
              errorCount: result.errorCount,
              topics: result.items.map((i) => ({
                topic: i.topic,
                questions: i.questions.length,
              })),
            },
          },
        ],
      };
    } catch (error) {
      const duration = Date.now() - startTime;
      this.logger.error({ error }, '[Educator Agent] Question generation failed');

      return {
        stepLogs: [
          ...state.stepLogs,
          {
            step: 'questionGeneration',
            status: 'failed',
            error: error instanceof Error ? error.message : String(error),
            metadata: { duration },
          },
        ],
      };
    }
  }

  /**
   * Store generated questions to cache for future reuse
   */
  private async storeGeneratedQuestionsToCache(
    resultItems: QuestionBatchResultItem[],
    subject: string,
    examTags: string[]
  ): Promise<void> {
    try {
      await this.cache.connect();

      let totalStored = 0;

      for (const item of resultItems) {
        if (!item.questions || item.questions.length === 0) {
          this.logger.warn(
            { topic: item.topic },
            '[Educator Agent] No questions to cache for topic'
          );
          continue;
        }

        // Build cache query for this topic (must match format used in cacheStage)
        const cacheQuery = `Generate questions on ${item.topic} for ${subject} (${examTags.join(', ')})`;

        // Convert questions to cacheable format (array of question objects)
        const questionsData = JSON.stringify(item.questions);

        this.logger.debug(
          {
            topic: item.topic,
            cacheQuery,
            questionsCount: item.questions.length,
            sampleQuestion: item.questions[0]?.q?.substring(0, 50),
          },
          '[Educator Agent] Storing questions to cache'
        );

        // Store in both direct and semantic cache
        await Promise.all([
          // Direct cache (exact match)
          this.cache.storeDirectCache(
            cacheQuery,
            questionsData,
            subject,
            {
              topic: item.topic,
              examTags,
              questionCount: item.questions.length,
              timestamp: Date.now(),
            }
          ),
          // Semantic cache (fuzzy match)
          this.cache.storeSemanticCache(
            cacheQuery,
            questionsData,
            subject,
            {
              topic: item.topic,
              examTags,
              questionCount: item.questions.length,
              timestamp: Date.now(),
            }
          ),
        ]);

        totalStored += item.questions.length;

        this.logger.info(
          {
            topic: item.topic,
            questionsStored: item.questions.length,
            subject,
            cacheKey: cacheQuery,
          },
          '[Educator Agent] ✅ Stored questions to cache'
        );
      }

      this.logger.info(
        {
          totalQuestionsStored: totalStored,
          topicsStored: resultItems.filter(i => i.questions?.length > 0).length,
        },
        '[Educator Agent] ✅ Cache storage complete'
      );
    } catch (error) {
      this.logger.error(
        { 
          error: error instanceof Error ? error.message : String(error),
          stack: error instanceof Error ? error.stack : undefined,
        },
        '[Educator Agent] ❌ Failed to store questions to cache (non-critical)'
      );
      // Don't throw - cache storage failure shouldn't break the pipeline
    }
  }

  /**
   * Execute the graph
   * Runs: blueprintNode → researchNode → promptRefinementNode → questionGenerationNode
   */
  async execute(
    userQuery: string,
    classification: Classification,
    assessmentCategory: AssessmentCategory = 'quiz'
  ): Promise<EducatorAgentState> {
    const initialState: EducatorAgentState = {
      userQuery,
      classification,
      assessmentCategory,
      assessmentRequest: null,
      cachedTopics: [],
      cachedTotal: 0,
      toGenerateTotal: 0,
      blueprint: null,
      webSearchResults: [],
      kbResults: [],
      refinedPrompts: [],
      generatedQuestions: [],
      stepLogs: [],
    };

    this.logger.info(
      { query: userQuery.substring(0, 60) },
      '[Educator Agent] Starting pipeline execution'
    );

    try {
      const result = await this.graph.invoke(initialState);

      this.logger.info(
        {
          query: userQuery.substring(0, 60),
          blueprint: result.blueprint?.topics.length || 0,
          research: `${result.webSearchResults.length} web + ${result.kbResults.length} kb`,
          prompts: result.refinedPrompts.length,
          questions: result.generatedQuestions.length,
        },
        '[Educator Agent] ✅ Pipeline complete'
      );

      return result as EducatorAgentState;
    } catch (error) {
      this.logger.error({ error }, '[Educator Agent] Pipeline execution failed');
      throw error;
    }
  }

  /**
   * Execute from standardized request format
   * Handles both cases: topics provided or auto-generated
   */
  async executeFromRequest(
    request: AssessmentRequest,
    classification: Classification
  ): Promise<EducatorAgentState> {
    // Build a query string for the pipeline
    const query = `Generate ${request.totalQuestions} questions on ${request.subject} for ${request.examTags.join(', ')}`;
    
    // For now, use 'quiz' as default - we'll enhance this later
    const assessmentCategory: AssessmentCategory = 'quiz';

    this.logger.info(
      { 
        examTags: request.examTags, 
        subject: request.subject, 
        totalQuestions: request.totalQuestions,
        topicsProvided: request.topics?.length || 0
      },
      '[Educator Agent] Starting pipeline with standardized request'
    );

    const initialState: EducatorAgentState = {
      userQuery: query,
      classification,
      assessmentCategory,
      assessmentRequest: request, // Pass request through state
      cachedTopics: [],
      cachedTotal: 0,
      toGenerateTotal: 0,
      blueprint: null,
      webSearchResults: [],
      kbResults: [],
      refinedPrompts: [],
      generatedQuestions: [],
      stepLogs: [],
    };

    this.logger.info(
      { query: query.substring(0, 60) },
      '[Educator Agent] Starting pipeline execution'
    );

    try {
      const result = await this.graph.invoke(initialState);

      this.logger.info(
        {
          query: query.substring(0, 60),
          blueprint: result.blueprint?.topics.length || 0,
          research: `${result.webSearchResults.length} web + ${result.kbResults.length} kb`,
          prompts: result.refinedPrompts.length,
          questions: result.generatedQuestions.length,
        },
        '[Educator Agent] ✅ Pipeline complete'
      );

      return result as EducatorAgentState;
    } catch (error) {
      this.logger.error({ error }, '[Educator Agent] Pipeline execution failed');
      throw error;
    }
  }
}

export function createEducatorAgent(logger: pino.Logger): EducatorAgent {
  return new EducatorAgent(logger);
}