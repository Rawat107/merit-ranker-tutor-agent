import pino from 'pino';
import { START, END, StateGraph, Annotation } from '@langchain/langgraph';
import { 
  Classification, 
  Document, 
  AssessmentCategory, 
  AssessmentRequest,
  EnrichedBlueprint,
  GeneratedQuestionOutput
} from '../types/index.js';
import { createBlueprintGenerator } from './generator/blueprintGenerator.ts';
import { createResearchBatchProcessor } from './generator/research.js';
import { PromptRefiner } from './generator/promptRefiner.ts';
import { QuestionBatchRunner } from './generator/batchRunner.ts';

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
  private graph: any;
  private logger: pino.Logger;

  constructor(logger: pino.Logger) {
    this.logger = logger;
    this.blueprintGenerator = createBlueprintGenerator(logger);
    this.researchProcessor = createResearchBatchProcessor(logger);
    this.promptRefiner = new PromptRefiner(logger);
    this.questionBatchRunner = new QuestionBatchRunner(logger);
    this.graph = this.buildGraph();
  }

  /**
   * Build LangGraph with nodes and edges
   */
  private buildGraph() {
    const workflow = new StateGraph(EducatorAgentAnnotation)
      // Add nodes
      .addNode('blueprintNode', this.blueprintNode.bind(this))
      .addNode('researchNode', this.researchNode.bind(this))
      .addNode('promptRefinementNode', this.promptRefinementNode.bind(this))
      .addNode('questionGenerationNode', this.questionGenerationNode.bind(this))
      // Add edges
      .addEdge(START, 'blueprintNode')
      .addEdge('blueprintNode', 'researchNode')
      .addEdge('researchNode', 'promptRefinementNode')
      .addEdge('promptRefinementNode', 'questionGenerationNode')
      .addEdge('questionGenerationNode', END);

    return workflow.compile();
  }

  /**
   * NODE: Blueprint Generation
   * Input: assessmentRequest + classification
   * Output: blueprint in standardized format
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
        // Try to extract question count from the query
        const match = state.userQuery.match(/(\d+)\s+questions?/i);
        const questionCount = match ? parseInt(match[1], 10) : 20;
        
        request = {
          examTags: ['General'],
          subject: state.classification.subject || 'General',
          totalQuestions: questionCount,
        };
      }
      
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

    this.logger.info(
      { step: 'research', topicsCount: state.blueprint.topics.length },
      '[Educator Agent] Starting parallel research for all topics'
    );

    try {
      // Enrich blueprint with research results for each topic
      const enrichedBlueprint = await this.researchProcessor.enrichBlueprint(
        state.blueprint,
        state.classification,
        5 // maxConcurrency for parallel research
      );

      const duration = Date.now() - startTime;

      // Count successful research results
      const topicsWithWeb = enrichedBlueprint.topics.filter((t: any) => t.web1).length;
      const topicsWithKB = enrichedBlueprint.topics.filter((t: any) => t.kb1).length;

      this.logger.info(
        {
          step: 'research',
          topicsCount: enrichedBlueprint.topics.length,
          topicsWithWeb,
          topicsWithKB,
          duration,
        },
        '[Educator Agent] Research complete'
      );

      return {
        blueprint: enrichedBlueprint, // Replace blueprint with enriched version
        stepLogs: [
          ...state.stepLogs,
          {
            step: 'research',
            status: 'completed',
            metadata: {
              duration,
              topicsCount: enrichedBlueprint.topics.length,
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

    this.logger.info(
      {
        step: 'promptRefinement',
        topics: state.blueprint.topics.length,
      },
      '[Educator Agent] Starting prompt refinement with enriched topics'
    );

    try {
      const refinedPrompts = await this.promptRefiner.refinePrompts(
        state.blueprint.topics, // Already enriched with research from researchNode
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

      // Flatten all questions from all topics
      const allQuestions = result.items.flatMap((item) => item.questions);

      this.logger.info(
        {
          step: 'questionGeneration',
          totalQuestions: result.totalQuestions,
          successCount: result.successCount,
          errorCount: result.errorCount,
          duration,
        },
        '[Educator Agent] Question generation complete'
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
              totalQuestions: result.totalQuestions,
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

    const initialState: any = {
      userQuery: query,
      classification,
      assessmentCategory,
      assessmentRequest: request, // Pass request through state
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