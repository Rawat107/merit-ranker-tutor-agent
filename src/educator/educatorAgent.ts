import pino from 'pino';
import { START, END, StateGraph, Annotation } from '@langchain/langgraph';
import { Classification, Document } from '../types/index.js';
import { createBlueprintGenerator } from './generator/blueprintGenerator.ts';
import { webSearchTool } from '../tools/webSearch.ts';
import { AWSKnowledgeBaseRetriever } from '../retriever/AwsKBRetriever.ts';
import { PromptRefiner } from './generator/promptRefiner.ts';
import { QuestionBatchRunner, GeneratedQuestion } from './generator/batchRunner.ts';

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
  generatedQuestions: Annotation<GeneratedQuestion[]>({
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
  private kbRetriever: AWSKnowledgeBaseRetriever;
  private promptRefiner: PromptRefiner;
  private questionBatchRunner: QuestionBatchRunner;
  private graph: any;
  private logger: pino.Logger;

  constructor(logger: pino.Logger) {
    this.logger = logger;
    this.blueprintGenerator = createBlueprintGenerator(logger);
    this.kbRetriever = new AWSKnowledgeBaseRetriever(logger);
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
   * Input: userQuery + classification
   * Output: blueprint with topics, model selection, strategy
   */
  private async blueprintNode(
    state: EducatorAgentState
  ): Promise<Partial<EducatorAgentState>> {
    const startTime = Date.now();

    this.logger.info(
      { step: 'blueprint', query: state.userQuery.substring(0, 60) },
      '[Educator Agent] Starting blueprint generation'
    );

    try {
      const blueprint = await this.blueprintGenerator.generateBlueprint(
        state.userQuery,
        state.classification
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
              model: blueprint.selectedModel.name,
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
   * NODE: Research (Parallel execution)
   * Input: userQuery, classification
   * Output: webSearchResults + kbResults (fetched in parallel)
   */
  private async researchNode(
    state: EducatorAgentState
  ): Promise<Partial<EducatorAgentState>> {
    const startTime = Date.now();

    this.logger.info(
      { step: 'research', query: state.userQuery.substring(0, 60) },
      '[Educator Agent] Starting parallel research'
    );

    try {
      // Run both in parallel
      const [webResults, kbResults] = await Promise.all([
        webSearchTool(
          state.userQuery,
          state.classification.subject,
          process.env.TAVILY_API_KEY || '',
          this.logger
        ),
        this.kbRetriever.getRelevantDocuments(state.userQuery, {
          subject: state.classification.subject,
          level: state.classification.level,
          k: 4,
        }),
      ]);

      // Take top 4 from each
      const topWebResults = webResults.slice(0, 4);
      const topKbResults = kbResults.slice(0, 4);

      const duration = Date.now() - startTime;

      this.logger.info(
        {
          step: 'research',
          webCount: topWebResults.length,
          kbCount: topKbResults.length,
          duration,
        },
        '[Educator Agent] Research complete'
      );

      return {
        webSearchResults: topWebResults,
        kbResults: topKbResults,
        stepLogs: [
          ...state.stepLogs,
          {
            step: 'research',
            status: 'completed',
            metadata: {
              duration,
              webSearchCount: topWebResults.length,
              kbCount: topKbResults.length,
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
        webResults: state.webSearchResults.length,
        kbResults: state.kbResults.length,
      },
      '[Educator Agent] Starting prompt refinement'
    );

    try {
      const refinedPrompts = await this.promptRefiner.refinePrompts(
        state.userQuery,
        state.blueprint.topics,
        state.webSearchResults,
        state.kbResults,
        state.classification.subject,
        state.classification.level
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
        model: state.blueprint.selectedModel.name,
      },
      '[Educator Agent] Starting batch question generation'
    );

    try {
      const result = await this.questionBatchRunner.runBatch(
        state.refinedPrompts,
        state.blueprint.selectedModel,
        state.classification.level,
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
                topic: i.topicName,
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
    classification: Classification
  ): Promise<EducatorAgentState> {
    const initialState: EducatorAgentState = {
      userQuery,
      classification,
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
}

export function createEducatorAgent(logger: pino.Logger): EducatorAgent {
  return new EducatorAgent(logger);
}