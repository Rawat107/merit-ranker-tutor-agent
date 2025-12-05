import pino from 'pino';
import { z } from 'zod';
import { modelConfigService } from '../../config/modelConfig.ts';
import { buildPromptRefinementPrompt } from '../../utils/promptTemplates.ts';
import { createTierLLM } from '../../llm/tierLLM.ts';
import { Document } from '../../types/index.js';

/**
 * PROMPT REFINER
 * Second model in the pipeline
 * 
 * Input: blueprint topics + research (web + KB)
 * Output: array of refined prompts per topic
 * 
 * Uses: Claude Sonnet 3.5 (intermediate level)
 * Job: Take blueprint + research and create context-aware prompts
 */

const RefinedPromptSchema = z.object({
  topicName: z.string(),
  questionCount: z.number().min(1),
  difficulty: z.enum(['basic', 'intermediate', 'advanced']),
  prompt: z.string(),
  researchSources: z.array(z.string()),
  keywords: z.array(z.string()),
});

export type RefinedPrompt = z.infer<typeof RefinedPromptSchema>;

export class PromptRefiner {
  constructor(private logger: pino.Logger) {}

  /**
   * Main method: Refine prompts using research context
   */
  async refinePrompts(
    userQuery: string,
    blueprintTopics: Array<{
      topicName: string;
      description: string;
      difficulty: 'basic' | 'intermediate' | 'advanced';
      questionCount: number;
      priority: 'high' | 'medium' | 'low';
    }>,
    webSearchResults: Document[],
    awsKbResults: Document[],
    subject: string,
    level: 'basic' | 'intermediate' | 'advanced'
  ): Promise<RefinedPrompt[]> {
    try {
      this.logger.info(
        {
          topics: blueprintTopics.length,
          webResults: webSearchResults.length,
          kbResults: awsKbResults.length,
          subject,
          level,
        },
        '[PromptRefiner] Starting prompt refinement'
      );

      // STEP 1: Get prompt_refinement config (only intermediate level)
      const refinementConfig = modelConfigService.getModelConfig(
        { subject: 'prompt_refinement', level: 'intermediate', confidence: 0.9 },
        'free'
      );

      const modelRegistry = modelConfigService.getModelRegistryEntry(
        refinementConfig.modelId
      );

      if (!modelRegistry) {
        throw new Error(
          `Model registry not found for ${refinementConfig.modelId}`
        );
      }

      this.logger.debug(
        {
          modelId: refinementConfig.modelId,
          temperature: refinementConfig.temperature,
          maxTokens: refinementConfig.maxTokens,
        },
        '[PromptRefiner] Model config retrieved'
      );

      // STEP 2: Create LLM instance (always intermediate)
      const llm = createTierLLM(
        'intermediate',
        modelRegistry,
        this.logger,
        refinementConfig.temperature,
        refinementConfig.maxTokens
      );

      this.logger.debug(
        { modelInfo: llm.getModelInfo() },
        '[PromptRefiner] LLM instance created'
      );

      // STEP 3: Build refinement prompt with research context
      const refinementPrompt = buildPromptRefinementPrompt(
        userQuery,
        blueprintTopics,
        webSearchResults.map((d) => ({
          text: d.text,
          url: (d.metadata?.url as string) || undefined,
        })),
        awsKbResults.map((d) => ({
          text: d.text,
          source: (d.metadata?.source as string) || undefined,
        })),
        subject,
        level
      );

      this.logger.debug(
        { promptLength: refinementPrompt.length },
        '[PromptRefiner] Refinement prompt built'
      );

      // STEP 4: Call LLM to refine prompts
      const refinedJson = await llm.generate(refinementPrompt);

      this.logger.debug(
        { responseLength: refinedJson.length },
        '[PromptRefiner] LLM generated refined prompts'
      );

      // STEP 5: Parse JSON response
      let parsedRefinedPrompts: any[];
      try {
        parsedRefinedPrompts = JSON.parse(refinedJson);

        // Ensure it's an array
        if (!Array.isArray(parsedRefinedPrompts)) {
          parsedRefinedPrompts = [parsedRefinedPrompts];
        }
      } catch (parseError) {
        this.logger.warn(
          { error: parseError, response: refinedJson.substring(0, 200) },
          '[PromptRefiner] Failed to parse LLM response, using fallback'
        );
        parsedRefinedPrompts = this.fallbackRefinedPrompts(
          blueprintTopics,
          webSearchResults,
          awsKbResults
        );
      }

      // STEP 6: Validate and enhance each prompt
      const refinedPrompts: RefinedPrompt[] = parsedRefinedPrompts
        .map((p: any) => {
          try {
            // Validate structure
            return RefinedPromptSchema.parse({
              topicName: p.topicName || 'Unknown Topic',
              questionCount: p.questionCount || 5,
              difficulty: p.difficulty || level,
              prompt: p.prompt || 'Generate questions on this topic',
              researchSources: p.researchSources || [],
              keywords: p.keywords || [],
            });
          } catch (validationError) {
            this.logger.warn(
              { error: validationError, prompt: p },
              '[PromptRefiner] Validation failed for prompt, using enhanced version'
            );
            return {
              topicName: p.topicName || 'Unknown Topic',
              questionCount: p.questionCount || 5,
              difficulty: level,
              prompt:
                p.prompt ||
                `Generate ${p.questionCount || 5} questions on ${p.topicName || 'this topic'}`,
              researchSources: p.researchSources || ['web', 'kb'],
              keywords: p.keywords || [],
            };
          }
        })
        .filter((p) => p !== null);

      this.logger.info(
        {
          refinedCount: refinedPrompts.length,
          topicsCovered: refinedPrompts.map((p) => p.topicName),
        },
        '[PromptRefiner] ✅ Prompt refinement complete'
      );

      return refinedPrompts;
    } catch (error) {
      this.logger.error({ error }, '[PromptRefiner] ❌ Prompt refinement failed');
      throw error;
    }
  }

  /**
   * Fallback: Generate refined prompts heuristically
   */
  private fallbackRefinedPrompts(
    blueprintTopics: Array<{
      topicName: string;
      description: string;
      difficulty: string;
      questionCount: number;
      priority: string;
    }>,
    webSearchResults: Document[],
    awsKbResults: Document[]
  ): RefinedPrompt[] {
    this.logger.info(
      '[PromptRefiner] Using fallback refined prompts generation'
    );

    return blueprintTopics.map((topic, index) => {
      // Select research items for this topic
      const webSource =
        webSearchResults[index % webSearchResults.length]?.metadata?.url ||
        'web-search';
      const kbSource =
        awsKbResults[index % awsKbResults.length]?.metadata?.source ||
        'knowledge-base';

      return {
        topicName: topic.topicName,
        questionCount: topic.questionCount,
        difficulty: topic.difficulty as 'basic' | 'intermediate' | 'advanced',
        prompt: `Generate ${topic.questionCount} ${topic.difficulty} level questions on "${topic.topicName}". ${topic.description}`,
        researchSources: [webSource, kbSource],
        keywords: topic.topicName
          .toLowerCase()
          .split(' ')
          .slice(0, 3),
      };
    });
  }
}

export function createPromptRefiner(logger: pino.Logger): PromptRefiner {
  return new PromptRefiner(logger);
}