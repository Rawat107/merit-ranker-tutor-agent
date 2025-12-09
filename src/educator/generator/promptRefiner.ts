import pino from 'pino';
import { RunnableLambda } from '@langchain/core/runnables';
import { modelConfigService } from '../../config/modelConfig.js';
import { createTierLLM } from '../../llm/tierLLM.js';
import { buildPromptRefinementPrompt } from '../../utils/promptTemplates.js';
import type { EnrichedTopic, RefinedPrompt } from '../../types/index.js';

/**
 * PROMPT REFINER
 * Processes enriched topics from research and generates structured prompts
 * 
 * Input: Enriched topics with research (web + KB results)
 * Output: Array of refined prompts with patterns per difficulty level
 */

export class PromptRefiner {
  private readonly MAX_QUESTIONS_PER_PROMPT = 5; // Max questions per LLM call for accuracy

  constructor(private logger: pino.Logger) {}

  /**
   * Split topics with >5 questions into multiple sub-topics
   * This prevents incomplete LLM outputs due to token limits
   * 
   * Example: Topic with 10 questions → 2 topics with 5 questions each
   */
  private splitTopicsForAccuracy(topics: EnrichedTopic[]): EnrichedTopic[] {
    const result: EnrichedTopic[] = [];

    for (const topic of topics) {
      if (topic.noOfQuestions <= this.MAX_QUESTIONS_PER_PROMPT) {
        // No split needed
        result.push(topic);
      } else {
        // Split into multiple sub-topics
        const numSplits = Math.ceil(topic.noOfQuestions / this.MAX_QUESTIONS_PER_PROMPT);
        const questionsPerSplit = Math.ceil(topic.noOfQuestions / numSplits);
        
        this.logger.info(
          {
            topic: topic.topicName,
            originalQuestions: topic.noOfQuestions,
            splits: numSplits,
            questionsPerSplit,
          },
          '[PromptRefiner] Splitting topic for accuracy'
        );

        for (let i = 0; i < numSplits; i++) {
          const remainingQuestions = topic.noOfQuestions - (i * questionsPerSplit);
          const questionsForThisSplit = Math.min(questionsPerSplit, remainingQuestions);

          result.push({
            ...topic,
            noOfQuestions: questionsForThisSplit,
            // Keep same topic name - batchRunner will merge them back
            topicName: topic.topicName,
          });
        }
      }
    }

    return result;
  }

  /**
   * Main entry point: Process all enriched topics in batch
   * 
   * IMPORTANT: Splits topics with >5 questions into multiple sub-topics
   * to prevent incomplete LLM outputs due to token limits
   */
  async refinePrompts(
    enrichedTopics: EnrichedTopic[],
    examTags: string[],
    subject: string,
    classification: { level: string },
    maxConcurrency = 5
  ): Promise<RefinedPrompt[]> {
    const startTime = Date.now();

    // Split topics with >5 questions to prevent incomplete outputs
    const splitTopics = this.splitTopicsForAccuracy(enrichedTopics);

    this.logger.info(
      {
        originalTopicsCount: enrichedTopics.length,
        splitTopicsCount: splitTopics.length,
        topicNames: splitTopics.map((t: EnrichedTopic) => `${t.topicName} (${t.noOfQuestions}q)`),
        examTags,
        subject,
        maxConcurrency,
      },
      '[PromptRefiner] Starting batch prompt refinement (topics split for accuracy)'
    );

    try {
      // Create runnable for batch processing
      const runnableRefiner = RunnableLambda.from(async (topic: EnrichedTopic) => {
        return await this.refineSingleTopic(topic, examTags, subject, classification);
      });

      // Process all topics (including split ones) with maxConcurrency control
      const refinedPrompts = await runnableRefiner.batch(
        splitTopics,
        { maxConcurrency }
      );

      const duration = Date.now() - startTime;

      this.logger.info(
        {
          promptsGenerated: refinedPrompts.length,
          topicNamesProcessed: refinedPrompts.map(p => p.topic),
          duration,
        },
        '[PromptRefiner]  Batch refinement complete'
      );

      return refinedPrompts;
    } catch (error) {
      this.logger.error({ error }, '[PromptRefiner] Batch refinement failed');
      throw error;
    }
  }

  /**
   * Refine a single topic with LLM-based pattern extraction
   */
  private async refineSingleTopic(
    topic: EnrichedTopic,
    examTags: string[],
    subject: string,
    classification: { level: string }
  ): Promise<RefinedPrompt> {
    const startTime = Date.now();

    this.logger.info(
      { topicName: topic.topicName, levels: topic.level },
      '[PromptRefiner] Processing topic'
    );

    try {
      // Get LLM for pattern extraction
      const modelConfig = modelConfigService.getModelConfig(
        { subject: 'prompt_refinement', level: 'intermediate', confidence: 0.9 },
        'free'
      );

      const modelRegistry = modelConfigService.getModelRegistryEntry(
        modelConfig.modelId
      );

      if (!modelRegistry) {
        throw new Error(`Model registry not found for ${modelConfig.modelId}`);
      }

      const llm = createTierLLM(
        'intermediate',
        modelRegistry,
        this.logger,
        0.3, // Low temperature for consistent pattern extraction
        2048
      );

      // Build context from research
      const researchContext = this.buildResearchContext(topic);

      // Build prompt for LLM to extract patterns
      const extractionPrompt = this.buildPatternExtractionPrompt(
        topic,
        examTags,
        subject,
        researchContext
      );

      // Call LLM
      const response = await llm.generate(extractionPrompt);

      // Parse response
      const parsed = this.parsePatternResponse(response, topic, examTags);

      const duration = Date.now() - startTime;

      this.logger.info(
        {
          topicName: topic.topicName,
          patternsExtracted: Object.values(parsed.prompt.patterns).flat().length,
          duration,
        },
        '[PromptRefiner] Topic refined'
      );

      return parsed;
    } catch (error) {
      const duration = Date.now() - startTime;
      this.logger.error(
        { 
          topicName: topic.topicName, 
          error: error instanceof Error ? error.message : String(error),
          stack: error instanceof Error ? error.stack : undefined,
          duration 
        },
        '[PromptRefiner] Topic refinement failed, using fallback'
      );

      // Return fallback prompt
      return this.createFallbackPrompt(topic, examTags, subject);
    }
  }

  /**
   * Build research context summary from web and KB results
   */
  private buildResearchContext(topic: EnrichedTopic): string {
    const parts: string[] = [];

    // Add web results
    if (topic.research.web && topic.research.web.length > 0) {
      parts.push('WEB RESEARCH:');
      topic.research.web.forEach((doc, i) => {
        if (doc && doc.text) {
          parts.push(`[${i + 1}] ${doc.text.substring(0, 300)}...`);
        }
      });
    }

    // Add KB results
    if (topic.research.kb && topic.research.kb.length > 0) {
      parts.push('\nKNOWLEDGE BASE:');
      topic.research.kb.forEach((doc, i) => {
        if (doc && doc.text) {
          parts.push(`[${i + 1}] ${doc.text.substring(0, 300)}...`);
        }
      });
    }

    return parts.join('\n');
  }

  /**
   * Build LLM prompt for pattern extraction
   */
  private buildPatternExtractionPrompt(
    topic: EnrichedTopic,
    examTags: string[],
    subject: string,
    researchContext: string
  ): string {
    return buildPromptRefinementPrompt(
      examTags,
      subject,
      topic.topicName,
      topic.level,
      topic.noOfQuestions,
      researchContext
    );
  }

  /**
   * Parse LLM response into RefinedPrompt
   */
  private parsePatternResponse(response: string, topic: EnrichedTopic, examTags: string[]): RefinedPrompt {
    try {
      // Remove markdown code blocks if present
      let cleaned = response.trim();
      if (cleaned.startsWith('```json')) {
        cleaned = cleaned.substring(7);
      }
      if (cleaned.startsWith('```')) {
        cleaned = cleaned.substring(3);
      }
      if (cleaned.endsWith('```')) {
        cleaned = cleaned.substring(0, cleaned.length - 3);
      }
      cleaned = cleaned.trim();

      const parsed = JSON.parse(cleaned);

      // Validate structure
      if (!parsed.prompt || !parsed.prompt.patterns) {
        throw new Error('Invalid response structure');
      }

      // CRITICAL: Force the original topic name (LLM sometimes changes it)
      parsed.topic = topic.topicName;

      return parsed;
    } catch (error) {
      this.logger.warn(
        { topicName: topic.topicName, error, response: response.substring(0, 200) },
        '[PromptRefiner] Failed to parse LLM response, using fallback'
      );

      return this.createFallbackPrompt(topic, examTags, '');
    }
  }

  /**
   * Create fallback prompt when LLM fails
   */
  private createFallbackPrompt(
    topic: EnrichedTopic,
    examTags: string[],
    subject: string
  ): RefinedPrompt {
    this.logger.info(
      { topicName: topic.topicName },
      '[PromptRefiner] Using fallback prompt generation'
    );

    // Create basic patterns for each level
    const patterns: { [key: string]: string[] } = {};
    
    topic.level.forEach(lvl => {
      patterns[lvl] = [
        `Calculate ${topic.topicName} with basic formulas`,
        `Solve ${topic.topicName} word problems`,
        `Apply ${topic.topicName} concepts to real scenarios`,
        `${topic.topicName} with multiple steps`,
        `Mixed ${topic.topicName} problems`,
      ];
    });

    return {
      topic: topic.topicName,
      prompt: {
        noOfQuestions: topic.noOfQuestions,
        patterns,
        numberRanges: {
          min: 1,
          max: 1000,
          decimals: false,
        },
        optionStyle: examTags[0] || 'standard',
        avoid: ['repetitive numbers', 'same phrasing', 'obvious patterns'],
        context: `Generate ${topic.noOfQuestions} questions on ${topic.topicName} covering fundamental concepts and applications.`,
      },
    };
  }
}

export function createPromptRefiner(logger: pino.Logger): PromptRefiner {
  return new PromptRefiner(logger);
}