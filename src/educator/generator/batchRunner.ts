import pino from 'pino';
import { z } from 'zod';
import { RunnableLambda } from '@langchain/core/runnables';
import { modelConfigService } from '../../config/modelConfig.js';
import { createTierLLM } from '../../llm/tierLLM.js';
import { 
  Classification, 
  AssessmentCategory, 
  AssessmentQuestion,
  RefinedPrompt,
  GeneratedQuestion,
  GeneratedQuestionOutput,
  QuestionBatchResultItem,
  QuestionBatchResult
} from '../../types/index.js';
import { buildBatchQuestionPrompt, getResponseFormatByIntent } from '../../utils/promptTemplates.js';

/**
 * QUESTION BATCH RUNNER (using LangChain .batch() method)
 * Final step in the pipeline: generates questions in parallel batches
 * 
 * Input: refined prompts (topic + context-aware prompt)
 * Output: generated questions with answers
 * 
 * Uses: Model selected from blueprint (basic/intermediate/advanced tier)
 * Uses LangChain's .batch() for cleaner concurrent execution
 */

// Question schema for validation
const QuestionSchema = z.object({
  question: z.string(),
  options: z.array(z.string()).optional(),
  correctAnswer: z.string(),
  explanation: z.string().optional(),
  difficulty: z.enum(['basic', 'intermediate', 'advanced']),
  topic: z.string(),
});

export class QuestionBatchRunner {
  constructor(private logger: pino.Logger) {}

  /**
   * Build prompt for question generation using patterns per difficulty level
   */
  private buildQuestionGenerationPrompt(refinedPrompt: RefinedPrompt, includeExplanation: boolean): string {
    const { topic, prompt } = refinedPrompt;
    const { noOfQuestions, patterns, numberRanges, optionStyle, avoid, context } = prompt;

    return buildBatchQuestionPrompt(
      topic,
      context,
      patterns,
      noOfQuestions,
      numberRanges,
      optionStyle,
      avoid,
      includeExplanation
    );
  }

  /**
   * Run all refined prompts through LLM in parallel batches using LangChain .batch()
   * Supports all assessment categories: quiz, mock_test, test_series
   */
  async runBatch(
    refinedPrompts: RefinedPrompt[],
    assessmentCategory: AssessmentCategory = 'quiz',
    classification?: Classification,
    maxConcurrency = 5
  ): Promise<QuestionBatchResult> {
    const startTime = Date.now();

    if (!refinedPrompts.length) {
      this.logger.warn('[BatchRunner] No refined prompts provided');
      return { 
        items: [], 
        totalQuestions: 0, 
        successCount: 0, 
        errorCount: 0, 
        duration: 0 
      };
    }

    this.logger.info(
      {
        prompts: refinedPrompts.length,
        topicNames: refinedPrompts.map(p => p.topic),
        assessmentCategory,
        maxConcurrency,
      },
      '[BatchRunner] Starting batch question generation with patterns'
    );

    // Get model config for question generation (intermediate tier)
    const modelConfig = modelConfigService.getModelConfig(
      { subject: 'question_generation', level: 'intermediate', confidence: 0.9 },
      'free'
    );

    const registryEntry = modelConfigService.getModelRegistryEntry(modelConfig.modelId);
    if (!registryEntry) {
      throw new Error(`Model registry entry not found for ${modelConfig.modelId}`);
    }

    // Create LLM instance
    const llm = createTierLLM(
      'intermediate',
      registryEntry,
      this.logger,
      modelConfig.temperature,
      3072 // Higher token limit for question generation
    );

    // Wrap our LLM in a RunnableLambda to make it batch-compatible
    const runnableLLM = RunnableLambda.from(async (refinedPrompt: RefinedPrompt) => {
      const includeExplanation = assessmentCategory !== 'test_series';
      const promptText = this.buildQuestionGenerationPrompt(refinedPrompt, includeExplanation);
      return await llm.generate(promptText);
    });

    this.logger.info(
      { 
        batchSize: refinedPrompts.length, 
        topicsToGenerate: refinedPrompts.map(p => `${p.topic} (${p.prompt.noOfQuestions}q)`),
        maxConcurrency 
      },
      '[BatchRunner] Executing .batch() with concurrent requests'
    );

    // Execute batch with maxConcurrency - pass refined prompts directly
    const batchResults = await runnableLLM.batch(
      refinedPrompts,
      { maxConcurrency }
    );

    // Process results
    const items: QuestionBatchResultItem[] = [];

    for (let i = 0; i < batchResults.length; i++) {
      const result = batchResults[i];
      const refinedPrompt = refinedPrompts[i];

      // Check if this is an error result
      if (!result || (typeof result === 'object' && result !== null && 'error' in result)) {
        const errorMsg = (result && typeof result === 'object' && 'error' in result) ? String((result as any).error) : 'Unknown error';
        
        this.logger.error(
          { error: errorMsg, topic: refinedPrompt.topic },
          '[BatchRunner] Generation failed for topic'
        );
        
        items.push({
          topic: refinedPrompt.topic,
          questions: [],
          error: errorMsg,
        });
        continue;
      }

      // Extract content from LLM response
      const rawResponse = typeof result === 'string' ? result : JSON.stringify(result);

      // Parse JSON response
      let questions: GeneratedQuestionOutput[] = [];
      try {
        let jsonText = rawResponse.trim();
        
        // Remove markdown code blocks if present
        const codeBlockMatch = jsonText.match(/```(?:json)?\s*\n?([\s\S]*?)\n?```/);
        if (codeBlockMatch) {
          jsonText = codeBlockMatch[1].trim();
        }
        
        // Extract JSON object
        const objectMatch = jsonText.match(/\{[\s\S]*\}/);
        if (objectMatch) {
          jsonText = objectMatch[0];
        }
        
        const parsed = JSON.parse(jsonText);
        
        // Support multiple response formats
        const questionsArray = parsed.questions || parsed.quiz || parsed.items || (Array.isArray(parsed) ? parsed : []);
        
        if (Array.isArray(questionsArray)) {
          questions = questionsArray
            .map((q: any, idx: number) => {
              const questionText = q.question || q.q || '';
              const options = q.options || q.choices || [];
              const correctAnswer = q.correctAnswer || q.answer || q.correct || '';
              const explanation = q.explanation || q.exp || q.reason || '';

              // Convert answer to 1-based index
              let answerIndex: number;
              if (typeof correctAnswer === 'number') {
                answerIndex = correctAnswer;
              } else if (typeof correctAnswer === 'string') {
                // Try to find matching option
                const matchIndex = options.findIndex((opt: string) => 
                  opt.toLowerCase().trim() === correctAnswer.toLowerCase().trim()
                );
                answerIndex = matchIndex >= 0 ? matchIndex + 1 : 1;
              } else {
                answerIndex = 1;
              }

              const output: GeneratedQuestionOutput = {
                slotId: idx + 1,
                q: questionText,
                options: options,
                answer: answerIndex,
              };

              // Add explanation only if not test_series
              if (assessmentCategory !== 'test_series' && explanation) {
                output.explanation = explanation;
              }

              return output;
            })
            .filter((q) => q.q && q.options.length >= 2);
        }

        this.logger.info(
          {
            topic: refinedPrompt.topic,
            questionsGenerated: questions.length,
            requestedCount: refinedPrompt.prompt.noOfQuestions,
          },
          '[BatchRunner] ✅ Generation complete for topic'
        );
      } catch (parseError) {
        this.logger.warn(
          { error: parseError, topic: refinedPrompt.topic, responsePreview: rawResponse.substring(0, 300) },
          '[BatchRunner] Failed to parse JSON'
        );
      }

      items.push({
        topic: refinedPrompt.topic,
        questions,
        rawResponse,
      });
    }

    const successCount = items.filter((i: QuestionBatchResultItem) => !i.error && i.questions.length > 0).length;
    const errorCount = items.length - successCount;
    const totalQuestions = items.reduce((sum: number, item: QuestionBatchResultItem) => sum + item.questions.length, 0);
    const duration = Date.now() - startTime;

    this.logger.info(
      { 
        successCount, 
        errorCount, 
        totalQuestions,
        duration: `${duration}ms` 
      },
      '[BatchRunner] ✅ Batch generation finished'
    );

    return { items, totalQuestions, successCount, errorCount, duration };
  }
}

export function createQuestionBatchRunner(logger: pino.Logger): QuestionBatchRunner {
  return new QuestionBatchRunner(logger);
}
