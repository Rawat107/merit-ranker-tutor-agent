import pino from 'pino';
import { z } from 'zod';
import { RunnableLambda } from '@langchain/core/runnables';
import { modelConfigService } from '../../config/modelConfig.js';
import { createTierLLM } from '../../llm/tierLLM.js';

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
  explanation: z.string(),
  difficulty: z.enum(['basic', 'intermediate', 'advanced']),
  topic: z.string(),
});

export type GeneratedQuestion = z.infer<typeof QuestionSchema>;

export interface RefinedPrompt {
  topicName: string;
  prompt: string;
  questionCount: number;
  difficulty: 'basic' | 'intermediate' | 'advanced';
  keywords: string[];
  researchSources: string[];
}

export interface QuestionBatchResultItem {
  topicName: string;
  questionCount: number;
  questions: GeneratedQuestion[];
  rawResponse: string;
  error?: string;
}

export interface QuestionBatchResult {
  items: QuestionBatchResultItem[];
  totalQuestions: number;
  successCount: number;
  errorCount: number;
  duration: number;
}

export class QuestionBatchRunner {
  constructor(private logger: pino.Logger) {}

  /**
   * Run all refined prompts through LLM in parallel batches using LangChain .batch()
   */
  async runBatch(
    refinedPrompts: RefinedPrompt[],
    selectedModel: {
      modelId: string;
      name: string;
      temperature: number;
    },
    level: 'basic' | 'intermediate' | 'advanced',
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
        model: selectedModel.name,
        modelId: selectedModel.modelId,
        maxConcurrency,
      },
      '[BatchRunner] Starting batch question generation using LangChain .batch()'
    );

    // Get model registry entry
    const registryEntry = modelConfigService.getModelRegistryEntry(selectedModel.modelId);
    if (!registryEntry) {
      throw new Error(`Model registry entry not found for ${selectedModel.modelId}`);
    }

    // Create LLM instance for this tier
    const llm = createTierLLM(
      level,
      registryEntry,
      this.logger,
      selectedModel.temperature,
      2048 // Higher token limit for question generation
    );

    // Wrap our LLM in a RunnableLambda to make it batch-compatible
    const runnableLLM = RunnableLambda.from(async (refinedPrompt: RefinedPrompt) => {
      // The refined prompt already contains complete instructions like:
      // "Generate 3 questions related to Python lists and tuples, covering..."
      // We just need to append the JSON format requirement
      const promptText = `${refinedPrompt.prompt}

CRITICAL: Respond with ONLY valid JSON. No other text before or after.

Format:
{
  "questions": [
    {
      "question": "Your question text?",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correctAnswer": "Exact text of correct option",
      "explanation": "Why this is correct"
    }
  ]
}`;

      return await llm.generate(promptText);
    });

    this.logger.info(
      { batchSize: refinedPrompts.length, maxConcurrency },
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
          { error: errorMsg, topic: refinedPrompt.topicName },
          '[BatchRunner] Generation failed for topic'
        );
        
        items.push({
          topicName: refinedPrompt.topicName,
          questionCount: refinedPrompt.questionCount,
          questions: [],
          rawResponse: '',
          error: errorMsg,
        });
        continue;
      }

      // Extract content from LLM response
      const rawResponse = typeof result === 'string' ? result : JSON.stringify(result);

      // Parse JSON response with robust extraction
      let questions: GeneratedQuestion[] = [];
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
            .map((q: any) => ({
              question: q.question || q.q || '',
              options: q.options || q.choices || [],
              correctAnswer: q.correctAnswer || q.answer || q.correct || '',
              explanation: q.explanation || q.exp || q.reason || '',
              difficulty: refinedPrompt.difficulty,
              topic: refinedPrompt.topicName,
            }))
            .filter((q) => q.question && q.options.length >= 2 && q.correctAnswer);

          // Validate with Zod
          questions = questions.filter((q) => {
            try {
              QuestionSchema.parse(q);
              return true;
            } catch {
              return false;
            }
          });
        }

        this.logger.info(
          {
            topic: refinedPrompt.topicName,
            questionsGenerated: questions.length,
            requestedCount: refinedPrompt.questionCount,
          },
          '[BatchRunner] ✅ Generation complete for topic'
        );
      } catch (parseError) {
        this.logger.warn(
          { error: parseError, topic: refinedPrompt.topicName, responsePreview: rawResponse.substring(0, 300) },
          '[BatchRunner] Failed to parse JSON'
        );
      }

      items.push({
        topicName: refinedPrompt.topicName,
        questionCount: refinedPrompt.questionCount,
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
