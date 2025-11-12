import { ILLM } from '../llm/ILLM.js';
import { ModelSelector } from '../llm/ModelSelector.js';
import { createTierLLM } from '../llm/tierLLM.js';
import { Classification, Document, LLMOptions } from '../types/index.js';
import { buildEvaluationPrompt } from '../utils/promptTemplates.js';
import { modelConfigService } from '../config/modelConfig.js';
import pino from 'pino';

export interface EvaluatePromptInput {
  userQuery: string;
  classification: Classification;
  topDocument: Document | null;
  userPrefs?: Record<string, any>;
  subscription?: string;
}

export interface EvaluatePromptOutput {
  answer: string;
  modelUsed: string;
  levelUsed: string;
  tokens?: number;
  latency?: number;
}

/**
 * Evaluates the final prompt using tier-specific LLMs (Basic/Intermediate/Advanced)
 * 
 * Confidence-based routing:
 * - confidence > 90% + Math/Reasoning/Logic → Intermediate LLM
 * - confidence > 90% + English/General/Simple → Basic LLM
 * - confidence ≤ 90% → Advanced LLM
 */
export class EvaluatePrompt {
  private modelSelector: ModelSelector;
  private logger: pino.Logger;

  constructor(modelSelector: ModelSelector, logger: pino.Logger) {
    this.modelSelector = modelSelector;
    this.logger = logger;
  }

  /**
   * Main evaluation method
   */
  async evaluate(input: EvaluatePromptInput): Promise<EvaluatePromptOutput> {
    const startTime = Date.now();

    this.logger.info(
      {
        query: input.userQuery.substring(0, 80),
        subject: input.classification.subject,
        confidence: input.classification.confidence,
        intent: (input.classification as any).intent,
      },
      '[EvaluatePrompt] Starting evaluation'
    );

    try {
      // STEP 1: Decide which model tier to use based on confidence
      const modelTier = this.selectModelTier(input.classification);

      this.logger.debug(
        {
          confidence: input.classification.confidence,
          tier: modelTier,
          subject: input.classification.subject,
          intent: (input.classification as any).intent,
        },
        '[EvaluatePrompt] Model tier selected'
      );

      // STEP 2: Get model config for the tier
      const config = modelConfigService.getModelConfig(
        {
          ...input.classification,
          level: modelTier,
        },
        input.subscription || 'free'
      );

      const registryEntry = modelConfigService.getModelRegistryEntry(config.modelId);

      if (!registryEntry) {
        throw new Error(`Registry entry not found for model ${config.modelId}`);
      }

      // STEP 3: Create tier-specific LLM using temperature and maxTokens from config
      const llm = createTierLLM(
        modelTier,
        registryEntry,
        this.logger,
        config.temperature,
        config.maxTokens
      );

      this.logger.info(
        {
          modelInfo: llm.getModelInfo(),
          tier: modelTier,
          temperature: config.temperature,
          maxTokens: config.maxTokens,
        },
        '[EvaluatePrompt] Tier-specific LLM created'
      );

      // STEP 4: Build evaluation prompt with all context
      const promptTemplate = buildEvaluationPrompt(
        input.userQuery,
        input.classification,
        input.topDocument,
        (input.classification as any).intent || 'factual_retrieval',
        input.userPrefs
      );

      // STEP 5: Format the prompt with user query
      const prompt = await promptTemplate.format({ query: input.userQuery });

      this.logger.debug(
        { promptLength: prompt.length, tier: modelTier },
        '[EvaluatePrompt] Prompt formatted'
      );

      // STEP 6: Invoke tier-specific LLM
      const answer = await llm.generate(prompt);

      const latency = Date.now() - startTime;

      this.logger.info(
        {
          modelTier,
          latency,
          answerLength: answer.length,
          modelInfo: llm.getModelInfo(),
        },
        '[EvaluatePrompt] Evaluation complete'
      );

      return {
        answer,
        modelUsed: llm.getModelInfo().modelId,
        levelUsed: modelTier,
        latency,
      };
    } catch (error) {
      this.logger.error(
        { error, query: input.userQuery.substring(0, 80) },
        '[EvaluatePrompt] Evaluation failed'
      );
      throw error;
    }
  }

  /**
   * Select model tier based on confidence and subject/intent
   */
  private selectModelTier(classification: Classification): 'basic' | 'intermediate' | 'advanced' {
    const confidence = classification.confidence;
    const subject = classification.subject;
    const intent = (classification as any).intent || 'factual_retrieval';

    // High confidence threshold
    if (confidence > 0.9) {
      return this.selectTierForHighConfidence(subject, intent);
    }

    // Low confidence: use most capable model
    return 'advanced';
  }

  /**
   * For high confidence (>90%), select between basic and intermediate
   */
  private selectTierForHighConfidence(
    subject: string,
    intent: string
  ): 'basic' | 'intermediate' {
    // Subjects that benefit from intermediate model
    const intermediateSubjects = ['math', 'reasoning', 'science'];
    const intermediateIntents = [
      'step_by_step_explanation',
      'problem_solving',
      'reasoning_puzzle',
      'comparative_analysis',
    ];

    const useIntermediate =
      intermediateSubjects.includes(subject) || intermediateIntents.includes(intent);

    return useIntermediate ? 'intermediate' : 'basic';
  }

  /**
   * Get temperature by model tier
   */
  private getTemperatureByTier(tier: 'basic' | 'intermediate' | 'advanced'): number {
    const temps: Record<string, number> = {
      basic: 0.2, // Lower temp for consistent, simple answers
      intermediate: 0.1, // Lower temp for accurate step-by-step
      advanced: 0.0, // Deterministic for rigorous answers
    };
    return temps[tier];
  }

  /**
   * Get max tokens by model tier
   */
  private getMaxTokensByTier(tier: 'basic' | 'intermediate' | 'advanced'): number {
    const tokenLimits: Record<string, number> = {
      basic: 800, // Basic answers are concise
      intermediate: 1200, // Room for step-by-step
      advanced: 2000, // Room for proofs and detailed reasoning
    };
    return tokenLimits[tier];
  }
}

/**
 * Factory function to create EvaluatePrompt instance
 */
export function createEvaluatePrompt(
  modelSelector: ModelSelector,
  logger: pino.Logger
): EvaluatePrompt {
  return new EvaluatePrompt(modelSelector, logger);
}