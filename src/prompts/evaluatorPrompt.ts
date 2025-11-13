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
   * Streaming evaluation method
   */
  async evaluateStreaming(
    input: EvaluatePromptInput,
    callbacks: {
      onToken: (token: string) => void;
      onMetadata: (metadata: any) => void;
      onComplete: (result: EvaluatePromptOutput) => void;
      onError: (error: Error) => void;
    }
  ): Promise<void> {
    const startTime = Date.now();

    this.logger.info(
      {
        query: input.userQuery.substring(0, 80),
        subject: input.classification.subject,
        confidence: input.classification.confidence,
        intent: (input.classification as any).intent,
      },
      '[EvaluatePrompt] Starting streaming evaluation'
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
        '[EvaluatePrompt] Model tier selected for streaming'
      );

      // Send metadata about model selection
      callbacks.onMetadata({
        modelTier,
        subject: input.classification.subject,
        confidence: input.classification.confidence,
      });

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
        '[EvaluatePrompt] Tier-specific LLM created for streaming'
      );

      // STEP 4: Build evaluation prompt with all context
      const prompt = buildEvaluationPrompt(
        input.userQuery,
        input.classification,
        input.topDocument,
        (input.classification as any).intent || 'factual_retrieval',
        input.userPrefs
      );

      this.logger.debug(
        { promptLength: prompt.length, tier: modelTier },
        '[EvaluatePrompt] Prompt formatted for streaming'
      );

      let fullAnswer = '';

      // STEP 5: Stream from tier-specific LLM

      await llm.stream(prompt, {
        onToken: (token: string) => {
          fullAnswer += token;
          callbacks.onToken(token);
        },
        onComplete: () => {
          const latency = Date.now() - startTime;

          this.logger.info(
            {
              modelTier,
              latency,
              answerLength: fullAnswer.length,
              modelInfo: llm.getModelInfo(),
            },
            '[EvaluatePrompt] Streaming evaluation complete'
          );

          callbacks.onComplete({
            answer: fullAnswer,
            modelUsed: llm.getModelInfo().modelId,
            levelUsed: modelTier,
            latency,
          });
        },
        onError: (error: Error) => {
          this.logger.error(
            { error, query: input.userQuery.substring(0, 80) },
            '[EvaluatePrompt] Streaming evaluation failed'
          );
          callbacks.onError(error);
        },
      });
    } catch (error) {
      this.logger.error(
        { error, query: input.userQuery.substring(0, 80) },
        '[EvaluatePrompt] Streaming evaluation setup failed'
      );
      callbacks.onError(error as Error);
    }
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
      const prompt = buildEvaluationPrompt(
        input.userQuery,
        input.classification,
        input.topDocument,
        (input.classification as any).intent || 'factual_retrieval',
        input.userPrefs
      );

      this.logger.debug(
        { promptLength: prompt.length, tier: modelTier },
        '[EvaluatePrompt] Prompt formatted'
      );

      // STEP 5: Invoke tier-specific LLM
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
   * Select model tier based on classification level
   * 
   * Rules:
   * - Respects the classification level directly
   * - Basic → basic model
   * - Intermediate → intermediate model
   * - Advanced → advanced model
   */
  private selectModelTier(classification: Classification): 'basic' | 'intermediate' | 'advanced' {
    const level = classification.level;
    const confidence = classification.confidence;

    this.logger.debug(
      { level, confidence },
      '[EvaluatePrompt] Selecting model tier based on classification level'
    );

    // Directly use the classification level
    if (level === 'basic') {
      this.logger.info('[EvaluatePrompt] Using BASIC model for basic-level question');
      return 'basic';
    }

    if (level === 'intermediate') {
      this.logger.info('[EvaluatePrompt] Using INTERMEDIATE model for intermediate-level question');
      return 'intermediate';
    }

    if (level === 'advanced') {
      this.logger.info('[EvaluatePrompt] Using ADVANCED model for advanced-level question');
      return 'advanced';
    }

    // Fallback: use intermediate as safe default
    this.logger.warn({ level }, '[EvaluatePrompt] Unknown level, defaulting to intermediate');
    return 'intermediate';
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