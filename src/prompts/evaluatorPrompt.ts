import { ILLM } from '../llm/ILLM.js';
import { ModelSelector } from '../llm/ModelSelector.js';
import { createTierLLM } from '../llm/tierLLM.js';
import { Classification, Document, LLMOptions, BaseMessage } from '../types/index.js';
import { buildEvaluationPrompt } from '../utils/promptTemplates.js';
import { modelConfigService } from '../config/modelConfig.js';
import { linguaCompressor } from '../compression/lingua_compressor.js';
import pino from 'pino';

/**
 * Get status message based on tier and subject
 */
function getStatusMessage(tier: 'basic' | 'intermediate' | 'advanced', subject: string): string {
  // Tier-based thinking messages
  const thinkingMessages = {
    basic: 'Thinking...',
    intermediate: 'Thinking harder...',
    advanced: 'Thinking deeply...'
  };

  // Subject-specific action messages
  const subjectMessages: Record<string, string> = {
    math: 'Solving the problem...',
    reasoning: 'Working through the logic...',
    science: 'Analyzing the concepts...',
    history: 'Researching historical context...',
    english_grammar: 'Checking grammar rules...',
    general_knowledge: 'Finding information...',
    current_affairs: 'Searching latest updates...'
  };

  return subjectMessages[subject] || thinkingMessages[tier];
}

export interface EvaluatePromptInput {
  userQuery: string;
  classification: Classification;
  topDocument: Document | null;
  userPrefs?: Record<string, string>;
  subscription?: string;
  conversationHistory?: string;
  userName?: string | null;
}

export interface EvaluatePromptOutput {
  answer: string;
  modelUsed: string;
  levelUsed: string;
  tokens?: number;
  latency?: number;
}

/**
 * Compress different parts of the prompt with different compression rates
 *
 * Strategy:
 * - System prompt: Light compression (rate=0.7 = keep 70%)
 * - Conversation history: Aggressive compression (rate=0.4 = keep 40%)
 * - Reference material: NO compression (critical info)
 * - User query: NO compression (user's actual question)
 *
 * FIX: Now receives pre-split parts from buildEvaluationPrompt()
 */
async function compressPromptSelectively(
  systemPrompt: string,
  conversationHistory: string,
  referenceAndQuery: string,
  logger: pino.Logger
): Promise<string> {
  const startTime = Date.now();

  try {
    logger.debug(
      {
        systemLength: systemPrompt.length,
        historyLength: conversationHistory.length,
        refQueryLength: referenceAndQuery.length
      },
      '[Compression] Starting selective compression'
    );

    // 1. Compress system prompt lightly (keep structure but remove redundancy)
    let compressedSystem = systemPrompt;
    if (systemPrompt.length > 300) {
      logger.debug({ originalLength: systemPrompt.length }, '[Compression] Compressing system prompt...');
      compressedSystem = await linguaCompressor.compress(systemPrompt, { rate: 0.7 });
      logger.info(
        {
          original: systemPrompt.length,
          compressed: compressedSystem.length,
          saved: systemPrompt.length - compressedSystem.length,
          ratio: ((1 - compressedSystem.length / systemPrompt.length) * 100).toFixed(1) + '%'
        },
        '[Compression] ✅ System prompt compressed'
      );
    }

    // 2. Compress conversation history aggressively (biggest token consumer)
    let compressedHistory = conversationHistory;
    if (conversationHistory.trim().length > 0) {
      logger.debug({ originalLength: conversationHistory.length }, '[Compression] Compressing conversation history...');
      compressedHistory = await linguaCompressor.compress(conversationHistory, { rate: 0.25 });
      logger.info(
        {
          original: conversationHistory.length,
          compressed: compressedHistory.length,
          saved: conversationHistory.length - compressedHistory.length,
          ratio: ((1 - compressedHistory.length / conversationHistory.length) * 100).toFixed(1) + '%'
        },
        '[Compression] ✅ Conversation history compressed'
      );
    }

    // 3. Keep reference material and user query as-is (NO compression)
    // These are critical and must remain unchanged

    // Combine all parts
    const finalPrompt = `${compressedSystem}${compressedHistory}${referenceAndQuery}`;

    const totalTime = Date.now() - startTime;
    const originalTotal = systemPrompt.length + conversationHistory.length + referenceAndQuery.length;
    const compressionRatio = (((originalTotal - finalPrompt.length) / originalTotal) * 100).toFixed(1);

    logger.info(
      {
        originalChars: originalTotal,
        compressedChars: finalPrompt.length,
        savedChars: originalTotal - finalPrompt.length,
        originalTokens: Math.floor(originalTotal / 4),
        compressedTokens: Math.floor(finalPrompt.length / 4),
        savedTokens: Math.floor((originalTotal - finalPrompt.length) / 4),
        compressionRatio: `${compressionRatio}%`,
        compressionTime: `${totalTime}ms`
      },
      '[Compression] ✅✅✅ SELECTIVE COMPRESSION COMPLETE'
    );

    return finalPrompt;
  } catch (error) {
    logger.error({ error }, '[Compression] ❌ Compression failed, using original');
    // Fallback: return uncompressed but properly formatted
    return `${systemPrompt}${conversationHistory}${referenceAndQuery}`;
  }
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
        intent: (input.classification as any).intent
      },
      '[EvaluatePrompt] Starting streaming evaluation'
    );

    try {
      // STEP 1: Decide which model tier to use based on classification level
      const modelTier = this.selectModelTier(input.classification);
      this.logger.debug(
        {
          confidence: input.classification.confidence,
          tier: modelTier,
          subject: input.classification.subject,
          intent: (input.classification as any).intent
        },
        '[EvaluatePrompt] Model tier selected for streaming'
      );

      // Send tier-based thinking status
      const statusMessage = getStatusMessage(modelTier, input.classification.subject);

      // Send metadata about model selection (status for frontend to display)
      callbacks.onMetadata({
        modelTier,
        subject: input.classification.subject,
        confidence: input.classification.confidence,
        status: statusMessage
      });

      // STEP 2: Get model config for the tier
      const config = modelConfigService.getModelConfig(
        {
          ...input.classification,
          level: modelTier
        },
        input.subscription || 'free'
      );

      const registryEntry = modelConfigService.getModelRegistryEntry(config.modelId);
      if (!registryEntry) {
        throw new Error(`Registry entry not found for model ${config.modelId}`);
      }

      // STEP 3: Create tier-specific LLM using temperature and maxTokens from config
      const llm = createTierLLM(modelTier, registryEntry, this.logger, config.temperature, config.maxTokens);

      this.logger.info(
        {
          modelInfo: llm.getModelInfo(),
          tier: modelTier,
          temperature: config.temperature,
          maxTokens: config.maxTokens
        },
        '[EvaluatePrompt] Tier-specific LLM created for streaming'
      );

      // STEP 4: Build evaluation prompt - now returns 3 parts
      const promptParts = buildEvaluationPrompt(
        input.userQuery,
        input.classification,
        input.topDocument,
        (input.classification as any).intent || 'factual_retrieval',
        input.userPrefs,
        input.conversationHistory,
        input.userName
      );

      // STEP 4.5: Apply selective compression (FIX: Main change)
      const prompt = await compressPromptSelectively(
        promptParts.systemPrompt,
        promptParts.conversationHistory,
        promptParts.referenceAndQuery,
        this.logger
      );

      this.logger.debug(
        {
          originalSystemLength: promptParts.systemPrompt.length,
          originalHistoryLength: promptParts.conversationHistory.length,
          compressedPromptLength: prompt.length,
          tier: modelTier
        },
        '[EvaluatePrompt] Prompt prepared for streaming'
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
              modelInfo: llm.getModelInfo()
            },
            '[EvaluatePrompt] ✅ Streaming evaluation complete'
          );
          callbacks.onComplete({
            answer: fullAnswer,
            modelUsed: llm.getModelInfo().modelId,
            levelUsed: modelTier,
            latency
          });
        },
        onError: (error: Error) => {
          this.logger.error(
            { error, query: input.userQuery.substring(0, 80) },
            '[EvaluatePrompt] ❌ Streaming evaluation failed'
          );
          callbacks.onError(error);
        }
      });
    } catch (error) {
      this.logger.error(
        { error, query: input.userQuery.substring(0, 80) },
        '[EvaluatePrompt] ❌ Streaming evaluation setup failed'
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
        intent: (input.classification as any).intent
      },
      '[EvaluatePrompt] Starting evaluation'
    );

    try {
      // STEP 1: Decide which model tier to use based on classification level
      const modelTier = this.selectModelTier(input.classification);
      this.logger.debug(
        {
          confidence: input.classification.confidence,
          tier: modelTier,
          subject: input.classification.subject,
          intent: (input.classification as any).intent
        },
        '[EvaluatePrompt] Model tier selected'
      );

      // STEP 2: Get model config for the tier
      const config = modelConfigService.getModelConfig(
        {
          ...input.classification,
          level: modelTier
        },
        input.subscription || 'free'
      );

      const registryEntry = modelConfigService.getModelRegistryEntry(config.modelId);
      if (!registryEntry) {
        throw new Error(`Registry entry not found for model ${config.modelId}`);
      }

      // STEP 3: Create tier-specific LLM using temperature and maxTokens from config
      const llm = createTierLLM(modelTier, registryEntry, this.logger, config.temperature, config.maxTokens);

      this.logger.info(
        {
          modelInfo: llm.getModelInfo(),
          tier: modelTier,
          temperature: config.temperature,
          maxTokens: config.maxTokens
        },
        '[EvaluatePrompt] Tier-specific LLM created'
      );

      // STEP 4: Build evaluation prompt - now returns 3 parts
      const promptParts = buildEvaluationPrompt(
        input.userQuery,
        input.classification,
        input.topDocument,
        (input.classification as any).intent || 'factual_retrieval',
        input.userPrefs,
        input.conversationHistory,
        input.userName
      );

      // STEP 4.5: Apply selective compression (FIX: Main change)
      const prompt = await compressPromptSelectively(
        promptParts.systemPrompt,
        promptParts.conversationHistory,
        promptParts.referenceAndQuery,
        this.logger
      );

      this.logger.debug(
        {
          originalSystemLength: promptParts.systemPrompt.length,
          originalHistoryLength: promptParts.conversationHistory.length,
          compressedPromptLength: prompt.length,
          tier: modelTier
        },
        '[EvaluatePrompt] Prompt prepared'
      );

      // STEP 5: Invoke tier-specific LLM
      const answer = await llm.generate(prompt);

      const latency = Date.now() - startTime;

      this.logger.info(
        {
          modelTier,
          latency,
          answerLength: answer.length,
          modelInfo: llm.getModelInfo()
        },
        '[EvaluatePrompt] ✅ Evaluation complete'
      );

      return {
        answer,
        modelUsed: llm.getModelInfo().modelId,
        levelUsed: modelTier,
        latency
      };
    } catch (error) {
      this.logger.error(
        { error, query: input.userQuery.substring(0, 80) },
        '[EvaluatePrompt] ❌ Evaluation failed'
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
export function createEvaluatePrompt(modelSelector: ModelSelector, logger: pino.Logger): EvaluatePrompt {
  return new EvaluatePrompt(modelSelector, logger);
}