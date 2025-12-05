import { ModelSelector } from '../llm/ModelSelector.js';
import { createTierLLM } from '../llm/tierLLM.js';
import { Classification, Document } from '../types/index.js';
import {
  buildEvaluationPrompt,
  buildMCQGenerationPrompt,
  buildNoteGenerationPrompt,
  buildMockTestPrompt,
  buildQuizEvaluationPrompt,
  buildHindiResponseWrapper
} from '../utils/promptTemplates.js';
import { modelConfigService } from '../config/modelConfig.js';
import { linguaCompressor } from '../compression/lingua_compressor.js';
import pino from 'pino';

function getStatusMessage(tier: 'basic' | 'intermediate' | 'advanced', subject: string): string {
  const thinkingMessages = {
    basic: 'Thinking...',
    intermediate: 'Thinking harder...',
    advanced: 'Thinking deeply...'
  };

  const subjectMessages: Record<string, string> = {
    math: 'Solving the problem...',
    reasoning: 'Working through the logic...',
    science: 'Analyzing the concepts...',
    history: 'Researching historical context...',
    english_grammar: 'Checking grammar rules...',
    general_knowledge: 'Finding information...',
    current_affairs: 'Searching latest updates...',
    mcq_generation: 'Creating questions...',
    note_generation: 'Generating notes...',
    mock_test_generation: 'Preparing test...',
    quiz_evaluation: 'Evaluating answer...'
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
 * Reference compression rate: 0.5 (as requested - half the tokens)
 */
async function compressContextInParallel(
  conversationHistory: string,
  reference: string,
  logger: pino.Logger
): Promise<{ compressedHistory: string; compressedReference: string; savedTokens: number }> {
  const startTime = Date.now();

  try {
    logger.debug(
      {
        historyLength: conversationHistory.length,
        refLength: reference.length
      },
      '[Compression] Starting PARALLEL context compression'
    );

    //  Compress both in parallel - aggressive rates
    const [compressedHistory, compressedReference] = await Promise.all([
      // History: 25% compression rate (keep most content but compress aggressively)
      conversationHistory.trim().length > 100
        ? linguaCompressor.compress(conversationHistory, { rate: 0.25 })
        : conversationHistory,
      // Reference: 50% compression rate (keep important info)
      reference.trim().length > 100
        ? linguaCompressor.compress(reference, { rate: 0.5 })
        : reference
    ]);

    const originalContextLength = conversationHistory.length + reference.length;
    const compressedContextLength = compressedHistory.length + compressedReference.length;
    const savedChars = originalContextLength - compressedContextLength;
    const savedTokens = Math.floor(savedChars / 4);
    const compressionRatio = (
      ((originalContextLength - compressedContextLength) / originalContextLength) * 100
    ).toFixed(1);

    const totalTime = Date.now() - startTime;

    logger.info(
      {
        originalHistoryLength: conversationHistory.length,
        compressedHistoryLength: compressedHistory.length,
        originalRefLength: reference.length,
        compressedRefLength: compressedReference.length,
        totalOriginal: originalContextLength,
        totalCompressed: compressedContextLength,
        savedChars,
        savedTokens,
        compressionRatio: `${compressionRatio}%`,
        processingTime: `${totalTime}ms`
      },
      '[Compression]  PARALLEL COMPRESSION DONE (history: 25%, reference: 50%)'
    );

    return { compressedHistory, compressedReference, savedTokens };
  } catch (error) {
    logger.error({ error }, '[Compression]  Parallel compression failed, using original');
    return {
      compressedHistory: conversationHistory,
      compressedReference: reference,
      savedTokens: 0
    };
  }
}
/**
 */
async function compressSystemPromptIfNeeded(
  systemPrompt: string,
  compressedHistory: string,
  compressedReference: string,
  currentQuery: string,
  logger: pino.Logger
): Promise<string> {
  // ✅ Calculate tokens including current query
  const combinedContextLength =
    compressedHistory.length + compressedReference.length + currentQuery.length;
  const estimatedContextTokens = Math.floor(combinedContextLength / 4);

  logger.debug(
    {
      systemPromptLength: systemPrompt.length,
      compressedContextTokens: estimatedContextTokens,
      historyTokens: Math.floor(compressedHistory.length / 4),
      refTokens: Math.floor(compressedReference.length / 4),
      queryTokens: Math.floor(currentQuery.length / 4),
      threshold: 1000
    },
    '[Compression] Checking if system prompt needs compression'
  );

  // ✅ Threshold is 1000 tokens (was 1200)
  if (estimatedContextTokens > 1000) {
    logger.info(
      { contextTokens: estimatedContextTokens, threshold: 1000 },
      '[Compression] Context > 1000 tokens, compressing system prompt (LIGHT 70%)'
    );

    const startTime = Date.now();
    const compressedSystem = await linguaCompressor.compress(systemPrompt, { rate: 0.7 });
    const totalTime = Date.now() - startTime;

    logger.info(
      {
        original: systemPrompt.length,
        compressed: compressedSystem.length,
        saved: systemPrompt.length - compressedSystem.length,
        ratio: ((1 - compressedSystem.length / systemPrompt.length) * 100).toFixed(1) + '%',
        processingTime: `${totalTime}ms`
      },
      '[Compression] ✅ System prompt compressed (LIGHT 70%)'
    );

    return compressedSystem;
  } else {
    logger.info(
      { contextTokens: estimatedContextTokens, threshold: 1000 },
      '[Compression] Context <= 1000 tokens, using ORIGINAL system prompt'
    );
    return systemPrompt;
  }
}

/**
 * Parse quiz evaluation input
 */
function parseQuizEvalInput(input: string): {
  question: string;
  studentAnswer: string;
  correctAnswer: string;
} {
  const questionMatch = input.match(/Question:\s*([^|]+)/i);
  const studentMatch = input.match(/Student Answer:\s*([^|]+)/i);
  const correctMatch = input.match(/Correct Answer:\s*(.+?)$/i);

  return {
    question: questionMatch ? questionMatch[1].trim() : input.substring(0, 100),
    studentAnswer: studentMatch ? studentMatch[1].trim() : 'Not provided',
    correctAnswer: correctMatch ? correctMatch[1].trim() : 'Not provided'
  };
}

/**
 * Main EvaluatePrompt class
 */
export class EvaluatePrompt {
  private modelSelector: ModelSelector;
  private logger: pino.Logger;

  constructor(modelSelector: ModelSelector, logger: pino.Logger) {
    this.modelSelector = modelSelector;
    this.logger = logger;
  }

  /**
   * FIXED: Build prompt with COMPRESSION-RESISTANT structure
   * Uses explicit markers that survive aggressive compression
    */
  private async buildPromptByIntent(
    intent: string,
    input: EvaluatePromptInput,
    classification: Classification
  ): Promise<string> {
    const responseLanguage = (classification as any).responseLanguage || 'en';

    switch (intent) {
      case 'mcq_generation': {
        const difficulty = classification.level as 'basic' | 'intermediate' | 'advanced';
        return buildMCQGenerationPrompt(input.userQuery, 5, difficulty, classification.subject);
      }

      case 'note_generation': {
        const sourceText = input.topDocument?.text || '';
        return buildNoteGenerationPrompt(input.userQuery, sourceText, true);
      }

      case 'mock_test_generation': {
        const topics = [input.userQuery];
        return buildMockTestPrompt(classification.subject, topics, 10, 60);
      }

      case 'quiz_evaluation': {
        const { question, studentAnswer, correctAnswer } = parseQuizEvalInput(input.userQuery);
        return buildQuizEvaluationPrompt(question, studentAnswer, correctAnswer, classification.subject);
      }

      default: {
        const promptParts = buildEvaluationPrompt(
          input.userQuery,
          classification,
          input.topDocument,
          intent,
          input.userPrefs,
          input.conversationHistory,
          input.userName
        );

        const systemPrompt = responseLanguage === 'hi'
          ? buildHindiResponseWrapper(promptParts.systemPrompt)
          : promptParts.systemPrompt;

        //  Step 1: Compress history + reference IN PARALLEL
        const { compressedHistory, compressedReference, savedTokens } = await compressContextInParallel(
          promptParts.conversationHistory,
          promptParts.reference,
          this.logger
        );

        //  Step 2: Check if system needs compression (include query size)
        const finalSystemPrompt = await compressSystemPromptIfNeeded(
          systemPrompt,
          compressedHistory,
          compressedReference,
          promptParts.currentQuery,
          this.logger
        );

        // Step 3: Final combination
        const finalPrompt = `${finalSystemPrompt}\n\n${compressedHistory}\n\n${compressedReference}\n\n${promptParts.currentQuery}`;

        this.logger.info(
          {
            originalLength:
              systemPrompt.length +
              promptParts.conversationHistory.length +
              promptParts.reference.length +
              promptParts.currentQuery.length,
            compressedLength: finalPrompt.length,
            savedTokens,
            estimatedTokens: Math.floor(finalPrompt.length / 4),
            processingStrategy: 'PARALLEL(history:25%+ref:50%) → CONDITIONAL(system @ 1000 tokens) → QUERY uncompressed'
          },
          '[Compression]  PIPELINE COMPLETE'
        );

        return finalPrompt;
      }
    }
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
    const intent = (input.classification as any).intent || 'factual_retrieval';
    const responseLanguage = (input.classification as any).responseLanguage || 'en';

    this.logger.info(
      {
        query: input.userQuery.substring(0, 80),
        subject: input.classification.subject,
        confidence: input.classification.confidence,
        intent
      },
      '[EvaluatePrompt] Starting streaming evaluation'
    );

    try {
      const modelTier = this.selectModelTier(input.classification);
      const statusMessage = getStatusMessage(modelTier, intent);

      this.logger.debug(
        {
          confidence: input.classification.confidence,
          tier: modelTier,
          subject: input.classification.subject,
          intent
        },
        '[EvaluatePrompt] Model tier selected for streaming'
      );

      callbacks.onMetadata({
        modelTier,
        subject: input.classification.subject,
        confidence: input.classification.confidence,
        status: statusMessage,
        intent,
        language: responseLanguage
      });

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

      const llm = createTierLLM(modelTier, registryEntry, this.logger, config.temperature, config.maxTokens);

      this.logger.info(
        {
          modelInfo: llm.getModelInfo(),
          tier: modelTier,
          temperature: config.temperature,
          maxTokens: config.maxTokens,
          intent
        },
        '[EvaluatePrompt] Tier-specific LLM created for streaming'
      );

      const prompt = await this.buildPromptByIntent(intent, input, input.classification);

      this.logger.debug(
        {
          promptLength: prompt.length,
          tier: modelTier,
          intent
        },
        '[EvaluatePrompt] Prompt prepared for streaming'
      );

      let fullAnswer = '';

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
              intent
            },
            '[EvaluatePrompt]  Streaming evaluation complete'
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
            { error, query: input.userQuery.substring(0, 80), intent },
            '[EvaluatePrompt]  Streaming evaluation failed'
          );
          callbacks.onError(error);
        }
      });
    } catch (error) {
      this.logger.error(
        { error, query: input.userQuery.substring(0, 80), intent },
        '[EvaluatePrompt]  Streaming evaluation setup failed'
      );
      callbacks.onError(error as Error);
    }
  }

  /**
   * Main evaluation method
   */
  async evaluate(input: EvaluatePromptInput): Promise<EvaluatePromptOutput> {
    const startTime = Date.now();
    const intent = (input.classification as any).intent || 'factual_retrieval';

    this.logger.info(
      {
        query: input.userQuery.substring(0, 80),
        subject: input.classification.subject,
        confidence: input.classification.confidence,
        intent
      },
      '[EvaluatePrompt] Starting evaluation'
    );

    try {
      const modelTier = this.selectModelTier(input.classification);

      this.logger.debug(
        {
          confidence: input.classification.confidence,
          tier: modelTier,
          subject: input.classification.subject,
          intent
        },
        '[EvaluatePrompt] Model tier selected'
      );

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

      const llm = createTierLLM(modelTier, registryEntry, this.logger, config.temperature, config.maxTokens);

      this.logger.info(
        {
          modelInfo: llm.getModelInfo(),
          tier: modelTier,
          temperature: config.temperature,
          maxTokens: config.maxTokens,
          intent
        },
        '[EvaluatePrompt] Tier-specific LLM created'
      );

      const prompt = await this.buildPromptByIntent(intent, input, input.classification);

      this.logger.debug(
        {
          promptLength: prompt.length,
          tier: modelTier,
          intent
        },
        '[EvaluatePrompt] Prompt prepared'
      );

      const answer = await llm.generate(prompt);

      const latency = Date.now() - startTime;

      this.logger.info(
        {
          modelTier,
          latency,
          answerLength: answer.length,
          modelInfo: llm.getModelInfo(),
          intent
        },
        '[EvaluatePrompt]  Evaluation complete'
      );

      return {
        answer,
        modelUsed: llm.getModelInfo().modelId,
        levelUsed: modelTier,
        latency
      };
    } catch (error) {
      this.logger.error(
        { error, query: input.userQuery.substring(0, 80), intent },
        '[EvaluatePrompt]  Evaluation failed'
      );
      throw error;
    }
  }

  /**
   * Select model tier based on classification level
   */
  private selectModelTier(classification: Classification): 'basic' | 'intermediate' | 'advanced' {
    const level = classification.level;

    this.logger.debug({ level }, '[EvaluatePrompt] Selecting model tier based on classification level');

    switch (level) {
      case 'basic':
        this.logger.info('[EvaluatePrompt] Using BASIC model for basic-level question');
        return 'basic';

      case 'intermediate':
        this.logger.info('[EvaluatePrompt] Using INTERMEDIATE model for intermediate-level question');
        return 'intermediate';

      case 'advanced':
        this.logger.info('[EvaluatePrompt] Using ADVANCED model for advanced-level question');
        return 'advanced';

      default:
        this.logger.warn({ level }, '[EvaluatePrompt] Unknown level, defaulting to intermediate');
        return 'intermediate';
    }
  }
}

/**
 * Factory function
 */
export function createEvaluatePrompt(
  modelSelector: ModelSelector,
  logger: pino.Logger
): EvaluatePrompt {
  return new EvaluatePrompt(modelSelector, logger);
}
