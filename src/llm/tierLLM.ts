import { ILLM } from './ILLM.js';
import { ModelRegistryEntry } from '../types/index.js';
import { BaseTierLLM } from './BaseTierLLM.js';
import pino from 'pino';

/**
 * BasicLLM - Fast, cost-efficient model for simple queries
 * Best for: Factual retrieval, simple definitions, general knowledge
 * Temperature and MaxTokens are configured per subject/level in modelConfig.ts
 */
export class BasicLLM extends BaseTierLLM {
  constructor(registryEntry: ModelRegistryEntry, logger: pino.Logger, temperature: number, maxTokens: number) {
    super('BasicLLM', registryEntry, logger, temperature, maxTokens);
  }
}

/**
 * IntermediateLLM - Balanced model for step-by-step explanations
 * Best for: Math problems, step-by-step reasoning, structured explanations
 * Temperature and MaxTokens are configured per subject/level in modelConfig.ts
 */
export class IntermediateLLM extends BaseTierLLM {
  constructor(registryEntry: ModelRegistryEntry, logger: pino.Logger, temperature: number, maxTokens: number) {
    super('IntermediateLLM', registryEntry, logger, temperature, maxTokens);
  }
}

/**
 * AdvancedLLM - Most capable model for complex reasoning and verification
 * Best for: Complex reasoning, proofs, rigorous analysis, uncertain queries
 * Temperature and MaxTokens are configured per subject/level in modelConfig.ts
 */
export class AdvancedLLM extends BaseTierLLM {
  constructor(registryEntry: ModelRegistryEntry, logger: pino.Logger, temperature: number, maxTokens: number) {
    super('AdvancedLLM', registryEntry, logger, temperature, maxTokens);
  }
}

/**
 * Factory function to create appropriate model with required temperature and maxTokens from modelConfig
 */
export function createTierLLM(
  tier: 'basic' | 'intermediate' | 'advanced',
  registryEntry: ModelRegistryEntry,
  logger: pino.Logger,
  temperature: number,
  maxTokens: number
): ILLM {
  switch (tier) {
    case 'basic':
      return new BasicLLM(registryEntry, logger, temperature, maxTokens);
    case 'intermediate':
      return new IntermediateLLM(registryEntry, logger, temperature, maxTokens);
    case 'advanced':
      return new AdvancedLLM(registryEntry, logger, temperature, maxTokens);
    default:
      return new IntermediateLLM(registryEntry, logger, temperature, maxTokens);
  }
}