import pino from 'pino';
import { ILLM } from './ILLM.js';
import { ModelRegistryEntry } from '../types/index.js';

/**
 * Generic provider adapter interface for future direct integrations (OpenAI SDK, Anthropic SDK, etc.)
 */
export interface LLMProviderAdapter {
  name: string;
  create(entry: ModelRegistryEntry, logger: pino.Logger): ILLM;
  supports(modelId: string): boolean;
}

// Placeholder adapters map; currently all routed through Bedrock wrapper.
export const providerAdapters: LLMProviderAdapter[] = [];
