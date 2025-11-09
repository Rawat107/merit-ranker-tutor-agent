import { LLMOptions, StreamingCallbacks } from '../types/index.js';

/**
 * LangChain-compatible LLM interface abstraction used across adapters.
 */
export interface ILLM {
  generate(prompt: string, options?: LLMOptions): Promise<string>;
  stream(prompt: string, callbacks: StreamingCallbacks, options?: LLMOptions): Promise<void>;
  isAvailable(): Promise<boolean>;
  getModelInfo(): { modelId: string; provider: string; region: string };
}
