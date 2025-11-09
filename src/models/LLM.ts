import { LLMOptions, StreamingCallbacks } from '../types/index.js';

export interface ILLM {
  /**
   * Generate text completion for a prompt
   */
  generate(prompt: string, options?: LLMOptions): Promise<string>;

  /**
   * Stream text completion with callbacks
   */
  stream(prompt: string, callbacks: StreamingCallbacks, options?: LLMOptions): Promise<void>;

  /**
   * Check if the model is available
   */
  isAvailable(): Promise<boolean>;
}