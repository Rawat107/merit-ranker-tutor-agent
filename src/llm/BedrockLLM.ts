import { ILLM } from './ILLM.js';
import { LLMOptions, StreamingCallbacks, ModelRegistryEntry } from '../types/index.js';
import pino from 'pino';

/**
 * AWS Bedrock LLM implementation (mocked). Replace TODO blocks with real BedrockRuntimeClient.
 */
export class BedrockLLM implements ILLM {
  private registryEntry: ModelRegistryEntry;
  private provider: string;
  private logger: pino.Logger;

  constructor(registryEntry: ModelRegistryEntry, provider: string, logger: pino.Logger) {
    this.registryEntry = registryEntry;
    this.provider = provider;
    this.logger = logger;
  }

  async generate(prompt: string, options: LLMOptions = {}): Promise<string> {
    this.logger.info({ model: this.registryEntry.bedrockId }, 'LLM.generate invoked');
    // TODO: Implement Bedrock invoke logic
    await new Promise(r => setTimeout(r, 50));
    return `Mock(${this.provider}) => ${prompt.slice(0, 60)}...`;
  }

  async stream(prompt: string, callbacks: StreamingCallbacks, options: LLMOptions = {}): Promise<void> {
    this.logger.info({ model: this.registryEntry.bedrockId }, 'LLM.stream invoked');
    // TODO: Implement Bedrock streaming invoke
    const tokens = (`Mock streaming response for: ${prompt.slice(0, 40)}...`).split(' ');
    try {
      for (const t of tokens) {
        await new Promise(r => setTimeout(r, 30));
        callbacks.onToken(t + ' ');
      }
      callbacks.onComplete();
    } catch (err) {
      callbacks.onError(err as Error);
    }
  }

  async isAvailable(): Promise<boolean> {
    // TODO: quick health check (e.g., lightweight invoke)
    return true;
  }

  getModelInfo() { return { modelId: this.registryEntry.bedrockId, provider: this.provider, region: this.registryEntry.region }; }
}
