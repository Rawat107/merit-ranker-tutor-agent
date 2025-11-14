import { ChatBedrockConverse } from '@langchain/aws';
import { ILLM } from './ILLM.js';
import { LLMOptions, ModelRegistryEntry, StreamingCallbacks } from '../types/index.js';
import { extractContent, extractStreamChunkContent } from './llmHelpers.js';
import pino from 'pino';

/**
 * Base LLM class with shared implementation
 * Reduces code duplication across tier-specific LLM classes
 */
export abstract class BaseTierLLM implements ILLM {
  protected llm: ChatBedrockConverse;
  protected registryEntry: ModelRegistryEntry;
  protected logger: pino.Logger;
  protected temperature: number;
  protected maxTokens: number;
  protected tierName: string;

  constructor(
    tierName: string,
    registryEntry: ModelRegistryEntry,
    logger: pino.Logger,
    temperature: number,
    maxTokens: number
  ) {
    this.tierName = tierName;
    this.registryEntry = registryEntry;
    this.logger = logger;
    this.temperature = temperature;
    this.maxTokens = maxTokens;

    // Use inference profile ARN if available, otherwise use bedrockId
    const modelId = registryEntry.inferenceProfileArn || registryEntry.bedrockId;

    this.llm = new ChatBedrockConverse({
      model: modelId,
      region: registryEntry.region,
      temperature: this.temperature,
      maxTokens: this.maxTokens,
      credentials: {
        accessKeyId: process.env.AWS_ACCESS_KEY_ID || '',
        secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY || '',
      },
    });

    this.logger.info(
      { model: modelId, temperature: this.temperature, maxTokens: this.maxTokens },
      `[${this.tierName}] Initialized`
    );
  }

  async generate(prompt: string, options: LLMOptions = {}): Promise<string> {
    try {
      this.logger.debug({ promptLength: prompt.length }, `[${this.tierName}] Generating response`);

      const response = await this.llm.invoke([
        {
          role: 'user',
          content: prompt,
        },
      ]);

      const content = extractContent(response.content);

      this.logger.info(
        { responseLength: content.length, model: this.registryEntry.bedrockId },
        `[${this.tierName}] Response generated`
      );

      return content;
    } catch (error) {
      this.logger.error({ error }, `[${this.tierName}] Generation failed`);
      throw error;
    }
  }

  async stream(
    prompt: string,
    callbacks: StreamingCallbacks,
    options: LLMOptions = {}
  ): Promise<void> {
    try {
      this.logger.debug({ promptLength: prompt.length }, `[${this.tierName}] Starting stream`);

      const stream = await this.llm.stream([
        {
          role: 'user',
          content: prompt,
        },
      ]);

      for await (const chunk of stream) {
        const content = extractStreamChunkContent(chunk.content);

        if (content && callbacks.onToken) {
          callbacks.onToken(content);
        }
      }

      if (callbacks.onComplete) {
        callbacks.onComplete();
      }

      this.logger.info({ model: this.registryEntry.bedrockId }, `[${this.tierName}] Stream completed`);
    } catch (error) {
      this.logger.error({ error }, `[${this.tierName}] Stream failed`);
      if (callbacks.onError) {
        callbacks.onError(error as Error);
      }
      throw error;
    }
  }

  async isAvailable(): Promise<boolean> {
    return true;
  }

  getModelInfo() {
    return {
      modelId: this.registryEntry.bedrockId,
      provider: 'bedrock',
      region: this.registryEntry.region,
    };
  }
}
