import { ChatBedrockConverse } from '@langchain/aws';
import { ILLM } from './ILLM.js';
import { LLMOptions, ModelRegistryEntry } from '../types/index.js';
import pino from 'pino';

/**
 * BasicLLM - Fast, cost-efficient model for simple queries
 * Best for: Factual retrieval, simple definitions, general knowledge
 * Temperature and MaxTokens are configured per subject/level in modelConfig.ts
 */
export class BasicLLM implements ILLM {
  private llm: ChatBedrockConverse;
  private registryEntry: ModelRegistryEntry;
  private logger: pino.Logger;
  private temperature: number;
  private maxTokens: number;

  constructor(registryEntry: ModelRegistryEntry, logger: pino.Logger, temperature: number, maxTokens: number) {
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
      '[BasicLLM] Initialized'
    );
  }

  async generate(prompt: string, options: LLMOptions = {}): Promise<string> {
    try {
      this.logger.debug({ promptLength: prompt.length }, '[BasicLLM] Generating response');

      const response = await this.llm.invoke([
        {
          role: 'user',
          content: prompt,
        },
      ]);

      // Properly extract content - handle both string and array of content blocks
      let content: string;
      if (typeof response.content === 'string') {
        content = response.content;
      } else if (Array.isArray(response.content)) {
        // If it's an array of content blocks, extract text from each
        content = response.content
          .map(block => {
            if (typeof block === 'string') return block;
            if (block && typeof block === 'object' && 'text' in block) return block.text;
            return JSON.stringify(block);
          })
          .join('');
      } else {
        content = JSON.stringify(response.content);
      }

      this.logger.info(
        { responseLength: content.length, model: this.registryEntry.bedrockId },
        '[BasicLLM] Response generated'
      );

      return content;
    } catch (error) {
      this.logger.error({ error }, '[BasicLLM] Generation failed');
      throw error;
    }
  }

  async stream(
    prompt: string,
    callbacks: any,
    options: LLMOptions = {}
  ): Promise<void> {
    try {
      this.logger.debug({ promptLength: prompt.length }, '[BasicLLM] Starting stream');

      const stream = await this.llm.stream([
        {
          role: 'user',
          content: prompt,
        },
      ]);

      for await (const chunk of stream) {
        // Extract content from the chunk
        let content: string;
        if (typeof chunk.content === 'string') {
          content = chunk.content;
        } else if (Array.isArray(chunk.content)) {
          content = chunk.content
            .map(block => {
              if (typeof block === 'string') return block;
              if (block && typeof block === 'object' && 'text' in block) return block.text;
              return '';
            })
            .join('');
        } else {
          content = '';
        }

        if (content && callbacks.onToken) {
          callbacks.onToken(content);
        }
      }

      if (callbacks.onComplete) {
        callbacks.onComplete();
      }

      this.logger.info({ model: this.registryEntry.bedrockId }, '[BasicLLM] Stream completed');
    } catch (error) {
      this.logger.error({ error }, '[BasicLLM] Stream failed');
      if (callbacks.onError) {
        callbacks.onError(error);
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

/**
 * IntermediateLLM - Balanced model for step-by-step explanations
 * Best for: Math problems, step-by-step reasoning, structured explanations
 * Temperature and MaxTokens are configured per subject/level in modelConfig.ts
 */
export class IntermediateLLM implements ILLM {
  private llm: ChatBedrockConverse;
  private registryEntry: ModelRegistryEntry;
  private logger: pino.Logger;
  private temperature: number;
  private maxTokens: number;

  constructor(registryEntry: ModelRegistryEntry, logger: pino.Logger, temperature: number, maxTokens: number) {
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
      '[IntermediateLLM] Initialized'
    );
  }

  async generate(prompt: string, options: LLMOptions = {}): Promise<string> {
    try {
      this.logger.debug({ promptLength: prompt.length }, '[IntermediateLLM] Generating response');

      const response = await this.llm.invoke([
        {
          role: 'user',
          content: prompt,
        },
      ]);

      // Properly extract content - handle both string and array of content blocks
      let content: string;
      if (typeof response.content === 'string') {
        content = response.content;
      } else if (Array.isArray(response.content)) {
        // If it's an array of content blocks, extract text from each
        content = response.content
          .map(block => {
            if (typeof block === 'string') return block;
            if (block && typeof block === 'object' && 'text' in block) return block.text;
            return JSON.stringify(block);
          })
          .join('');
      } else {
        content = JSON.stringify(response.content);
      }

      this.logger.info(
        { responseLength: content.length, model: this.registryEntry.bedrockId },
        '[IntermediateLLM] Response generated'
      );

      return content;
    } catch (error) {
      this.logger.error({ error }, '[IntermediateLLM] Generation failed');
      throw error;
    }
  }

  async stream(
    prompt: string,
    callbacks: any,
    options: LLMOptions = {}
  ): Promise<void> {
    try {
      this.logger.debug({ promptLength: prompt.length }, '[IntermediateLLM] Starting stream');

      const stream = await this.llm.stream([
        {
          role: 'user',
          content: prompt,
        },
      ]);

      for await (const chunk of stream) {
        // Extract content from the chunk
        let content: string;
        if (typeof chunk.content === 'string') {
          content = chunk.content;
        } else if (Array.isArray(chunk.content)) {
          content = chunk.content
            .map(block => {
              if (typeof block === 'string') return block;
              if (block && typeof block === 'object' && 'text' in block) return block.text;
              return '';
            })
            .join('');
        } else {
          content = '';
        }

        if (content && callbacks.onToken) {
          callbacks.onToken(content);
        }
      }

      if (callbacks.onComplete) {
        callbacks.onComplete();
      }

      this.logger.info({ model: this.registryEntry.bedrockId }, '[IntermediateLLM] Stream completed');
    } catch (error) {
      this.logger.error({ error }, '[IntermediateLLM] Stream failed');
      if (callbacks.onError) {
        callbacks.onError(error);
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

/**
 * AdvancedLLM - Most capable model for complex reasoning and verification
 * Best for: Complex reasoning, proofs, rigorous analysis, uncertain queries
 * Temperature and MaxTokens are configured per subject/level in modelConfig.ts
 */
export class AdvancedLLM implements ILLM {
  private llm: ChatBedrockConverse;
  private registryEntry: ModelRegistryEntry;
  private logger: pino.Logger;
  private temperature: number;
  private maxTokens: number;

  constructor(registryEntry: ModelRegistryEntry, logger: pino.Logger, temperature: number, maxTokens: number) {
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
      '[AdvancedLLM] Initialized'
    );
  }

  async generate(prompt: string, options: LLMOptions = {}): Promise<string> {
    try {
      this.logger.debug({ promptLength: prompt.length }, '[AdvancedLLM] Generating response');

      const response = await this.llm.invoke([
        {
          role: 'user',
          content: prompt,
        },
      ]);

      // Properly extract content - handle both string and array of content blocks
      let content: string;
      if (typeof response.content === 'string') {
        content = response.content;
      } else if (Array.isArray(response.content)) {
        // If it's an array of content blocks, extract text from each
        content = response.content
          .map(block => {
            if (typeof block === 'string') return block;
            if (block && typeof block === 'object' && 'text' in block) return block.text;
            return JSON.stringify(block);
          })
          .join('');
      } else {
        content = JSON.stringify(response.content);
      }

      this.logger.info(
        { responseLength: content.length, model: this.registryEntry.bedrockId },
        '[AdvancedLLM] Response generated'
      );

      return content;
    } catch (error) {
      this.logger.error({ error }, '[AdvancedLLM] Generation failed');
      throw error;
    }
  }

  async stream(
    prompt: string,
    callbacks: any,
    options: LLMOptions = {}
  ): Promise<void> {
    try {
      this.logger.debug({ promptLength: prompt.length }, '[AdvancedLLM] Starting stream');

      const stream = await this.llm.stream([
        {
          role: 'user',
          content: prompt,
        },
      ]);

      for await (const chunk of stream) {
        // Extract content from the chunk
        let content: string;
        if (typeof chunk.content === 'string') {
          content = chunk.content;
        } else if (Array.isArray(chunk.content)) {
          content = chunk.content
            .map(block => {
              if (typeof block === 'string') return block;
              if (block && typeof block === 'object' && 'text' in block) return block.text;
              return '';
            })
            .join('');
        } else {
          content = '';
        }

        if (content && callbacks.onToken) {
          callbacks.onToken(content);
        }
      }

      if (callbacks.onComplete) {
        callbacks.onComplete();
      }

      this.logger.info({ model: this.registryEntry.bedrockId }, '[AdvancedLLM] Stream completed');
    } catch (error) {
      this.logger.error({ error }, '[AdvancedLLM] Stream failed');
      if (callbacks.onError) {
        callbacks.onError(error);
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