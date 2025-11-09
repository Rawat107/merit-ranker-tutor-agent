import { ILLM } from './LLM';
import { BedrockLLM } from './BedrockLLM';
import { modelConfigService } from '../config/modelConfig.js';
import { Classification, ModelRegistryEntry } from '../types/index.js';
import pino from 'pino';

/**
 * LangChain-compatible Model Selection Service
 * Selects appropriate models based on subject, level, and subscription
 */
export class ModelSelector {
  private logger: pino.Logger;
  private modelInstances: Map<string, ILLM> = new Map();

  constructor(logger: pino.Logger) {
    this.logger = logger;
  }

  /**
   * Get appropriate LLM instance based on classification and subscription
   */
  async getLLM(classification: Classification, userSubscription: string = 'free'): Promise<ILLM> {
    const modelConfig = modelConfigService.getModelConfig(classification, userSubscription);
    const registryEntry = modelConfigService.getModelRegistryEntry(modelConfig.modelId);

    if (!registryEntry) {
      this.logger.warn(`Model ${modelConfig.modelId} not found in registry, using default`);
      return this.getDefaultLLM();
    }

    // Check if we already have an instance for this model
    const cacheKey = `${modelConfig.modelId}-${registryEntry.region}`;
    if (this.modelInstances.has(cacheKey)) {
      this.logger.info(`Using cached LLM instance: ${cacheKey}`);
      return this.modelInstances.get(cacheKey)!;
    }

    // Create new instance based on model type
    const llmInstance = this.createLLMInstance(registryEntry, modelConfig);
    this.modelInstances.set(cacheKey, llmInstance);
    
    this.logger.info(`Created new LLM instance: ${cacheKey} for ${classification.subject}/${classification.level}`);
    return llmInstance;
  }

  /**
   * Get LLM for classifier (lightweight, fast model)
   */
  async getClassifierLLM(): Promise<ILLM> {
    const classifierConfig = modelConfigService.getClassifierConfig();
    const registryEntry = modelConfigService.getModelRegistryEntry(classifierConfig.modelId);
    
    if (!registryEntry) {
      return this.getDefaultLLM();
    }

    const cacheKey = `classifier-${classifierConfig.modelId}`;
    if (!this.modelInstances.has(cacheKey)) {
      const llmInstance = this.createLLMInstance(registryEntry, classifierConfig);
      this.modelInstances.set(cacheKey, llmInstance);
    }

    return this.modelInstances.get(cacheKey)!;
  }

  /**
   * Get LLM for reranker (specialized model for relevance scoring)
   */
  async getRerankerLLM(): Promise<ILLM> {
    const rerankerConfig = modelConfigService.getRerankerConfig();
    const registryEntry = modelConfigService.getModelRegistryEntry(rerankerConfig.modelId);
    
    if (!registryEntry) {
      return this.getDefaultLLM();
    }

    const cacheKey = `reranker-${rerankerConfig.modelId}`;
    if (!this.modelInstances.has(cacheKey)) {
      const llmInstance = this.createLLMInstance(registryEntry, rerankerConfig);
      this.modelInstances.set(cacheKey, llmInstance);
    }

    return this.modelInstances.get(cacheKey)!;
  }

  /**
   * Create LLM instance based on model provider and type
   */
  private createLLMInstance(registryEntry: ModelRegistryEntry, modelConfig: any): ILLM {
    // Determine provider based on model ID patterns
    const provider = this.determineProvider(registryEntry.bedrockId);
    
    // For now, all models go through Bedrock
    return new BedrockLLM(registryEntry, provider, this.logger);
    
    // TODO: Add support for direct API calls to OpenAI, Anthropic, etc.
    // if (provider === 'openai' && process.env.OPENAI_API_KEY) {
    //   return new OpenAILLM(registryEntry, this.logger);
    // }
  }

  /**
   * Determine model provider from model ID
   */
  private determineProvider(modelId: string): string {
    if (modelId.includes('anthropic') || modelId.includes('claude')) {
      return 'anthropic';
    } else if (modelId.includes('openai') || modelId.includes('gpt')) {
      return 'openai';
    } else if (modelId.includes('amazon') || modelId.includes('nova')) {
      return 'amazon';
    } else {
      return 'generic';
    }
  }

  /**
   * Get default fallback LLM
   */
  private getDefaultLLM(): ILLM {
    const defaultEntry: ModelRegistryEntry = {
      bedrockId: 'anthropic.claude-3-sonnet-20240229-v1:0',
      invokeId: 'anthropic.claude-3-sonnet-20240229-v1:0',
      inferenceProfileArn: null,
      region: 'ap-south-1'
    };
    
    return new BedrockLLM(defaultEntry, 'anthropic', this.logger);
  }

  /**
   * Clear model instances cache (useful for testing or configuration changes)
   */
  clearCache(): void {
    this.modelInstances.clear();
    this.logger.info('Model instance cache cleared');
  }

  /**
   * Get cache statistics
   */
  getCacheStats(): { total: number; models: string[] } {
    return {
      total: this.modelInstances.size,
      models: Array.from(this.modelInstances.keys())
    };
  }
}