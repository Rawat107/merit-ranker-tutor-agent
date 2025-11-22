import pino from 'pino';
import { Classification, ModelRegistryEntry } from '../types/index.js';
import { modelConfigService } from '../config/modelConfig.js';
import { ILLM } from './ILLM.js';
import { createTierLLM } from './tierLLM.js';


/**
 * ModelSelector chooses the best model/provider given subject, level, subscription and query intent.
 * Uses SWITCH CASE for fast model selection (cost-efficient & DRY)
 */
export class ModelSelector {
  private logger: pino.Logger;
  private cache = new Map<string, ILLM>();

  constructor(logger: pino.Logger) {
    this.logger = logger;
  }

  /**
   * Main getLLM method - selects model based on classification and subscription
   * Gets temperature, maxTokens, modelId from modelConfig
   */
  async getLLM(
    classification: Classification,
    subscription = 'free',
    query = ''
  ): Promise<ILLM> {
    try {
      // Get config with temperature, maxTokens, modelId from modelConfig
      const cfg = modelConfigService.getModelConfig(classification, subscription);
      const entry = modelConfigService.getModelRegistryEntry(cfg.modelId);

      if (!entry) {
        this.logger.warn(
          { modelId: cfg.modelId },
          '[ModelSelector] Registry entry missing; using fallback'
        );
        return this.getFallback(classification, subscription);
      }

      const cacheKey = `${entry.bedrockId}:${classification.subject}:${classification.level}`;

      // Check cache - reuse if already created
      if (this.cache.has(cacheKey)) {
        return this.cache.get(cacheKey)!;
      }

      // Create tier LLM with config values (temperature, maxTokens from modelConfig)
      const tier = classification.level as 'basic' | 'intermediate' | 'advanced';
      const llm = createTierLLM(tier, entry, this.logger, cfg.temperature, cfg.maxTokens);
      this.cache.set(cacheKey, llm);

      this.logger.info(
        { 
          model: entry.bedrockId, 
          tier, 
          temperature: cfg.temperature, 
          maxTokens: cfg.maxTokens 
        },
        '[ModelSelector] Created LLM from modelConfig'
      );

      return llm;
    } catch (error) {
      this.logger.error({ error }, '[ModelSelector] Error in getLLM');
      return this.getFallback(classification, subscription);
    }
  }

  /**
   * Get pre-configured Classifier LLM
   * Uses classifier config from modelConfig (nova-micro with temp=0.0, tokens=200)
   */
  async getClassifierLLM(): Promise<ILLM> {
    try {
      const cfg = modelConfigService.getClassifierConfig();
      const entry = modelConfigService.getModelRegistryEntry(cfg.modelId);

      if (!entry) {
        return this.getFallback({ subject: 'general', level: 'basic', confidence: 1.0 }, 'free');
      }

      const cacheKey = `classifier:${entry.bedrockId}`;

      if (this.cache.has(cacheKey)) {
        return this.cache.get(cacheKey)!;
      }

      // Classifier uses basic tier with config values
      const llm = createTierLLM('basic', entry, this.logger, cfg.temperature, cfg.maxTokens);
      this.cache.set(cacheKey, llm);

      return llm;
    } catch (error) {
      this.logger.error({ error }, '[ModelSelector] Error in getClassifierLLM');
      return this.getFallback({ subject: 'general', level: 'basic', confidence: 1.0 }, 'free');
    }
  }

  /**
   * Get pre-configured Reranker LLM
   * Uses reranker config from modelConfig (cohere rerank with temp=0.0, tokens=200)
   */
  async getRerankerLLM(): Promise<ILLM> {
    try {
      const cfg = modelConfigService.getRerankerConfig();
      const entry = modelConfigService.getModelRegistryEntry(cfg.modelId);

      if (!entry) {
        return this.getFallback({ subject: 'general', level: 'basic', confidence: 1.0 }, 'free');
      }

      const cacheKey = `reranker:${entry.bedrockId}`;

      if (this.cache.has(cacheKey)) {
        return this.cache.get(cacheKey)!;
      }

      // Reranker uses basic tier with config values
      const llm = createTierLLM('basic', entry, this.logger, cfg.temperature, cfg.maxTokens);
      this.cache.set(cacheKey, llm);

      return llm;
    } catch (error) {
      this.logger.error({ error }, '[ModelSelector] Error in getRerankerLLM');
      return this.getFallback({ subject: 'general', level: 'basic', confidence: 1.0 }, 'free');
    }
  }

  /**
   * Get fallback LLM (always available)
   * Uses general/basic config from modelConfig with all registry details
   */
  private getFallback(classification: Classification, subscription: string): ILLM {
    try {
      // Use general/basic as fallback - all config comes from modelConfig
      const fallbackClassification: Classification = {
        subject: classification?.subject || 'general',
        level: classification?.level || 'basic',
        confidence: classification?.confidence || 1.0,
      };

      const cfg = modelConfigService.getModelConfig(fallbackClassification, subscription);
      const entry = modelConfigService.getModelRegistryEntry(cfg.modelId);

      if (!entry) {
        // Last resort hardcoded fallback if modelConfig fails
        this.logger.error('[ModelSelector] ModelConfig failed, using hardcoded fallback');
        const hardcodedEntry: ModelRegistryEntry = {
          bedrockId: 'anthropic.claude-3-sonnet-20240229-v1:0',
          invokeId: 'anthropic.claude-3-sonnet-20240229-v1:0',
          inferenceProfileArn: 'arn:aws:bedrock:ap-south-1:558069890997:inference-profile/apac.anthropic.claude-3-sonnet-20240229-v1:0',
          region: 'ap-south-1',
        };
        return createTierLLM('basic', hardcodedEntry, this.logger, 0.2, 800);
      }

      // Return fallback with config values (temperature, maxTokens, modelId, region, ARN all from modelConfig)
      return createTierLLM(
        fallbackClassification.level as 'basic' | 'intermediate' | 'advanced',
        entry,
        this.logger,
        cfg.temperature,
        cfg.maxTokens
      );
    } catch (error) {
      this.logger.error({ error }, '[ModelSelector] Fallback creation failed');
      // Absolute last resort
      const emergencyEntry: ModelRegistryEntry = {
        bedrockId: 'anthropic.claude-3-sonnet-20240229-v1:0',
        invokeId: 'anthropic.claude-3-sonnet-20240229-v1:0',
        inferenceProfileArn: 'arn:aws:bedrock:ap-south-1:558069890997:inference-profile/apac.anthropic.claude-3-sonnet-20240229-v1:0',
        region: 'ap-south-1',
      };
      return createTierLLM('basic', emergencyEntry, this.logger, 0.2, 800);
    }
  }

  async getFastLLM(classification: Classification, subscription = 'free'): Promise<ILLM> {
  // For high-confidence basic queries, use nova-micro (cheaper & faster)
    if (classification.confidence > 0.9 && classification.level === 'basic') {
      const fastEntry = modelConfigService.getModelRegistryEntry('amazon.nova-micro-v1:0');
      if (fastEntry) {
        this.logger.info({ model: 'amazon.nova-micro-v1:0' }, 'Using fast model for basic query');
        return createTierLLM('basic', fastEntry, this.logger, 0.2, 800);
      }
    }
    
    // For intermediate with high confidence, use haiku instead of sonnet
    if (classification.confidence > 0.85 && classification.level === 'intermediate') {
      const fastEntry = modelConfigService.getModelRegistryEntry('anthropic.claude-3-haiku-20240307-v1:0');
      if (fastEntry) {
        this.logger.info({ model: 'anthropic.claude-3-haiku' }, 'Using fast model for intermediate query');
        return createTierLLM('intermediate', fastEntry, this.logger, 0.3, 1200);
      }
    }
    
    // Default to regular selection
    return this.getLLM(classification, subscription);
}
  
  /**
   * Cache management
   */
  clearCache(): void {
    this.cache.clear();
    this.logger.info('[ModelSelector] Cache cleared');
  }

  getCacheStats(): { size: number; keys: string[] } {
    return {
      size: this.cache.size,
      keys: [...this.cache.keys()],
    };
  }
}