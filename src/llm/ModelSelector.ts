import pino from 'pino';
import { Classification, ModelRegistryEntry } from '../types/index.js';
import { modelConfigService } from '../config/modelConfig.js';
import { ILLM } from './ILLM.js';
import { BedrockLLM } from './BedrockLLM.js';
import { providerAdapters } from './ProviderRegistry.js';

interface ProviderDecisionContext {
  classification: Classification;
  subscription: string;
  query: string;
}

/**
 * ModelSelector chooses the best model/provider given subject, level, subscription and query intent.
 */
export class ModelSelector {
  private logger: pino.Logger;
  private cache = new Map<string, ILLM>();

  constructor(logger: pino.Logger) { this.logger = logger; }

  async getLLM(classification: Classification, subscription = 'free', query = ''): Promise<ILLM> {
    const cfg = modelConfigService.getModelConfig(classification, subscription);
    const entry = modelConfigService.getModelRegistryEntry(cfg.modelId);
    if (!entry) {
      this.logger.warn({ modelId: cfg.modelId }, 'Registry entry missing; using fallback');
      return this.getFallback();
    }
    const provider = this.decideProvider({ classification, subscription, query });
    const key = `${entry.bedrockId}:${provider}`;
    if (this.cache.has(key)) return this.cache.get(key)!;
    const llm = this.instantiate(entry, provider);
    this.cache.set(key, llm);
    return llm;
  }

  async getClassifierLLM(): Promise<ILLM> {
    const cfg = modelConfigService.getClassifierConfig();
    const entry = modelConfigService.getModelRegistryEntry(cfg.modelId);
    return entry ? this.reuseOrCreate(entry, 'classifier') : this.getFallback();
  }

  async getRerankerLLM(): Promise<ILLM> {
    const cfg = modelConfigService.getRerankerConfig();
    const entry = modelConfigService.getModelRegistryEntry(cfg.modelId);
    return entry ? this.reuseOrCreate(entry, 'reranker') : this.getFallback();
  }

  private reuseOrCreate(entry: ModelRegistryEntry, tag: string): ILLM {
    const provider = this.inferProvider(entry.bedrockId);
    const key = `${tag}:${entry.bedrockId}:${provider}`;
    if (this.cache.has(key)) return this.cache.get(key)!;
    const llm = this.instantiate(entry, provider);
    this.cache.set(key, llm);
    return llm;
  }

  private instantiate(entry: ModelRegistryEntry, provider: string): ILLM {
    // Try custom adapters first
    const adapter = providerAdapters.find(a => a.supports(entry.bedrockId));
    if (adapter) {
      this.logger.info({ model: entry.bedrockId, provider: adapter.name }, 'Using direct adapter');
      return adapter.create(entry, this.logger);
    }
    // Default to Bedrock wrapper
    return new BedrockLLM(entry, provider, this.logger);
  }

  private decideProvider(ctx: ProviderDecisionContext): string {
    // Intent heuristics
    const q = ctx.query.toLowerCase();
    if (q.includes('proof') || q.includes('derive') || q.includes('rigorous')) return 'anthropic';
    if (q.includes('recent') || q.includes('news')) return 'openai';
    if (ctx.classification.subject === 'math' && ctx.classification.level === 'advanced') return 'openai';
    if (ctx.classification.subject === 'reasoning' && ctx.classification.level !== 'basic') return 'anthropic';
    // Subscription influence
    if (ctx.subscription === 'premium') return 'openai';
    // Fallback on inferred provider from model id
    const cfg = modelConfigService.getModelConfig(ctx.classification, ctx.subscription);
    return this.inferProvider(cfg.modelId);
  }

  private inferProvider(modelId: string): string {
    if (modelId.includes('anthropic') || modelId.includes('claude')) return 'anthropic';
    if (modelId.includes('openai') || modelId.includes('gpt')) return 'openai';
    if (modelId.includes('amazon') || modelId.includes('nova')) return 'amazon';
    return 'generic';
  }

  private getFallback(): ILLM {
    const entry: ModelRegistryEntry = { bedrockId: 'anthropic.claude-3-sonnet-20240229-v1:0', invokeId: 'anthropic.claude-3-sonnet-20240229-v1:0', inferenceProfileArn: null, region: 'ap-south-1' };
    return new BedrockLLM(entry, 'anthropic', this.logger);
  }

  clearCache() { this.cache.clear(); }
  getCacheStats() { return { size: this.cache.size, keys: [...this.cache.keys()] }; }
}
