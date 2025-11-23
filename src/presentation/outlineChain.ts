import { RedisCache } from '../cache/RedisCache.js';
import { webSearchTool } from '../tools/webSearch.js';
import { createTierLLM } from '../llm/tierLLM.js';
import { modelConfigService } from '../config/modelConfig.js';
import { buildPresentationOutlinePrompt } from '../utils/promptTemplates.js';
import {
  SlideOutlineRequest,
  PresentationOutlineResponse,
  SlideOutline,
} from '../types/index.js';
import pino from 'pino';
import crypto from 'crypto';

/**
 * PresentationOutlineChain - Generates presentation outlines using LangChain
 * Uses RedisCache's storeDirectCache and checkDirectCache methods
 */
export class PresentationOutlineChain {
  constructor(
    private redisCache: RedisCache,
    private logger: pino.Logger,
    private tavilyApiKey: string
  ) {}

  /**
   * Generate presentation outline
   */
  async generateOutline(
    request: SlideOutlineRequest
  ): Promise<PresentationOutlineResponse> {
    const slideId = crypto.randomUUID();

    try {
      this.logger.info(
        { slideId, title: request.title },
        '[PresentationOutline] Starting generation'
      );

      // Step 1: Web Search
      const searchResults = await this.performWebSearch(request);

      // Step 2: Generate outline using LLM
      const outline = await this.generateOutlineWithLLM(request, searchResults);

      // Step 3: Build response
      const response: PresentationOutlineResponse = {
        slideId,
        userId: request.userId,
        title: request.title,
        status: 'READY',
        noOfSlides: request.noOfSlides,
        level: request.level,
        language: request.language,
        designStyle: request.designStyle,
        colorTheme: request.colorTheme,
        outline,
        webSearchResults: JSON.stringify(searchResults),
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      };

      // Step 4: Cache outline using RedisCache.storeDirectCache
      const cacheKey = `presentation:outline:${slideId}`;
      await this.redisCache.storeDirectCache(
        cacheKey,
        JSON.stringify(response),
        'presentation',
        { slideId, userId: request.userId, title: request.title }
      );

      this.logger.info(
        { slideId, outlineSize: outline.length },
        '[PresentationOutline] Generated and cached'
      );

      return response;
    } catch (error) {
      this.logger.error(
        { slideId, error },
        '[PresentationOutline] Generation failed'
      );
      throw error;
    }
  }

  /**
   * Web search using Tavily
   */
  private async performWebSearch(request: SlideOutlineRequest): Promise<any> {
    try {
      this.logger.info(
        { title: request.title },
        '[PresentationOutline] Performing web search'
      );

      const searchQuery = `${request.title} ${request.description || ''}`;

      const results = await webSearchTool(
        searchQuery,
        request.designStyle,
        this.tavilyApiKey,
        this.logger
      );

      return Array.isArray(results) ? results.slice(0, 5) : [];
    } catch (error) {
      this.logger.warn(
        { error },
        '[PresentationOutline] Web search failed, continuing without results'
      );
      return [];
    }
  }

  /**
   * Generate outline using LangChain Bedrock LLM
   */
  private async generateOutlineWithLLM(
    request: SlideOutlineRequest,
    searchResults: any[]
  ): Promise<SlideOutline[]> {
    try {
      const searchContext = searchResults
        .map((r) => `- ${r.title || ''}: ${r.text || r.content || ''}`)
        .join('\n');

      const prompt = buildPresentationOutlinePrompt(request, searchContext);

      const config = modelConfigService.getModelConfig(
        {
          subject: 'general',
          level: 'intermediate',
          confidence: 0.9,
        },
        'free'
      );

      const registryEntry = modelConfigService.getModelRegistryEntry(
        config.modelId
      );

      if (!registryEntry) {
        throw new Error(`Registry entry not found for model ${config.modelId}`);
      }

      const llm = createTierLLM(
        'intermediate',
        registryEntry,
        this.logger,
        0.7,
        2000
      );

      this.logger.info(
        { modelInfo: llm.getModelInfo() },
        '[PresentationOutline] LLM created'
      );

      const response = await llm.generate(prompt);

      const jsonMatch = response.match(/\[[\s\S]*\]/);
      if (!jsonMatch) {
        this.logger.warn(
          { response: response.substring(0, 200) },
          '[PresentationOutline] Failed to extract JSON'
        );
        throw new Error('Failed to parse outline JSON from LLM response');
      }

      const outline: SlideOutline[] = JSON.parse(jsonMatch[0]);

      this.logger.info(
        { slideCount: outline.length },
        '[PresentationOutline] Outline parsed successfully'
      );

      return outline;
    } catch (error) {
      this.logger.error(
        { error },
        '[PresentationOutline] LLM generation failed'
      );
      throw error;
    }
  }

  /**
   * Update outline (merge user edits)
   */
  async updateOutline(
    slideId: string,
    updates: Partial<SlideOutline>[]
  ): Promise<PresentationOutlineResponse> {
    try {
      // Retrieve from cache
      const cacheKey = `presentation:outline:${slideId}`;
      const cached = await this.redisCache.checkDirectCache(cacheKey, 'presentation');

      if (!cached) {
        throw new Error('Outline not found in cache');
      }

      const response: PresentationOutlineResponse = JSON.parse(cached.response);

      // Merge updates
      response.outline = response.outline.map((slide) => {
        const update = updates.find((u) => u.slideNumber === slide.slideNumber);
        return update ? { ...slide, ...update } : slide;
      });

      response.updatedAt = new Date().toISOString();

      // Update cache
      await this.redisCache.storeDirectCache(
        cacheKey,
        JSON.stringify(response),
        'presentation',
        { slideId, userId: response.userId }
      );

      this.logger.info({ slideId }, '[PresentationOutline] Updated in cache');
      return response;
    } catch (error) {
      this.logger.error(
        { slideId, error },
        '[PresentationOutline] Update failed'
      );
      throw error;
    }
  }

  /**
   * Get outline from Redis using checkDirectCache
   */
  async getOutline(slideId: string): Promise<PresentationOutlineResponse> {
    const cacheKey = `presentation:outline:${slideId}`;
    const cached = await this.redisCache.checkDirectCache(cacheKey, 'presentation');

    if (!cached) {
      throw new Error('Outline not found in cache');
    }

    return JSON.parse(cached.response);
  }
}
