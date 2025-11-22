import { createTierLLM } from '../llm/tierLLM.js';
import { modelConfigService } from '../config/modelConfig.js';
import { buildSlideContentPrompt } from '../utils/promptTemplates.js';
import { PresentationSlideContent, SlideOutline } from '../types/index.js';
import { createImageGenerator, TogetherImageGenerator, ImageModelList } from '../tools/imageGeneration.js';
import pino from 'pino';

/**
 * PresentationContentChain - Generates slide content + images using Together AI for images
 */
export class PresentationContentChain {
  private imageGenerator: TogetherImageGenerator;
  constructor(private logger: pino.Logger) {
    this.imageGenerator = createImageGenerator(this.logger);
  }

  /**
   * Generate content + images for all slides
   */
  async generateSlideContent(
    outline: SlideOutline[],
    context: string
  ): Promise<PresentationSlideContent[]> {
    const slidesContent: PresentationSlideContent[] = [];
    for (const slide of outline) {
      try {
        this.logger.info(
          { slideNumber: slide.slideNumber },
          '[PresentationContent] Generating content'
        );
        // Generate content using LLM
        const contentData = await this.generateContent(slide, context);

        // Generate image directly via Together AI
        const imagePrompt = slide.title; // Can enhance if you want richer prompts
        const imageUrl = await this.imageGenerator.generate(
          imagePrompt
        );

        slidesContent.push({
          slideNumber: slide.slideNumber,
          title: slide.title,
          content: contentData.content,
          keyPoints: slide.keyPoints,
          imageUrl: imageUrl || undefined,
          imageAlt: imagePrompt,
          speakerNotes: slide.speakerNotes,
        });
      } catch (error) {
        this.logger.error(
          {
            slideNumber: slide.slideNumber,
            error: error instanceof Error ? error.message : String(error),
            stack: error instanceof Error ? error.stack : undefined,
          },
          '[PresentationContent] Content or Image generation failed'
        );
        // Fallback: basic content without image
        slidesContent.push({
          slideNumber: slide.slideNumber,
          title: slide.title,
          content: `## ${slide.title}\n\n${slide.keyPoints
            .map((p) => `- ${p}`)
            .join('\n')}`,
          keyPoints: slide.keyPoints,
          imageUrl: undefined,
          imageAlt: slide.title,
          speakerNotes: slide.speakerNotes,
        });
      }
    }
    return slidesContent;
  }

  /**
   * Generate content using LangChain LLM
   */
  private async generateContent(
    slide: SlideOutline,
    context: string
  ): Promise<{ content: string }> {
    try {
      const prompt = buildSlideContentPrompt(slide, context);

      // Get model config
      const config = modelConfigService.getModelConfig(
        {
          subject: 'general',
          level: 'basic',
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
      // Create LLM using tier system
      const llm = createTierLLM(
        'basic',
        registryEntry,
        this.logger,
        0.7,
        800
      );
      // Generate content
      const response = await llm.generate(prompt);

      this.logger.debug(
        { responseLength: response.length },
        '[PresentationContent] LLM response received'
      );

      // Return as one markdown block (you may parse more JSON if you want: see above)
      return { content: response };
    } catch (error) {
      this.logger.error(
        {
          error: error instanceof Error ? error.message : String(error),
          slide: slide.slideNumber,
        },
        '[PresentationContent] Content generation error'
      );
      throw error;
    }
  }
}
