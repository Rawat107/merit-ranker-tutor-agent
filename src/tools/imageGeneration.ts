import Together from "together-ai";
import pino from "pino";

export type ImageModelList =
  | "black-forest-labs/FLUX1.1-pro"
  | "black-forest-labs/FLUX.1-schnell"
  | "black-forest-labs/FLUX.1-schnell-Free"
  | "black-forest-labs/FLUX.1-pro"
  | "black-forest-labs/FLUX.1-dev";

const DEFAULT_MODEL: ImageModelList = "black-forest-labs/FLUX.1-schnell-Free";

export class TogetherImageGenerator {
  private client: Together;
  private logger: pino.Logger;

  constructor(logger: pino.Logger, apiKey?: string) {
    this.logger = logger;
    this.client = new Together({
      apiKey: apiKey || process.env.TOGETHER_AI_API_KEY,
    });
  }

  async generate(
    prompt: string,
    model: ImageModelList = DEFAULT_MODEL
  ): Promise<string> {
    this.logger.info({ model, prompt }, "[TogetherImageGenerator] Generating image");

    const response = (await this.client.images.create({
      model,
      prompt,
      width: 1024,
      height: 768,
      steps: model.includes("schnell") ? 4 : 28,
      n: 1,
    })) as unknown as {
      data: { url: string }[];
    };

    const imageUrl = response?.data?.[0]?.url;

    if (!imageUrl) {
      this.logger.error(`[TogetherImageGenerator] No image url returned for prompt: ${prompt}`);
      throw new Error("No image url returned from Together AI");
    }
    this.logger.info({ imageUrl }, "[TogetherImageGenerator] Image created");
    return imageUrl;
  }
}

export function createImageGenerator(logger: pino.Logger) {
  return new TogetherImageGenerator(logger);
}
