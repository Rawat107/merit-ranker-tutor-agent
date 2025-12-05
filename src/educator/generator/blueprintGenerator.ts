import pino from "pino";
import { z } from "zod";
import { modelConfigService } from "../../config/modelConfig.ts";
import { buildBlueprintGenerationPrompt } from "../../utils/promptTemplates.ts";
import { createTierLLM } from "../../llm/tierLLM.ts";
import { Classification } from "../../types/index.ts";

export const BlueprintSchema = z.object({
  totalQuestions: z.number().min(1).max(100),
  numberOfBatches: z.number().min(1),
  topics: z.array(
    z.object({
      topicName: z.string(),
      description: z.string(),
      difficulty: z.enum(["basic", "intermediate", "advanced"]),
      questionCount: z.number(),
      priority: z.enum(["high", "medium", "low"]),
    })
  ),
  selectedModel: z.object({
    name: z.string(),
    modelId: z.string(),
    temperature: z.number(),
    reason: z.string(),
  }),
  pipelineNodes: z.array(z.string()),
  generationStrategy: z.object({
    batchSize: z.number(),
    concurrency: z.number(),
    retryLimit: z.number(),
  }),
  metadata: z.object({
    estimatedTime: z.string(),
    userPreferences: z.record(z.any()).optional(),
    contextFromResearch: z.boolean(),
  }),
});

export type Blueprint = z.infer<typeof BlueprintSchema>;

/**
 * BLUEPRINT GENERATOR
 * Creates structured plans for question generation
 *
 * Uses existing infrastructure:
 * - modelConfigService for model selection
 * - createTierLLM for LLM initialization to generate blueprint
 * - buildBlueprintGenerationPrompt from promptTemplates
 */
export class BlueprintGenerator {
  constructor(private logger: pino.Logger) {}

  /**
   * Main entry point
   * Generates blueprint from query + classification using LLM
   */
  async generateBlueprint(
    query: string,
    classification: Classification
  ): Promise<Blueprint> {
    try {
      this.logger.info(
        {
          query: query.substring(0, 60),
          subject: classification.subject,
          level: classification.level,
        },
        "[Blueprint] Starting blueprint generation with LLM"
      );

      // STEP 1: Get model config for blueprint generation (Intermediate tier)
      const modelConfig = modelConfigService.getModelConfig(
        classification,
        "free"
      );

      const modelRegistry = modelConfigService.getModelRegistryEntry(
        modelConfig.modelId
      );

      if (!modelRegistry) {
        throw new Error(
          `Model registry not found for ${modelConfig.modelId}`
        );
      }

      this.logger.debug(
        {
          modelId: modelConfig.modelId,
          temperature: modelConfig.temperature,
        },
        "[Blueprint] Model config retrieved"
      );

      // STEP 2: Create LLM instance for blueprint generation
      const tier = (classification.level as "basic" | "intermediate" | "advanced");
      const llm = createTierLLM(
        tier,
        modelRegistry,
        this.logger,
        0.2, // Deterministic: low temperature for consistent planning
        1024 // Max tokens for blueprint
      );

      this.logger.debug(
        { modelInfo: llm.getModelInfo() },
        "[Blueprint] LLM instance created for blueprint generation"
      );

      // STEP 3: Build prompt using buildBlueprintGenerationPrompt
      const prompt = buildBlueprintGenerationPrompt(query, classification);

      this.logger.debug(
        { promptLength: prompt.length },
        "[Blueprint] Prompt built for blueprint generation"
      );

      // STEP 4: Generate blueprint via LLM
      const blueprintJson = await llm.generate(prompt);

      this.logger.debug(
        { responseLength: blueprintJson.length },
        "[Blueprint] LLM generated blueprint response"
      );

      // STEP 5: Parse and validate JSON response
      let parsedBlueprint: any;
      try {
        parsedBlueprint = JSON.parse(blueprintJson);
      } catch (parseError) {
        this.logger.warn(
          { error: parseError, response: blueprintJson.substring(0, 200) },
          "[Blueprint] Failed to parse LLM response as JSON, using fallback heuristics"
        );
        parsedBlueprint = this.fallbackBlueprintFromQuery(
          query,
          classification
        );
      }

      // STEP 6: Enhance with computed values
      const totalQuestions = parsedBlueprint.totalQuestions || this.extractQuestionCount(query);
      const numberOfBatches = Math.ceil(totalQuestions / 5);

      const blueprint: Blueprint = {
        totalQuestions,
        numberOfBatches,
        topics: parsedBlueprint.topics || this.generateTopics(classification, totalQuestions),
        selectedModel: {
          name: this.getModelName(modelConfig.modelId),
          modelId: modelConfig.modelId,
          temperature: modelConfig.temperature,
          reason:
            parsedBlueprint.selectedModel?.reason ||
            `${tier.charAt(0).toUpperCase() + tier.slice(1)} difficulty - model selected by modelConfigService`,
        },
        pipelineNodes: parsedBlueprint.pipelineNodes || [
          "research",
          "topic_batch_creation",
          "generate",
          "validate",
        ],
        generationStrategy: parsedBlueprint.generationStrategy || {
          batchSize: 5,
          concurrency: 5,
          retryLimit: 3,
        },
        metadata: {
          estimatedTime: `${numberOfBatches * 8} seconds`,
          userPreferences: {
            subject: classification.subject,
            level: classification.level,
          },
          contextFromResearch: false,
        },
      };

      // STEP 7: Validate blueprint
      const validated = BlueprintSchema.parse(blueprint);

      this.logger.info(
        {
          totalQuestions: validated.totalQuestions,
          numberOfBatches: validated.numberOfBatches,
          topicsCount: validated.topics.length,
          selectedModel: validated.selectedModel.name,
          modelId: validated.selectedModel.modelId,
        },
        "[Blueprint] ✅ Blueprint generated successfully via LLM"
      );

      return validated;
    } catch (error) {
      this.logger.error(
        { error },
        "[Blueprint] ❌ Blueprint generation failed"
      );
      throw error;
    }
  }

  /**
   * Fallback: Generate blueprint heuristically when LLM fails
   * Uses same logic as before but ensures it works
   */
  private fallbackBlueprintFromQuery(
    query: string,
    classification: Classification
  ): Partial<Blueprint> {
    this.logger.info(
      "[Blueprint] Using fallback heuristic blueprint generation"
    );

    const totalQuestions = this.extractQuestionCount(query);
    const topics = this.generateTopics(classification, totalQuestions);

    return {
      totalQuestions,
      topics,
      pipelineNodes: [
        "research",
        "topic_batch_creation",
        "generate",
        "validate",
      ],
      generationStrategy: {
        batchSize: 3,
        concurrency: 5,
        retryLimit: 3,
      },
    };
  }

  /**
   * Extract question count from query
   * Looks for patterns like "5 questions", "20 MCQs", etc.
   */
  private extractQuestionCount(query: string): number {
    const match = query.match(/(\d+)\s*(questions?|mcqs?|quiz|test)/i);
    if (match) {
      return Math.min(parseInt(match[1]), 100); // Max 100
    }
    return 5; // Default
  }

  /**
   * Generate topics based on subject
   * Distributes questions across relevant topics
   * Reduced to 2-3 questions per topic for better topic diversity
   */
  private generateTopics(
    classification: Classification,
    totalQuestions: number
  ): Blueprint["topics"] {
    const { subject, level } = classification;

    // Subject → Topics mapping
    const topicMap: Record<string, string[]> = {
      math: [
        "Arithmetic",
        "Algebra",
        "Geometry",
        "Trigonometry",
        "Calculus",
        "Statistics",
        "Linear Algebra",
      ],
      science: [
        "Physics",
        "Chemistry",
        "Biology",
        "Ecology",
        "Astronomy",
        "Earth Science",
        "Genetics",
      ],
      english_grammar: [
        "Nouns & Pronouns",
        "Verbs & Tenses",
        "Sentence Structure",
        "Punctuation",
        "Vocabulary",
        "Idioms & Phrases",
        "Reading Comprehension",
      ],
      history: [
        "Ancient Civilizations",
        "Medieval Period",
        "Modern History",
        "World Wars",
        "Independence Movements",
        "Renaissance",
        "Industrial Revolution",
      ],
      reasoning: [
        "Logic & Puzzles",
        "Pattern Recognition",
        "Probability",
        "Combinatorics",
        "Game Theory",
        "Analytical Reasoning",
        "Deductive Logic",
      ],
      general: [
        "General Knowledge",
        "Current Affairs",
        "Science & Technology",
        "History & Culture",
        "Geography",
        "Sports",
      ],
    };

    const subjectTopics =
      topicMap[subject as keyof typeof topicMap] || topicMap.general;

    // Distribute questions across more topics (2-3 per topic instead of 5)
    // This gives better diversity: 20 questions = 6-7 topics with 2-3 questions each
    const questionsPerTopic = Math.max(2, Math.ceil(totalQuestions / Math.ceil(totalQuestions / 3)));
    const topicsToUse = Math.min(
      Math.ceil(totalQuestions / questionsPerTopic),
      subjectTopics.length
    );

    return subjectTopics.slice(0, topicsToUse).map((topicName, index) => {
      const baseQuestionCount = Math.floor(totalQuestions / topicsToUse);
      const remainder = totalQuestions % topicsToUse;
      
      return {
        topicName,
        description: `Focus on ${topicName} at ${level} level. Include diverse question types covering different aspects.`,
        difficulty: level as "basic" | "intermediate" | "advanced",
        questionCount: index < remainder ? baseQuestionCount + 1 : baseQuestionCount,
        priority: index === 0 ? "high" : index < Math.ceil(topicsToUse / 2) ? "medium" : "low",
      };
    });
  }

  /**
   * Get human-readable model name
   */
  private getModelName(modelId: string): string {
    const names: Record<string, string> = {
      "anthropic.claude-3-haiku-20240307-v1:0": "Claude Haiku",
      "anthropic.claude-3-5-sonnet-20241022-v2:0": "Claude Sonnet 3.5",
      "anthropic.claude-sonnet-4-20250514-v1:0": "Claude Sonnet 4",
      "anthropic.claude-3-sonnet-20240229-v1:0": "Claude Sonnet 3",
    };

    return names[modelId] || "Custom Model";
  }
}

/**
 * Factory function
 */
export function createBlueprintGenerator(
  logger: pino.Logger
): BlueprintGenerator {
  return new BlueprintGenerator(logger);
}