import pino from "pino";
import { z } from "zod";
import { modelConfigService } from "../../config/modelConfig.ts";
import { buildBlueprintGenerationPrompt } from "../../utils/promptTemplates.ts";
import { createTierLLM } from "../../llm/tierLLM.ts";
import { Classification, AssessmentCategory } from "../../types/index.ts";
import { createTopicValidator, ValidatorConfig } from "./topicValidator.ts";

// Standardized blueprint schema matching AssessmentRequest structure
export const BlueprintSchema = z.object({
  examTags: z.array(z.string()),
  subject: z.string(),
  totalQuestions: z.number().min(1).max(150),
  topics: z.array(
    z.object({
      topicName: z.string(),
      level: z.array(z.enum(["easy", "medium", "hard", "mix"])),
      noOfQuestions: z.number(),
    })
  ),
});

export type Blueprint = z.infer<typeof BlueprintSchema>;

/**
 * BLUEPRINT GENERATOR
 * Creates structured plans for question generation
 *
 * Execution rules:
 * - If topics are missing/empty: generate blueprint with LLM
 * - If topics are provided: validate and normalize to totalQuestions, bypass blueprint
 * - Only adjust question counts, never invent new topics (unless input was empty)
 *
 * Uses existing infrastructure:
 * - modelConfigService for model selection
 * - createTierLLM for LLM initialization to generate blueprint
 * - buildBlueprintGenerationPrompt from promptTemplates
 * - TopicValidator for topic validation and normalization
 */
export class BlueprintGenerator {
  private topicValidator: ReturnType<typeof createTopicValidator>;
  private validatorConfig: ValidatorConfig;

  constructor(
    private logger: pino.Logger,
    validatorConfig?: ValidatorConfig
  ) {
    this.topicValidator = createTopicValidator(logger);
    this.validatorConfig = {
      maxQuestionsPerNewTopic: 10,
      distributionStrategy: "round-robin",
      allowBlueprintWhenTopicsProvided: false,
      ...validatorConfig,
    };
  }

  /**
   * Main entry point
   * 1. Validates and normalizes user-provided topics (if any)
   * 2. Generates blueprint only if topics are missing/empty
   * 3. Returns standardized request object with validated topics
   */
  async generateBlueprint(
    request: { examTags: string[]; subject: string; totalQuestions: number; topics?: any[] },
    classification: Classification,
    assessmentCategory: AssessmentCategory = 'quiz'
  ): Promise<Blueprint> {
    try {
      this.logger.info(
        {
          examTags: request.examTags,
          subject: request.subject,
          totalQuestions: request.totalQuestions,
          topicsProvided: request.topics?.length || 0,
          topicsArray: request.topics || [],
        },
        "[Blueprint] Starting blueprint generation (new execution rules)"
      );

      // Step 1: Validate and normalize provided topics
      const validationResult = await this.topicValidator.validateAndNormalizeTopics(
        request.topics,
        request.totalQuestions,
        this.validatorConfig
      );

      this.logger.info(
        {
          action: validationResult.action,
          reason: validationResult.reason,
          inputTopics: request.topics?.length || 0,
          outputTopics: validationResult.topics?.length || 0,
          metadata: validationResult.metadata,
        },
        "[Blueprint] Topic validation decision"
      );

      // Step 2: If topics are valid and present, use them as-is
      if (validationResult.action === "bypass_blueprint") {
        if (!validationResult.topics) {
          throw new Error("Validation returned bypass_blueprint but topics is null");
        }

        const blueprint: Blueprint = {
          examTags: request.examTags,
          subject: request.subject,
          totalQuestions: request.totalQuestions,
          topics: validationResult.topics.map(t => ({
            topicName: t.topicName,
            level: (t.level || ["mix"]) as ("easy" | "medium" | "hard" | "mix")[],
            noOfQuestions: t.noOfQuestions,
          })),
        };

        const validated = BlueprintSchema.parse(blueprint);
        this.logger.info(
          {
            totalQuestions: validated.totalQuestions,
            topicsCount: validated.topics.length,
            reason: validationResult.reason,
            change: validationResult.metadata.change || "none",
          },
          "[Blueprint] ✅ Using user-provided topics (normalized)"
        );
        return validated;
      }

      // Step 3: If validation failed or topics are invalid, return error
      if (!validationResult.isValid) {
        throw new Error(
          validationResult.metadata.error ||
            "Topic validation failed and blueprint is not allowed"
        );
      }

      // Step 4: Generate topics using LLM
      this.logger.info(
        {
          reason: validationResult.reason,
          totalQuestions: request.totalQuestions,
        },
        "[Blueprint] No valid topics provided, generating with LLM"
      );

      const generatedTopics = await this.generateTopicsWithLLM(
        request.subject,
        request.totalQuestions,
        classification,
        request.examTags
      );

      const blueprint: Blueprint = {
        examTags: request.examTags,
        subject: request.subject,
        totalQuestions: request.totalQuestions,
        topics: generatedTopics,
      };

      const validated = BlueprintSchema.parse(blueprint);
      this.logger.info(
        {
          totalQuestions: validated.totalQuestions,
          topicsCount: validated.topics.length,
          topicDetails: validated.topics.map(t => `${t.topicName} (${t.noOfQuestions}q)`),
        },
        "[Blueprint] ✅ Blueprint generated with LLM topics"
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
   * Generate topics using LLM when user doesn't provide them
   */
  private async generateTopicsWithLLM(
    subject: string,
    totalQuestions: number,
    classification: Classification,
    examTags: string[]
  ): Promise<Blueprint['topics']> {
    try {
      const modelConfig = modelConfigService.getModelConfig(classification, "free");
      const modelRegistry = modelConfigService.getModelRegistryEntry(modelConfig.modelId);
      
      if (!modelRegistry) {
        throw new Error(`Model registry not found for ${modelConfig.modelId}`);
      }

      const tier = classification.level as "basic" | "intermediate" | "advanced";
      const llm = createTierLLM(tier, modelRegistry, this.logger, 0.2, 1024);

      const maxQuestionsPerTopic = this.validatorConfig.maxQuestionsPerNewTopic || 10;
      const prompt = buildBlueprintGenerationPrompt(
        examTags,
        subject,
        totalQuestions,
        classification.level,
        maxQuestionsPerTopic
      );

      const response = await llm.generate(prompt);
      const parsed = JSON.parse(response);
      
      if (parsed.topics && Array.isArray(parsed.topics)) {
        return parsed.topics;
      }
      
      throw new Error('Invalid LLM response format');
    } catch (error) {
      this.logger.warn(
        { error },
        "[Blueprint] LLM topic generation failed, using fallback"
      );
      return this.generateTopicsFallback(subject, totalQuestions, classification);
    }
  }

  /**
   * Fallback: Generate topics heuristically when LLM fails
   */
  private generateTopicsFallback(
    subject: string,
    totalQuestions: number,
    classification: Classification
  ): Blueprint['topics'] {
    this.logger.info("[Blueprint] Using fallback heuristic topic generation");

    const topics = this.getTopicsForSubject(subject);
    const maxQuestionsPerTopic = this.validatorConfig.maxQuestionsPerNewTopic || 10;
    const topicCount = Math.min(
      5,
      Math.max(3, Math.ceil(totalQuestions / maxQuestionsPerTopic))
    );
    const questionsPerTopic = Math.floor(totalQuestions / topicCount);
    const remainder = totalQuestions % topicCount;

    const result: Blueprint['topics'] = [];
    for (let i = 0; i < topicCount && i < topics.length; i++) {
      result.push({
        topicName: topics[i],
        level: ["mix"],
        noOfQuestions: questionsPerTopic + (i < remainder ? 1 : 0),
      });
    }

    return result;
  }

  /**
   * Get topics for a given subject (helper method)
   */
  private getTopicsForSubject(subject: string): string[] {
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
      quantitative_aptitude: [
        "Profit and Loss",
        "Time and Work",
        "Time Speed Distance",
        "Ratio and Proportion",
        "Percentage",
        "Simple and Compound Interest",
        "Data Interpretation",
        "Number System",
        "Averages",
        "Mixtures and Alligations",
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

    const normalizedSubject = subject.toLowerCase().replace(/\s+/g, '_');
    return topicMap[normalizedSubject as keyof typeof topicMap] || topicMap.general;
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