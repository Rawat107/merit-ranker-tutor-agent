import pino from "pino";
import type { ValidatorConfig, ValidationResult } from "../../types/index.js";

/**
 * Topic Validator and Normalizer
 */

// Re-export types for convenience
export type { ValidatorConfig, ValidationResult };

export class TopicValidator {
  constructor(private logger: pino.Logger) {}

  /**
   * Main validation entry point
   *
   */
  async validateAndNormalizeTopics(
    topics: any[] | undefined,
    totalQuestions: number,
    config: ValidatorConfig = {}
  ): Promise<ValidationResult> {
    // Rule 1: If topics missing or empty, blueprint should generate them
    if (!topics || topics.length === 0) {
      this.logger.info(
        { topicsProvided: false, totalQuestions },
        "[TopicValidator] Topics not provided - blueprint will generate"
      );

      return {
        isValid: true,
        action: "generate_blueprint",
        topics: null,
        reason: "topics_missing",
        metadata: {
          topicsCount: 0,
          totalQuestions,
        },
      };
    }

    // Rule 2: If topics provided but blueprint explicitly requested
    if (config.allowBlueprintWhenTopicsProvided) {
      this.logger.info(
        { topicsCount: topics.length, totalQuestions },
        "[TopicValidator] Blueprint explicitly allowed despite topics - will regenerate"
      );

      return {
        isValid: true,
        action: "generate_blueprint",
        topics: null,
        reason: "explicit_blueprint_flag",
        metadata: {
          inputTopicsCount: topics.length,
          totalQuestions,
          allowBlueprintWhenTopicsProvided: true,
        },
      };
    }

    // Rule 3: Topics are provided - validate and normalize
    const sum = topics.reduce((acc, t) => acc + (t.noOfQuestions || 0), 0);

    this.logger.info(
      {
        topicsCount: topics.length,
        providedSum: sum,
        targetTotal: totalQuestions,
        difference: totalQuestions - sum,
      },
      "[TopicValidator] Validating provided topics"
    );

    // Perfect match
    if (sum === totalQuestions) {
      this.logger.info(
        { topicsCount: topics.length, totalQuestions },
        "[TopicValidator] Topics exactly match totalQuestions - bypass blueprint"
      );

      return {
        isValid: true,
        action: "bypass_blueprint",
        topics: topics,
        reason: "topics_exact_match",
        metadata: {
          topicsCount: topics.length,
          totalQuestions,
          sumMatch: "exact",
        },
      };
    }

    // Undercount: distribute remainder
    if (sum < totalQuestions) {
      const remainder = totalQuestions - sum;
      const adjusted = this.distributeRemainder(
        topics,
        remainder,
        config.distributionStrategy || "round-robin"
      );

      this.logger.info(
        {
          topicsCount: topics.length,
          originalSum: sum,
          remainder,
          newSum: adjusted.reduce((a, t) => a + t.noOfQuestions, 0),
          strategy: config.distributionStrategy || "round-robin",
        },
        "[TopicValidator] Topics undercount - distributed remainder across topics"
      );

      return {
        isValid: true,
        action: "bypass_blueprint",
        topics: adjusted,
        reason: "topics_adjusted_undercount",
        metadata: {
          originalTopicsCount: topics.length,
          totalQuestions,
          originalSum: sum,
          remainder,
          distributionStrategy: config.distributionStrategy || "round-robin",
          change: "increased_question_counts",
        },
      };
    }

    // Overcount: reduce counts from last/lower-priority topics
    const overage = sum - totalQuestions;
    const adjusted = this.reduceOvercount(topics, overage);

    if (!adjusted.valid) {
      this.logger.error(
        {
          topicsCount: topics.length,
          originalSum: sum,
          overage,
          minCountRequired: adjusted.minCountRequired,
        },
        "[TopicValidator] Cannot reduce topics to meet totalQuestions - would violate minimum of 1 per topic"
      );

      return {
        isValid: false,
        action: "generate_blueprint",
        topics: null,
        reason: "topics_overcount_unreducible",
        metadata: {
          topicsCount: topics.length,
          totalQuestions,
          originalSum: sum,
          overage,
          error:
            `Cannot reduce ${overage} questions while keeping each topic at minimum of 1. ` +
            `You have ${topics.length} topics requiring at least ${topics.length} questions total. ` +
            `Please adjust input topics or increase totalQuestions.`,
          minCountRequired: adjusted.minCountRequired,
        },
      };
    }

    this.logger.info(
      {
        topicsCount: topics.length,
        originalSum: sum,
        overage,
        newSum: adjusted.topics!.reduce((a, t) => a + t.noOfQuestions, 0),
      },
      "[TopicValidator] Topics overcount - reduced question counts"
    );

    return {
      isValid: true,
      action: "bypass_blueprint",
      topics: adjusted.topics!,
      reason: "topics_adjusted_overcount",
      metadata: {
        originalTopicsCount: topics.length,
        totalQuestions,
        originalSum: sum,
        overage,
        change: "decreased_question_counts",
      },
    };
  }

  /**
   * Distribute remainder across existing topics (round-robin or priority)
   */
  private distributeRemainder(
    topics: any[],
    remainder: number,
    strategy: "round-robin" | "priority" | "proportional"
  ): any[] {
    const adjusted = JSON.parse(JSON.stringify(topics)); // Deep copy

    if (strategy === "round-robin") {
      for (let i = 0; i < remainder; i++) {
        adjusted[i % adjusted.length].noOfQuestions += 1;
      }
    } else if (strategy === "priority") {
      // Add to first topic (highest priority)
      adjusted[0].noOfQuestions += remainder;
    } else if (strategy === "proportional") {
      // Distribute proportionally based on current counts
      adjusted.forEach((topic: any, index: number) => {
        const proportion = topic.noOfQuestions / topics.reduce((a: any, t: any) => a + t.noOfQuestions, 0);
        const additionalQuestions = Math.round(proportion * remainder);
        adjusted[index].noOfQuestions += additionalQuestions;
      });

      // Fix rounding errors by adjusting the last topic
      const actualSum = adjusted.reduce((a: any, t: any) => a + t.noOfQuestions, 0);
      const totalNeeded = topics.reduce((a: any, t: any) => a + t.noOfQuestions, 0) + remainder;
      if (actualSum !== totalNeeded) {
        adjusted[adjusted.length - 1].noOfQuestions += totalNeeded - actualSum;
      }
    }

    return adjusted;
  }

  /**
   * Reduce overcount by trimming question counts from last/lower-priority topics
   */
  private reduceOvercount(
    topics: any[],
    overage: number
  ): { valid: boolean; topics?: any[]; minCountRequired?: number } {
    // Check if it's possible to reduce
    // Minimum requirement: 1 question per topic
    const minRequired = topics.length;
    const currentSum = topics.reduce((acc, t) => acc + t.noOfQuestions, 0);
    const targetSum = currentSum - overage;

    if (targetSum < minRequired) {
      return {
        valid: false,
        minCountRequired: minRequired,
      };
    }

    const adjusted = JSON.parse(JSON.stringify(topics)); // Deep copy
    let remaining = overage;

    // Reduce from the end (last topics first - lowest priority)
    for (let i = adjusted.length - 1; i >= 0 && remaining > 0; i--) {
      const canReduce = adjusted[i].noOfQuestions - 1; // Minimum 1 per topic
      const toReduce = Math.min(canReduce, remaining);
      adjusted[i].noOfQuestions -= toReduce;
      remaining -= toReduce;
    }

    return {
      valid: true,
      topics: adjusted,
    };
  }
}

export function createTopicValidator(logger: pino.Logger): TopicValidator {
  return new TopicValidator(logger);
}
