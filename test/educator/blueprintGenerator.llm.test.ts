import { describe, it, expect, beforeEach } from "vitest";
import pino from "pino";
import {
  BlueprintGenerator,
  BlueprintSchema,
} from "../../src/educator/generator/blueprintGenerator";
import { Classification } from "../../src/types";
import { buildBlueprintGenerationPrompt } from "../../src/utils/promptTemplates";

/**
 * LLM INTEGRATION TESTS FOR BLUEPRINT GENERATOR
 * 
 * These tests validate that:
 * 1. The prompt template is correctly used
 * 2. The LLM receives properly formatted prompts
 * 3. The response parsing works correctly
 * 4. The blueprint schema validation works
 * 
 * This uses REAL LLM calls (not mocked) to verify end-to-end behavior
 */

describe("BlueprintGenerator - LLM Integration Tests", () => {
  let generator: BlueprintGenerator;
  let logger: pino.Logger;

  beforeEach(() => {
    logger = pino({ level: "silent" }); // Reduce noise in tests
    generator = new BlueprintGenerator(logger);
  });

  describe("Prompt Template Usage", () => {
    it("should generate correct prompt for basic math query", () => {
      const query = "Generate 10 math questions on arithmetic";
      const classification: Classification = {
        subject: "math",
        level: "basic",
        confidence: 0.95,
      };

      const prompt = buildBlueprintGenerationPrompt(query, classification);

      // Validate prompt contains required elements
      expect(prompt).toContain(query);
      expect(prompt).toContain("math");
      expect(prompt).toContain("basic");
      expect(prompt).toContain("totalQuestions");
      expect(prompt).toContain("numberOfBatches");
      expect(prompt).toContain("topics");
      expect(prompt).toContain("pipelineNodes");
      expect(prompt).toContain("generationStrategy");
    });

    it("should generate correct prompt for intermediate science query", () => {
      const query = "Create 20 physics questions about mechanics";
      const classification: Classification = {
        subject: "science",
        level: "intermediate",
        confidence: 0.88,
      };

      const prompt = buildBlueprintGenerationPrompt(query, classification);

      expect(prompt).toContain(query);
      expect(prompt).toContain("science");
      expect(prompt).toContain("intermediate");
      expect(prompt).toContain("physics");
    });

    it("should generate correct prompt for advanced reasoning query", () => {
      const query = "Generate 15 logic puzzle questions";
      const classification: Classification = {
        subject: "reasoning",
        level: "advanced",
        confidence: 0.92,
      };

      const prompt = buildBlueprintGenerationPrompt(query, classification);

      expect(prompt).toContain(query);
      expect(prompt).toContain("reasoning");
      expect(prompt).toContain("advanced");
      expect(prompt).toContain("logic");
    });
  });

  describe("LLM Response - Full Integration", () => {
    it("should generate blueprint with real LLM for math query", async () => {
      const query = "Generate 10 math questions on algebra";
      const classification: Classification = {
        subject: "math",
        level: "intermediate",
        confidence: 0.9,
      };

      const blueprint = await generator.generateBlueprint(query, classification);

      // Validate blueprint structure
      expect(blueprint).toBeDefined();
      expect(blueprint.totalQuestions).toBeGreaterThanOrEqual(5);
      expect(blueprint.totalQuestions).toBeLessThanOrEqual(100);
      expect(blueprint.numberOfBatches).toBeGreaterThan(0);
      expect(blueprint.topics).toBeInstanceOf(Array);
      expect(blueprint.topics.length).toBeGreaterThan(0);

      // Validate first topic
      const firstTopic = blueprint.topics[0];
      expect(firstTopic.topicName).toBeTruthy();
      expect(firstTopic.description).toBeTruthy();
      expect(["basic", "intermediate", "advanced"]).toContain(firstTopic.difficulty);
      expect(firstTopic.questionCount).toBeGreaterThan(0);
      expect(["high", "medium", "low"]).toContain(firstTopic.priority);

      // Validate selectedModel
      expect(blueprint.selectedModel).toBeDefined();
      expect(blueprint.selectedModel.name).toBeTruthy();
      expect(blueprint.selectedModel.modelId).toBeTruthy();
      expect(blueprint.selectedModel.temperature).toBeGreaterThanOrEqual(0);
      expect(blueprint.selectedModel.temperature).toBeLessThanOrEqual(1);

      // Validate pipelineNodes
      expect(blueprint.pipelineNodes).toContain("research");
      expect(blueprint.pipelineNodes).toContain("generate");
      expect(blueprint.pipelineNodes).toContain("validate");

      // Validate generationStrategy
      expect(blueprint.generationStrategy.batchSize).toBeGreaterThan(0);
      expect(blueprint.generationStrategy.concurrency).toBeGreaterThan(0);
      expect(blueprint.generationStrategy.retryLimit).toBeGreaterThan(0);

      // Validate metadata
      expect(blueprint.metadata.estimatedTime).toBeTruthy();
      expect(blueprint.metadata.contextFromResearch).toBeDefined();

      console.log("\n✅ Generated Blueprint (Math - Intermediate):");
      console.log(JSON.stringify(blueprint, null, 2));
    }, 30000); // 30 second timeout for LLM call

    it("should generate blueprint with real LLM for science query", async () => {
      const query = "Create 5 basic science questions about photosynthesis";
      const classification: Classification = {
        subject: "science",
        level: "basic",
        confidence: 0.95,
      };

      const blueprint = await generator.generateBlueprint(query, classification);

      expect(blueprint.totalQuestions).toBeGreaterThanOrEqual(5);
      expect(blueprint.topics[0].topicName).toBeTruthy();
      expect(blueprint.selectedModel.modelId).toContain("claude");

      console.log("\n✅ Generated Blueprint (Science - Basic):");
      console.log(JSON.stringify(blueprint, null, 2));
    }, 30000);

    it("should generate blueprint with real LLM for reasoning query", async () => {
      const query = "Generate 20 advanced reasoning questions on probability";
      const classification: Classification = {
        subject: "reasoning",
        level: "advanced",
        confidence: 0.92,
      };

      const blueprint = await generator.generateBlueprint(query, classification);

      expect(blueprint.totalQuestions).toBe(20);
      expect(blueprint.numberOfBatches).toBe(Math.ceil(20 / 5));
      
      // Should have multiple topics for 20 questions
      expect(blueprint.topics.length).toBeGreaterThan(1);

      // Total questions across topics should sum to totalQuestions
      const sumQuestions = blueprint.topics.reduce((sum, t) => sum + t.questionCount, 0);
      expect(sumQuestions).toBe(20);

      console.log("\n✅ Generated Blueprint (Reasoning - Advanced):");
      console.log(JSON.stringify(blueprint, null, 2));
    }, 30000);

    it("should handle query without explicit question count", async () => {
      const query = "Create some history questions about ancient civilizations";
      const classification: Classification = {
        subject: "history",
        level: "intermediate",
        confidence: 0.88,
      };

      const blueprint = await generator.generateBlueprint(query, classification);

      // Should default to 5 questions
      expect(blueprint.totalQuestions).toBeGreaterThanOrEqual(5);
      expect(blueprint.topics.length).toBeGreaterThan(0);

      console.log("\n✅ Generated Blueprint (History - Default Count):");
      console.log(JSON.stringify(blueprint, null, 2));
    }, 30000);
  });

  describe("Response Validation", () => {
    it("should validate blueprint schema for all fields", async () => {
      const query = "Generate 10 math questions";
      const classification: Classification = {
        subject: "math",
        level: "intermediate",
        confidence: 0.9,
      };

      const blueprint = await generator.generateBlueprint(query, classification);

      // Should not throw - validates via Zod
      const validated = BlueprintSchema.parse(blueprint);

      expect(validated).toBeDefined();
      expect(validated.totalQuestions).toBe(blueprint.totalQuestions);
      expect(validated.topics).toEqual(blueprint.topics);
    }, 30000);

    it("should handle LLM errors gracefully with fallback", async () => {
      const query = "Generate invalid query to test fallback";
      const classification: Classification = {
        subject: "general",
        level: "basic",
        confidence: 0.5,
      };

      // Even if LLM fails, should return valid blueprint via fallback
      const blueprint = await generator.generateBlueprint(query, classification);

      expect(blueprint).toBeDefined();
      expect(blueprint.totalQuestions).toBeGreaterThan(0);
      expect(blueprint.topics.length).toBeGreaterThan(0);

      console.log("\n✅ Fallback Blueprint Generated:");
      console.log(JSON.stringify(blueprint, null, 2));
    }, 30000);
  });

  describe("Model Selection Integration", () => {
    it("should select appropriate model based on difficulty level", async () => {
      const testCases: Array<{
        level: "basic" | "intermediate" | "advanced";
        expectedModelTier: string[];
      }> = [
        { level: "basic", expectedModelTier: ["haiku", "Haiku"] },
        { level: "intermediate", expectedModelTier: ["sonnet", "Sonnet"] },
        { level: "advanced", expectedModelTier: ["sonnet", "Sonnet"] },
      ];

      for (const testCase of testCases) {
        const query = `Generate 5 ${testCase.level} questions`;
        const classification: Classification = {
          subject: "math",
          level: testCase.level,
          confidence: 0.9,
        };

        const blueprint = await generator.generateBlueprint(query, classification);

        // Check that selected model name contains expected tier
        const modelNameLower = blueprint.selectedModel.name.toLowerCase();
        const hasTier = testCase.expectedModelTier.some((tier) =>
          modelNameLower.includes(tier.toLowerCase())
        );

        expect(hasTier).toBe(true);

        console.log(
          `\n✅ ${testCase.level} → ${blueprint.selectedModel.name} (${blueprint.selectedModel.modelId})`
        );
      }
    }, 90000); // 90 seconds for 3 LLM calls
  });

  describe("Prompt Content Verification", () => {
    it("should include all classification details in prompt", () => {
      const query = "Generate 15 reasoning questions";
      const classification: Classification = {
        subject: "reasoning",
        level: "intermediate",
        confidence: 0.9,
      };

      const prompt = buildBlueprintGenerationPrompt(query, classification);

      // Verify all classification fields are in prompt
      expect(prompt).toContain(classification.subject);
      expect(prompt).toContain(classification.level);
      expect(prompt).toContain(classification.confidence.toString());

      // Verify prompt includes expected output structure
      expect(prompt).toContain("totalQuestions");
      expect(prompt).toContain("numberOfBatches");
      expect(prompt).toContain("topics");
      expect(prompt).toContain("topicName");
      expect(prompt).toContain("description");
      expect(prompt).toContain("difficulty");
      expect(prompt).toContain("questionCount");
      expect(prompt).toContain("priority");
      expect(prompt).toContain("pipelineNodes");
      expect(prompt).toContain("generationStrategy");
      expect(prompt).toContain("batchSize");
      expect(prompt).toContain("concurrency");
      expect(prompt).toContain("retryLimit");

      console.log("\n✅ Prompt Template Structure:");
      console.log(prompt.substring(0, 500) + "...");
    });

    it("should include topic guidelines in prompt", () => {
      const query = "Generate math questions";
      const classification: Classification = {
        subject: "math",
        level: "basic",
        confidence: 0.9,
      };

      const prompt = buildBlueprintGenerationPrompt(query, classification);

      // Check for topic guidelines
      expect(prompt).toContain("arithmetic");
      expect(prompt).toContain("algebra");
      expect(prompt).toContain("geometry");
      expect(prompt).toContain("calculus");
    });
  });
});
