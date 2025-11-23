import { ChatBedrockConverse } from "@langchain/aws";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { z } from 'zod';
import {traceable} from 'langsmith/traceable';
import { Classification} from "../types/index.ts";
import { modelConfigService } from "../config/modelConfig.ts";
import { buildClassifierSystemPrompt, buildStandaloneRewritePrompt } from "../utils/promptTemplates.ts";
import pino from "pino";

// Define Zod schema for Classification with intent
const ClassificationSchema = z.object({
  subject: z.enum([
    'math',
    'science',
    'history',
    'literature',
    'general',
    'reasoning',
    'english_grammar',
    'general_knowledge',
    'current_affairs',
  ]).describe('The academic subject category of the query'),
  level: z.enum(['basic', 'intermediate', 'advanced'])
    .describe('The difficulty level of the query'),
  confidence: z.number()
    .min(0)
    .max(1)
    .describe('Confidence score between 0 and 1'),
  reasoning: z.string()
    .optional()
    .describe('Brief explanation of the classification decision'),
  intent: z.enum([
    'factual_retrieval',
    'step_by_step_explanation',
    'comparative_analysis',
    'problem_solving',
    'definition_lookup',
    'reasoning_puzzle',
    'creative_generation',
    'verification_check',
    'summarize',
    'change_tone',
    'proofread',
    'make_email_professional'
  ])
    .describe('The user intent - what type of response format is expected'),
  expectedFormat: z.string()
    .describe('Expected output format based on intent and subject')
});

export interface ClassificationWithIntent extends Classification {
  intent?: string;
  expectedFormat?: string;
}

/**
 * Enhanced Classifier with built-in Standalone Query Rewriting
 * STEP 1: Rewrite query to standalone (using same LLM)
 * STEP 2: Classify the standalone query
 */
export class Classifier {
  private subjects: string[] = [];
  private llm: ChatBedrockConverse | null = null;
  private classifyPromptTemplate: ChatPromptTemplate | null = null;
  private rewritePromptTemplate: ChatPromptTemplate | null = null;

  constructor(private logger: pino.Logger, useExternalLLM?: any) {
    this.subjects = modelConfigService.getAvailableSubjects();

    // Initialize classification prompt template
    this.classifyPromptTemplate = ChatPromptTemplate.fromMessages([
      ['system', buildClassifierSystemPrompt(this.subjects)],
      ['human', 'query: {query}']
    ]);

    // Initialize standalone rewrite prompt template
    this.rewritePromptTemplate = ChatPromptTemplate.fromMessages([
      ['system', buildStandaloneRewritePrompt()],
      ['human', 'Current Query: {query}\n\nConversation History (User queries only):\n{history}']
    ]);



    // Initialize Langchain ChatBedrockConverse if no external LLM provided
    if (!useExternalLLM) {
      this.initializeLLM();
    }
  }

  /**
   * Initialize ChatBedrockConverse with configuration from modelConfig
   */
  private initializeLLM(): void {
    try {
      const config = modelConfigService.getClassifierConfig();
      const modelRegistry = modelConfigService.getModelRegistryEntry(config.modelId);

      if (!modelRegistry) {
        this.logger.warn(
          { modelId: config.modelId },
          '[Classifier] Model registry entry not found, using heuristic fallback'
        );
        return;
      }

      // Use inference profile ARN if available, otherwise use bedrockId
      const modelId = modelRegistry.inferenceProfileArn || modelRegistry.bedrockId;
      this.llm = new ChatBedrockConverse({
        model: modelId,
        region: modelRegistry.region,
        temperature: config.temperature,
        maxTokens: config.maxTokens,
      });

      this.logger.info(
        { modelId, region: modelRegistry.region },
        '[Classifier] LangChain ChatBedrockConverse initialized (SINGLE MODEL for both rewrite + classify)'
      );
    } catch (error) {
      this.logger.error(
        { error },
        '[Classifier] Failed to initialize LLM, falling back to heuristic'
      );
      this.llm = null;
    }
  }

  /**
   * Main classify method with built-in standalone rewriting
   * STEP 1: Rewrite query to standalone (optional, if history provided)
   * STEP 2: Classify the (rewritten or original) query
   */
  async classify(query: string, conversationHistory?: string): Promise<ClassificationWithIntent> {
    try {
      this.logger.debug({
        hasLLM: !!this.llm,
        llmType: this.llm?.constructor?.name,
        query: query.substring(0, 60),
        hasHistory: !!conversationHistory
      }, '[Classifier] Starting classification pipeline');

      // STEP 1: Rewrite to standalone if history provided
      let finalQuery = query;
      if (conversationHistory && conversationHistory.trim().length > 0) {
        try {
          finalQuery = await this.rewriteToStandalone(query, conversationHistory);
          this.logger.info({
            original: query.substring(0, 60),
            standalone: finalQuery.substring(0, 60)
          }, '[Classifier] Query rewritten to standalone');
        } catch (error) {
          this.logger.warn(
            { error, query: query.substring(0, 60) },
            '[Classifier] Standalone rewrite failed, using original query'
          );
          // Fallback to original query
          finalQuery = query;
        }
      }

      // STEP 2: Classify (using LLM or heuristic)
      if (this.llm) {
        this.logger.info('[Classifier] Using LLM-based classification (on standalone query)');
        return await this.llmClassify(finalQuery);
      }

      this.logger.info('[Classifier] Using heuristic classification (no LLM available)');
      return this.heuristicClassify(finalQuery);

    } catch (error) {
      this.logger.error({ error, query: query.substring(0, 60) }, '[Classifier] Classification failed, using heuristic');
      return this.heuristicClassify(query);
    }
  }

  /**
   * Standalone query rewriter 
   */
  private rewriteToStandalone = traceable(
  async (query: string, conversationHistory: string): Promise<string> => {
    if (!this.llm) {
      throw new Error('LLM not initialized for standalone rewrite');
    }

    if (!this.rewritePromptTemplate) {
      throw new Error('Rewrite prompt template not initialized');
    }

    // Skip rewriting for very long queries (likely already standalone)
    if (query.length > 200) {
      this.logger.debug('[Classifier] Query too long (>200 chars), likely standalone - skipping rewrite');
      return query;
    }

    try {
      const chain = this.rewritePromptTemplate.pipe(this.llm);
      const response = await chain.invoke({
        query,
        history: conversationHistory
      });

      const content = typeof response.content === 'string'
        ? response.content
        : JSON.stringify(response.content);

      // Extract the rewritten query (clean up the response)
      let rewritten = content
        .replace(/^(Rewritten Query:|Standalone Question:|Question:)/i, '')
        .trim()
        .replace(/^["']|["']$/g, ''); // Remove quotes if present

      // Validate rewritten query
      if (rewritten.length < 5) {
        this.logger.warn('[Classifier] Rewrite produced invalid result, using original');
        return query;
      }

      return rewritten;
    } catch (error) {
      this.logger.error({ error }, '[Classifier] Standalone rewrite failed');
      throw error;
    }
  },
  { name: 'StandaloneRewriter', run_type: 'chain' }
);

  /**
   * LLM-based classification using LangChain with JSON mode
   */
  private llmClassify = traceable( async (query: string): Promise<ClassificationWithIntent> => {
    if (!this.llm) {
      throw new Error('LLM not initialized');
    }

    if (!this.classifyPromptTemplate) {
      throw new Error('Classification prompt template not initialized');
    }

    let result: any;
    try {
      const chain = this.classifyPromptTemplate.pipe(this.llm);
      const response = await chain.invoke({ query });

      const content = typeof response.content === 'string'
        ? response.content
        : JSON.stringify(response.content);

      try {
        result = JSON.parse(content);
      } catch (parseError) {
        const jsonMatch = content.match(/```json\s*([\s\S]*?)\s*```/) ||
          content.match(/```\s*([\s\S]*?)\s*```/) ||
          content.match(/\{[\s\S]*\}/);

        if (jsonMatch) {
          const jsonStr = jsonMatch[1] || jsonMatch[0];
          result = JSON.parse(jsonStr);
        } else {
          throw new Error('No valid JSON found in response');
        }
      }

      const validated = ClassificationSchema.parse(result);

      this.logger.debug({
        subject: validated.subject,
        level: validated.level,
        confidence: validated.confidence,
        intent: validated.intent,
        expectedFormat: validated.expectedFormat
      }, '[Classifier] LLM classification successful');

      return {
        subject: validated.subject,
        level: validated.level as 'basic' | 'intermediate' | 'advanced',
        confidence: validated.confidence,
        intent: validated.intent,
        expectedFormat: validated.expectedFormat
      };

    } catch (error: any) {
      this.logger.error(
        { error: error.message, query: query.substring(0, 60) },
        '[Classifier] LLM classification failed'
      );

      // Try to salvage if it's just a subject mapping issue
      if (error.name === 'ZodError' && result) {
        const config = modelConfigService.getClassificationConfig();
        if (result.subject && config.subjectMapping[result.subject as keyof typeof config.subjectMapping]) {
          this.logger.info({
            original: result.subject,
            mapped: config.subjectMapping[result.subject as keyof typeof config.subjectMapping]
          }, '[Classifier] Mapping non-standard subject');

          result.subject = config.subjectMapping[result.subject as keyof typeof config.subjectMapping];
          try {
            const validated = ClassificationSchema.parse(result);
            return {
              subject: validated.subject,
              level: validated.level as 'basic' | 'intermediate' | 'advanced',
              confidence: validated.confidence,
              intent: validated.intent,
              expectedFormat: validated.expectedFormat
            };
          } catch (retryError) {
            this.logger.error({ error: retryError }, '[Classifier] Retry after mapping failed');
          }
        }
      }

      throw error;
      }
    }, 
    { name: 'LLMClassifier', run_type: 'chain'}
  )

  /**
   * Helper method to count keyword matches
   */
  private countMatches(text: string, keywords: string[]): number {
    return keywords.filter(keyword => text.includes(keyword)).length;
  }

  /**
   * Helper method to detect user intent from query
   */
  private detectIntent(query: string): { intent: string; expectedFormat: string } {
    const q = query.toLowerCase();
    const config = modelConfigService.getClassificationConfig();

    // Count matches for each intent using patterns from config
    const intentScores: Record<string, number> = {};
    for (const [intent, patterns] of Object.entries(config.intentPatterns)) {
      intentScores[intent] = this.countMatches(q, patterns);
    }

    // Find intent with highest score
    let maxIntent = 'factual_retrieval';
    let maxScore = intentScores.factual_retrieval || 0;

    for (const [intent, score] of Object.entries(intentScores)) {
      if (score > maxScore) {
        maxIntent = intent;
        maxScore = score;
      }
    }

    const expectedFormat = config.expectedFormats[maxIntent as keyof typeof config.expectedFormats]
      || config.expectedFormats.factual_retrieval;

    return { intent: maxIntent, expectedFormat };
  }

  /**
   * Enhanced heuristic-based classification fallback
   */
  private heuristicClassify(query: string): ClassificationWithIntent {
    const q = query.toLowerCase();
    const config = modelConfigService.getClassificationConfig();

    let subject: string = 'general';
    let confidence: number = 0.5;

    // Score each subject based on keyword matches using config
    const subjectScores: { [key: string]: number } = {};
    let maxScore = 0;
    let maxSubject = 'general';

    for (const [subj, data] of Object.entries(config.subjectKeywords)) {
      const keywordMatches = this.countMatches(q, data.keywords);
      const symbolMatches = this.countMatches(q, data.symbols);
      const score = keywordMatches * 2 + symbolMatches * 3;

      subjectScores[subj] = score;
      if (score > maxScore) {
        maxScore = score;
        maxSubject = subj;
      }
    }

    // If we have matches, use the detected subject
    if (maxScore > 0) {
      subject = maxSubject;
      confidence = Math.min(0.85, 0.5 + (maxScore * 0.05));
    }

    // Level detection
    let level: 'basic' | 'intermediate' | 'advanced' = 'basic';
    const advancedMatches = this.countMatches(q, config.levelIndicators.advanced);
    const intermediateMatches = this.countMatches(q, config.levelIndicators.intermediate);
    const basicMatches = this.countMatches(q, config.levelIndicators.basic);

    // Complexity analysis
    const wordCount = q.split(/\s+/).length;
    const sentenceCount = q.split(/[.!?]+/).filter(s => s.trim().length > 0).length;
    const hasQuestionMarks = (q.match(/\?/g) || []).length;
    const hasMultipleTopics = this.detectMultipleTopics(subjectScores);
    const avgWordsPerSentence = wordCount / Math.max(sentenceCount, 1);

    const isConfusing = wordCount > 80 || hasQuestionMarks > 2 || hasMultipleTopics || avgWordsPerSentence > 30;

    if (advancedMatches > 0) {
      level = 'advanced';
      confidence = Math.min(0.9, confidence + 0.1);
    } else if (isConfusing) {
      level = 'advanced';
      confidence = Math.max(0.3, confidence - 0.2);
    } else if (intermediateMatches > 0) {
      level = 'intermediate';
      confidence = Math.min(0.8, confidence + 0.05);
    } else if (basicMatches > 0) {
      level = 'basic';
      confidence = Math.min(0.75, confidence + 0.05);
    } else {
      if (wordCount > 40) {
        level = 'advanced';
      } else if (wordCount > 20 || hasQuestionMarks > 1) {
        level = 'intermediate';
      }
    }

    // Detect user intent
    const { intent, expectedFormat } = this.detectIntent(query);

    this.logger.info({
      subject, level, confidence,
      query: query.substring(0, 80),
      intent, expectedFormat
    }, '[Classifier] Heuristic classification complete');

    return {
      subject,
      level,
      confidence,
      intent,
      expectedFormat
    };
  }

  /**
   * Detect if question mixes multiple unrelated topics
   */
  private detectMultipleTopics(subjectScores: { [key: string]: number }): boolean {
    const significantScores = Object.values(subjectScores).filter(score => score > 3);
    return significantScores.length >= 3;
  }
}

export function createClassifier(logger: pino.Logger): Classifier {
  return new Classifier(logger);
}