import { ChatBedrockConverse } from "@langchain/aws";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { z } from 'zod';
import { Classification} from "../types/index.ts";
import { modelConfigService } from "../config/modelConfig.ts";
import pino from "pino";


// Define Zod schema for Classification
// ensure it matches the structure of the Classification interface
const ClassificationSchema = z.object({
  subject: z.enum([ 
    'math',
    'science',
    'history',
    'literature',
    'general',
    'reasoning',
    'english_grammer',
    'general_knowledge',
    'current_affairs',
  ]).describe('The academic subject category of the query'),

  level: z.enum(['basic', 'intermediate', 'advanced'])
    .describe('The difficulty level of the query'),

  confidence: z.number()
    .min(0)
    .max(1)
    .describe('Confidence score between 0 and 1'),
  
  resoning: z.string()
    .optional()
    .describe('Breif explanation of the classification decision')
});

// Langachain query classifier using CHatBedrockConverse
// with structured output

export class Classifier{
  private subjects: string[] = [];
  private llm:  ChatBedrockConverse | null = null;
  private promptTemplate: ChatPromptTemplate | null = null;

  constructor(private logger: pino.Logger, useExternalLLM?: any){
    this.subjects = modelConfigService.getAvailableSubjects();

    //initialize Langchain prompt template
    this.promptTemplate = ChatPromptTemplate.fromMessages([
      ['system', this.getSystemPrompt()],
      ['human', 'query: {query}'] 
    ]);

    //initialize Langchain ChatBedrockConverse if no LLM
    if(!useExternalLLM){
      this.initializeLLM();
    }
  }

   // Initialize ChatBedrockConverse with configuration from modelConfig
  
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
        // Credentials are auto-loaded from AWS environment
      });

      this.logger.info(
        { modelId, region: modelRegistry.region },
        '[Classifier] LangChain ChatBedrockConverse initialized'
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
   * Classify query using LLM or heuristic fallback
   */
  async classify(query: string): Promise<Classification> {
    try {
      this.logger.debug({ 
        hasLLM: !!this.llm, 
        llmType: this.llm?.constructor?.name,
        query 
      }, '[Classifier] Starting classification');
      
      if (this.llm) {
        this.logger.info('[Classifier] Using LLM-based classification');
        return await this.llmClassify(query);
      }
      this.logger.info('[Classifier] Using heuristic classification (no LLM available)');
      return this.heuristicClassify(query);
    } catch (error) {
      this.logger.error({ error, query }, '[Classifier] LLM classification failed, using heuristic');
      return this.heuristicClassify(query);
    }
  }

  /**
   * LLM-based classification using LangChain with JSON mode
   */
  private async llmClassify(query: string): Promise<Classification> {
    if (!this.llm) {
      throw new Error('LLM not initialized');
    }

    if (!this.promptTemplate) {
      throw new Error('Prompt template not initialized');
    }

    try {
      // Use standard invoke with JSON parsing instead of withStructuredOutput
      // This is more reliable with Bedrock models that don't fully support tool calling
      const chain = this.promptTemplate.pipe(this.llm);
      
      // Invoke the chain
      const response = await chain.invoke({ query });

      // Parse the JSON response
      let result: any;
      const content = typeof response.content === 'string' 
        ? response.content 
        : JSON.stringify(response.content);
      
      try {
        // Try to parse the entire content as JSON
        result = JSON.parse(content);
      } catch (parseError) {
        // If direct parsing fails, try to extract JSON from markdown code blocks
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

      // Validate with Zod schema
      const validated = ClassificationSchema.parse(result);

      this.logger.debug(
        { subject: validated.subject, level: validated.level, confidence: validated.confidence },
        '[Classifier] LLM classification successful'
      );

      // Map to Classification interface
      return {
        subject: validated.subject,
        level: validated.level as 'basic' | 'intermediate' | 'advanced',
        confidence: validated.confidence,
      };
    } catch (error: any) {
      this.logger.error(
        { error: error.message, query },
        '[Classifier] LLM classification failed'
      );
      throw error;
    }
  }


  /**
   * Helper method to count keyword matches
   */
  private countMatches(text: string, keywords: string[]): number {
    return keywords.filter(keyword => text.includes(keyword)).length;
  }

  /**
   * Enhanced heuristic-based classification fallback
   */
  private heuristicClassify(query: string): Classification {
    const q = query.toLowerCase();
    let subject: string = 'general';
    let confidence: number = 0.5;

    // Enhanced keyword sets for each subject
    const subjectKeywords = {
      math: {
        keywords: [
          'math', 'equation', 'integral', 'derivative', 'algebra', 'calculus', 
          'geometry', 'trigonometry', 'sine', 'cosine', 'tangent', 'sin', 'cos', 'tan',
          'arithmetic', 'number', 'calculate', 'solve', 'sum', 'difference', 'product',
          'quotient', 'fraction', 'decimal', 'percentage', 'ratio', 'proportion',
          'angle', 'triangle', 'circle', 'square', 'polynomial', 'logarithm',
          'exponent', 'power', 'root', 'sqrt', 'function', 'graph', 'plot',
          'matrix', 'vector', 'statistics', 'probability', 'mean', 'median'
        ],
        symbols: ['+', '-', '*', '/', '=', '√', 'π', '∫', '∑', '^', '²', '³']
      },
      science: {
        keywords: [
          'physics', 'chemistry', 'biology', 'atom', 'molecule', 'cell', 'organism',
          'photosynthesis', 'respiration', 'mitosis', 'meiosis', 'dna', 'rna',
          'protein', 'enzyme', 'reaction', 'element', 'compound', 'periodic table',
          'energy', 'force', 'motion', 'velocity', 'acceleration', 'gravity',
          'electricity', 'magnetism', 'light', 'sound', 'wave', 'quantum',
          'ecosystem', 'evolution', 'species', 'genetic', 'nucleus', 'electron',
          'proton', 'neutron', 'chemical', 'organic', 'inorganic', 'catalyst'
        ],
        symbols: ['CO2', 'H2O', 'O2', 'N2', 'H+', 'pH']
      },
      english_grammar: {
        keywords: [
          'grammar', 'sentence', 'verb', 'noun', 'adjective', 'adverb', 'pronoun',
          'punctuation', 'comma', 'period', 'semicolon', 'apostrophe', 'quotation',
          'subject', 'predicate', 'clause', 'phrase', 'tense', 'past', 'present',
          'future', 'plural', 'singular', 'possessive', 'article', 'conjunction',
          'preposition', 'interjection', 'syntax', 'spelling', 'capitalization'
        ],
        symbols: ['.', ',', ';', ':', '!', '?', '"', "'"]
      },
      history: {
        keywords: [
          'history', 'war', 'ancient', 'civilization', 'empire', 'dynasty', 'king',
          'queen', 'battle', 'revolution', 'independence', 'colonial', 'century',
          'bc', 'ad', 'medieval', 'renaissance', 'industrial', 'world war',
          'treaty', 'constitution', 'democracy', 'monarchy', 'republic', 'slavery',
          'conquest', 'explorer', 'discovery', 'invention', 'historical',
          'emperor', 'pharaoh', 'sultan', 'kingdom', 'reign', 'throne',
          'mughal', 'ottoman', 'roman', 'persian', 'greek', 'egyptian',
          'historical event', 'timeline', 'era', 'period', 'age'
        ],
        symbols: []
      },
      literature: {
        keywords: [
          'literature', 'novel', 'poem', 'poetry', 'author', 'writer', 'book',
          'shakespeare', 'dickens', 'twain', 'hemingway', 'fiction', 'nonfiction',
          'character', 'plot', 'theme', 'metaphor', 'simile', 'irony', 'symbolism',
          'genre', 'narrative', 'prose', 'verse', 'stanza', 'rhyme', 'literary',
          'protagonist', 'antagonist', 'tragedy', 'comedy', 'drama'
        ],
        symbols: []
      },
      reasoning: {
        keywords: [
          'logic', 'reason', 'deduce', 'infer', 'puzzle', 'riddle', 'paradox',
          'syllogism', 'premise', 'conclusion', 'argument', 'fallacy', 'valid',
          'invalid', 'consistent', 'inconsistent', 'contradict', 'imply',
          'if then', 'therefore', 'because', 'assume', 'suppose', 'given that'
        ],
        symbols: ['→', '∴', '∵', '∀', '∃']
      },
      current_affairs: {
        keywords: [
          'news', 'recent', 'current', 'today', 'latest', 'yesterday', '2024', '2025',
          'political', 'election', 'government', 'policy', 'economic', 'market',
          'covid', 'pandemic', 'climate', 'technology', 'ai', 'social media',
          'trending', 'viral', 'breaking', 'announcement'
        ],
        symbols: []
      },
      general_knowledge: {
        keywords: [
          'what is', 'who is', 'where is', 'when did', 'define', 'meaning',
          'capital', 'country', 'continent', 'planet', 'largest', 'smallest',
          'fastest', 'tallest', 'oldest', 'inventor', 'discovery', 'fact'
        ],
        symbols: []
      }
    };

    // Score each subject based on keyword matches
    const subjectScores: { [key: string]: number } = {};
    let maxScore = 0;
    let maxSubject = 'general';

    for (const [subj, data] of Object.entries(subjectKeywords)) {
      const keywordMatches = this.countMatches(q, data.keywords);
      const symbolMatches = this.countMatches(q, data.symbols);
      const score = keywordMatches * 2 + symbolMatches * 3; // Symbols weighted more
      
      subjectScores[subj] = score;
      
      if (score > maxScore) {
        maxScore = score;
        maxSubject = subj;
      }
    }

    // If we have matches, use the detected subject
    if (maxScore > 0) {
      subject = maxSubject;
      // Adjust confidence based on match strength
      confidence = Math.min(0.85, 0.5 + (maxScore * 0.05));
    }

    // Level detection with more nuanced rules
    let level: 'basic' | 'intermediate' | 'advanced' = 'basic';
    
    // Advanced indicators
    const advancedIndicators = [
      'prove', 'proof', 'theorem', 'derivation', 'derive', 'rigorous',
      'complex analysis', 'advanced', 'research', 'dissertation',
      'sophisticated', 'intricate', 'elaborate', 'comprehensive study'
    ];
    
    // Intermediate indicators
    const intermediateIndicators = [
      'compare', 'contrast', 'explain', 'analyze', 'describe', 'discuss',
      'why', 'how does', 'what is the difference', 'steps', 'process',
      'relationship', 'connection', 'cause', 'effect', 'evaluate',
      'examine', 'interpret', 'illustrate', 'demonstrate'
    ];
    
    // Basic indicators
    const basicIndicators = [
      'what is', 'define', 'who is', 'where is', 'when', 'list',
      'name', 'identify', 'state', 'what are', 'simple', 'basic'
    ];

    const advancedMatches = this.countMatches(q, advancedIndicators);
    const intermediateMatches = this.countMatches(q, intermediateIndicators);
    const basicMatches = this.countMatches(q, basicIndicators);

    if (advancedMatches > 0) {
      level = 'advanced';
      confidence = Math.min(0.9, confidence + 0.1);
    } else if (intermediateMatches > 0) {
      level = 'intermediate';
      confidence = Math.min(0.8, confidence + 0.05);
    } else if (basicMatches > 0) {
      level = 'basic';
      confidence = Math.min(0.75, confidence + 0.05);
    } else {
      // Default level based on query complexity
      const wordCount = q.split(/\s+/).length;
      const hasQuestionMarks = (q.match(/\?/g) || []).length;
      
      if (wordCount > 20 || hasQuestionMarks > 1) {
        level = 'intermediate';
      } else {
        level = 'basic';
      }
    }

    this.logger.info(
      { subject, level, confidence, query, subjectScores },
      '[Classifier] Heuristic classification complete'
    );

    return {
      subject,
      level,
      confidence,
    };
  }

  

  private getSystemPrompt(): string {
    return `You are an expert query classifier for an AI tutoring system.

Your task is to classify student queries into:
1. Subject: ${this.subjects.join(', ')}
2. Level: basic, intermediate, advanced

Classification Guidelines:
- **basic**: Simple, straightforward questions or basic concepts (e.g., "What is photosynthesis?", "Define gravity")
- **intermediate**: Questions requiring explanation, comparison, or multi-step reasoning (e.g., "Compare mitosis and meiosis", "Explain why...")
- **advanced**: Complex problems requiring proofs, derivations, or expert-level analysis (e.g., "Prove the theorem", "Derive the equation")

Subject Detection Guidelines:
- **math**: Arithmetic, algebra, calculus, geometry, trigonometry, equations, mathematical operations
- **science**: Physics, chemistry, biology, scientific processes, experiments, natural phenomena
- **history**: Historical events, dates, civilizations, wars, historical figures, empires, dynasties
- **literature**: Books, poems, authors, literary analysis, writing styles
- **reasoning**: Logic puzzles, deduction, inference, critical thinking
- **english_grammar**: Grammar rules, sentence structure, parts of speech, punctuation
- **general_knowledge**: Facts, definitions, general information
- **current_affairs**: News, recent events, contemporary topics
- **general**: Queries that don't fit other categories

Provide a confidence score (0.0 to 1.0) indicating your certainty.

You MUST respond with ONLY a valid JSON object. Do not include any text before or after the JSON.
The JSON must have these fields: subject, level, confidence, reasoning (optional).

Example response: {{"subject": "math", "level": "intermediate", "confidence": 0.9, "reasoning": "Trigonometry question"}}`;
  }
  
}
