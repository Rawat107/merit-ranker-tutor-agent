import { ChatBedrockConverse } from "@langchain/aws";
import { ChatPromptTemplate } from "@langchain/core/prompts";
import { z } from 'zod';
import { Classification} from "../types/index.ts";
import { modelConfigService } from "../config/modelConfig.ts";
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
    'verification_check'
  ])
    .describe('The user intent - what type of response format is expected'),

  expectedFormat: z.string()
    .describe('Expected output format based on intent and subject')
});


export interface ClassificationWithIntent extends Classification {
  intent?: string;
  expectedFormat?: string;
}

// LangChain query classifier using ChatBedrockConverse
// with structured output and user intent detection
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
  async classify(query: string): Promise<ClassificationWithIntent> {
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
  private async llmClassify(query: string): Promise<ClassificationWithIntent> {
    if (!this.llm) {
      throw new Error('LLM not initialized');
    }

    if (!this.promptTemplate) {
      throw new Error('Prompt template not initialized');
    }

    let result: any; // Declare outside try block so it's accessible in catch

    try {
      const chain = this.promptTemplate.pipe(this.llm);
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

      this.logger.debug(
        { 
          subject: validated.subject, 
          level: validated.level, 
          confidence: validated.confidence,
          intent: validated.intent,
          expectedFormat: validated.expectedFormat
        },
        '[Classifier] LLM classification successful'
      );

      return {
        subject: validated.subject,
        level: validated.level as 'basic' | 'intermediate' | 'advanced',
        confidence: validated.confidence,
        intent: validated.intent,
        expectedFormat: validated.expectedFormat
      };
    } catch (error: any) {
      this.logger.error(
        { error: error.message, query },
        '[Classifier] LLM classification failed'
      );
      
      // Try to salvage the result if it's just a subject mapping issue
      if (error.name === 'ZodError' && result) {
        const subjectMapping: { [key: string]: string } = {
          'reasoning_puzzle': 'reasoning',
          'logic_puzzle': 'reasoning',
          'game_theory': 'reasoning',
          'english': 'english_grammar',
          'english_grammer': 'english_grammar',
          'language': 'english_grammar',
          'grammar': 'english_grammar'
        };
        
        if (result.subject && subjectMapping[result.subject]) {
          this.logger.info(
            { original: result.subject, mapped: subjectMapping[result.subject] },
            '[Classifier] Mapping non-standard subject'
          );
          result.subject = subjectMapping[result.subject];
          
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
  }

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

    // Factual retrieval patterns
    const factualPatterns = [
      'what is', 'who is', 'where is', 'when', 'how long', 'how many',
      'define', 'meaning', 'what are', 'list', 'name', 'identify',
      'what does', 'capital of', 'country', 'continent'
    ];

    // Step-by-step explanation patterns
    const stepsPatterns = [
      'how to', 'steps', 'process', 'procedure', 'explain how',
      'solve', 'calculate', 'find', 'derive', 'prove',
      'show me', 'work through', 'walk me'
    ];

    // Comparative analysis patterns
    const comparePatterns = [
      'compare', 'contrast', 'difference between', 'vs', 'versus',
      'similarities', 'differences', 'which is', 'better', 'advantage'
    ];

    // Problem solving patterns
    const problemPatterns = [
      'problem', 'puzzle', 'riddle', 'challenge', 'figure out',
      'resolve', 'answer', 'solution', 'help with'
    ];

    // Reasoning/logic patterns
    const reasoningPatterns = [
      'why', 'reason', 'logic', 'because', 'cause', 'effect',
      'what if', 'suppose', 'assume', 'imply', 'deduce'
    ];

    // Verification/check patterns
    const verificationPatterns = [
      'correct', 'right', 'wrong', 'check', 'verify', 'is this',
      'am i right', 'is this correct', 'validate'
    ];

    // Count matches for each intent
    const factualMatches = this.countMatches(q, factualPatterns);
    const stepsMatches = this.countMatches(q, stepsPatterns);
    const compareMatches = this.countMatches(q, comparePatterns);
    const problemMatches = this.countMatches(q, problemPatterns);
    const reasoningMatches = this.countMatches(q, reasoningPatterns);
    const verificationMatches = this.countMatches(q, verificationPatterns);

    // Determine primary intent
    let intent = 'factual_retrieval'; // default
    let maxMatches = factualMatches;
    let expectedFormat = 'Direct answer with bullet points or concise explanation';

    if (stepsMatches > maxMatches) {
      intent = 'step_by_step_explanation';
      expectedFormat = 'Numbered steps with explanations and final answer (LaTeX for math)';
      maxMatches = stepsMatches;
    }
    if (compareMatches > maxMatches) {
      intent = 'comparative_analysis';
      expectedFormat = 'Markdown table or structured comparison with key differences';
      maxMatches = compareMatches;
    }
    if (problemMatches > maxMatches) {
      intent = 'problem_solving';
      expectedFormat = 'Problem breakdown → approach → detailed solution → verification';
      maxMatches = problemMatches;
    }
    if (reasoningMatches > maxMatches) {
      intent = 'reasoning_puzzle';
      expectedFormat = 'Logical reasoning steps with clear conclusions';
      maxMatches = reasoningMatches;
    }
    if (verificationMatches > maxMatches) {
      intent = 'verification_check';
      expectedFormat = 'Yes/No answer with explanation and corrections if needed';
      maxMatches = verificationMatches;
    }

    return { intent, expectedFormat };
  }

  /**
   * Enhanced heuristic-based classification fallback
   */
  private heuristicClassify(query: string): ClassificationWithIntent {
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
          'if then', 'therefore', 'because', 'assume', 'suppose', 'given that',
          'strategy', 'game', 'winning', 'optimal', 'player', 'move', 'turn',
          'alternately', 'coins', 'stones', 'pile', 'heavier', 'lighter', 'identical',
          'nim', 'mastermind', 'cryptarithmetic', 'weighing'
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
    
    const advancedIndicators = [
      'prove', 'proof', 'theorem', 'derivation', 'derive', 'rigorous',
      'complex analysis', 'advanced', 'research', 'dissertation',
      'sophisticated', 'intricate', 'elaborate', 'comprehensive study'
    ];
    
    const intermediateIndicators = [
      'compare', 'contrast', 'explain', 'analyze', 'describe', 'discuss',
      'why', 'how does', 'what is the difference', 'steps', 'process',
      'relationship', 'connection', 'cause', 'effect', 'evaluate',
      'examine', 'interpret', 'illustrate', 'demonstrate'
    ];
    
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
      const wordCount = q.split(/\s+/).length;
      const hasQuestionMarks = (q.match(/\?/g) || []).length;
      
      if (wordCount > 20 || hasQuestionMarks > 1) {
        level = 'intermediate';
      } else {
        level = 'basic';
      }
    }

    // Detect user intent
    const { intent, expectedFormat } = this.detectIntent(query);

    this.logger.info(
      { subject, level, confidence, query, intent, expectedFormat },
      '[Classifier] Heuristic classification complete'
    );

    return {
      subject,
      level,
      confidence,
      intent,
      expectedFormat
    };
  }

  private getSystemPrompt(): string {
    return `You are an expert query classifier for an AI tutoring system.

Your task is to classify student queries into:
1. Subject: ${this.subjects.join(', ')}
2. Level: basic, intermediate, advanced
3. Intent: what type of response format the user expects
4. Expected Format: how the answer should be structured

**IMPORTANT: Subject must be EXACTLY one of these:** ${this.subjects.join(', ')}
Do NOT use variations like "reasoning_puzzle" (use "reasoning"), "game_theory" (use "reasoning"), "english" or "english_grammer" (use "english_grammar").

Subject Guidelines:
- **math**: arithmetic, algebra, calculus, geometry, equations, integrals
- **science**: physics, chemistry, biology, scientific concepts
- **reasoning**: logic puzzles, game theory, strategy games, deduction problems, riddles, nim games, weighing puzzles
- **english_grammar**: grammar, spelling, punctuation, sentence structure
- **history**: historical events, dates, civilizations
- **general**: questions that don't fit specific subjects

Classification Guidelines:
- **basic**: Simple, straightforward questions or basic concepts (e.g., "What is photosynthesis?", "Define gravity")
- **intermediate**: Questions requiring explanation, comparison, or multi-step reasoning (e.g., "Compare mitosis and meiosis", "Explain why...")
- **advanced**: Complex problems requiring proofs, derivations, or expert-level analysis (e.g., "Prove the theorem", "Derive the equation")

User Intent Types:
- **factual_retrieval**: Direct factual questions (What is? Who is? When? Where?)
- **step_by_step_explanation**: How-to questions, tutorials, derivations (How to? Solve? Steps? Process?)
- **comparative_analysis**: Compare/contrast questions (Compare? Difference between? vs?)
- **problem_solving**: Puzzle/challenge questions (Problem? Solve? Figure out?)
- **reasoning_puzzle**: Logic/reasoning questions (Why? Reason? Logic? Strategy games?)
- **verification_check**: Check/validate questions (Is this correct? Verify?)

Expected Format Examples:
- Factual: "Direct answer with bullet points"
- Steps: "Numbered steps 1,2,3... with LaTeX for math"
- Compare: "Markdown table for comparison"
- Problem: "Problem breakdown → approach → solution"
- Reasoning: "Logical chain with conclusions and winning strategy"
- Verify: "Yes/No with explanation"

Provide a confidence score (0.0 to 1.0) indicating your certainty.

You MUST respond with ONLY a valid JSON object. Do not include any text before or after the JSON.
The JSON must have these fields: subject, level, confidence, reasoning (optional), intent, expectedFormat.

Example response: {{"subject": "math", "level": "intermediate", "confidence": 0.9, "intent": "step_by_step_explanation", "expectedFormat": "Numbered steps with LaTeX formulas and verification", "reasoning": "Trigonometry problem asking for solution steps"}}`;
  }
}