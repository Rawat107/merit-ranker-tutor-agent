import { TutorConfig, ModelConfig, Classification, ModelRegistryEntry } from '../types/index.js';
import { config } from 'dotenv';
config();

// Enhanced model configuration based on your provided config
export const tutorConfigData: TutorConfig = {
  "defaultSubscription": "free",
  "plans": {
    "free": {
      "description": "Lowest cost — use cheaper models, small context, prioritize latency and low cost.",
      "preferredModels": ["gpt-3.5-turbo", "amazon.nova-micro-v1:0"],
      "maxTokensMultiplier": 0.8
    },
    "basic": {
      "description": "Balanced cost/quality — use mid-tier models like claude-3.5 or small Bedrock models.",
      "preferredModels": ["anthropic.claude-3-sonnet-20240229-v1:0", "anthropic.claude-sonnet-4-20250514-v1:0"],
      "maxTokensMultiplier": 1.0
    },
    "premium": {
      "description": "Higher quality — use stronger models for final verification and complex reasoning.",
      "preferredModels": ["anthropic.claude-3-5-sonnet-20241022-v2:0", "anthropic.claude-sonnet-4-20250514-v1:0"],
      "maxTokensMultiplier": 1.3
    }
  },
  "modelRegistry": {
    "gpt-3.5-turbo": { "bedrockId": "gpt-3.5-turbo", "invokeId": "gpt-3.5-turbo", "inferenceProfileArn": null, "region": "ap-south-1" },
    "amazon.nova-micro-v1:0": { "bedrockId": "amazon.nova-micro-v1:0", "invokeId": "amazon.nova-micro-v1:0", "inferenceProfileArn": "arn:aws:bedrock:ap-south-1:558069890997:inference-profile/apac.amazon.nova-micro-v1:0", "region": "ap-south-1" },
    "amazon.rerank-v1:0": { "bedrockId": "cohere.rerank-v3-5:0", "invokeId": "cohere.rerank-v3-5:0", "inferenceProfileArn": "arn:aws:bedrock:region::foundation-model/amazon.rerank-v1:0", "region": "ap-south-1" },
    "anthropic.claude-3-haiku-20240307-v1:0": { "bedrockId": "anthropic.claude-3-haiku-20240307-v1:0", "invokeId": "anthropic.claude-3-haiku-20240307-v1:0", "inferenceProfileArn": "arn:aws:bedrock:ap-south-1:558069890997:inference-profile/apac.anthropic.claude-3-haiku-20240307-v1:0", "region": "ap-south-1" },
    "anthropic.claude-3-sonnet-20240229-v1:0": { "bedrockId": "anthropic.claude-3-sonnet-20240229-v1:0", "invokeId": "anthropic.claude-3-sonnet-20240229-v1:0", "inferenceProfileArn": "arn:aws:bedrock:ap-south-1:558069890997:inference-profile/apac.anthropic.claude-3-sonnet-20240229-v1:0", "region": "ap-south-1" },
    "anthropic.claude-3-5-sonnet-20241022-v2:0": { "bedrockId": "anthropic.claude-3-5-sonnet-20241022-v2:0", "invokeId": "anthropic.claude-3-5-sonnet-20241022-v2:0", "inferenceProfileArn": "arn:aws:bedrock:ap-south-1:558069890997:inference-profile/apac.anthropic.claude-3-5-sonnet-20241022-v2:0", "region": "ap-south-1" },
    "anthropic.claude-sonnet-4-20250514-v1:0": { "bedrockId": "anthropic.claude-sonnet-4-20250514-v1:0", "invokeId": "anthropic.claude-sonnet-4-20250514-v1:0", "inferenceProfileArn": "arn:aws:bedrock:ap-south-1:558069890997:inference-profile/apac.anthropic.claude-sonnet-4-20250514-v1:0", "region": "ap-south-1" }
  },
  "classifier": {
    "modelId": "amazon.nova-micro-v1:0",
    "systemPrompt": "You are a lightweight classifier that identifies subject and difficulty level (basic/intermediate/advanced). Respond with JSON { \"subject\": \"<subject>\", \"level\": \"<basic|intermediate|advanced>\" }",
    "maxTokens": 200,
    "temperature": 0.0,
    "outputFormat": "json"
  },
  "reranker": {
    "modelId": "amazon.rerank-v1:0",
    "systemPrompt": "You are a reranker. Given a user query and a passage, score relevance between 0 and 1 and return JSON { \"score\": <0..1>, \"reason\": \"...\" }",
    "maxTokens": 200,
    "temperature": 0.0,
    "outputFormat": "json"
  },
  "subjects": {
    "general": {
      "basic": {
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        "maxTokens": 800,
        "temperature": 0.2,
        "systemPrompt": "You are a helpful tutor. Keep answers concise, accurate, and provide sources when available.",
        "outputFormat": "Answer with short explanation and final answer."
      },
      "intermediate": {
        "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
        "maxTokens": 1200,
        "temperature": 0.1,
        "systemPrompt": "You are an accurate tutor. Provide step-by-step answers and cite sources from provided context.",
        "outputFormat": "Detailed answer with citations and short summary."
      },
      "advanced": {
        "modelId": "anthropic.claude-sonnet-4-20250514-v1:0",
        "maxTokens": 1600,
        "temperature": 0.0,
        "systemPrompt": "You are an expert tutor. Provide authoritative, citation-backed answers, check reasoning, and flag uncertainty.",
        "outputFormat": "Authoritative answer with sources, reasoning steps, and confidence score."
      }
    },
    "math": {
      "basic": {
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        "maxTokens": 1000,
        "temperature": 0.2,
        "systemPrompt": "You are a math tutor. Provide step-by-step solutions and final answer. Use tools for computation if needed.",
        "outputFormat": "Steps + final solution"
      },
      "intermediate": {
        "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "maxTokens": 1200,
        "temperature": 0.1,
        "systemPrompt": "You are a math tutor. Provide rigorous steps and verify numeric results with computation.",
        "outputFormat": "Detailed steps + verification"
      },
      "advanced": {
        "modelId": "anthropic.claude-sonnet-4-20250514-v1:0",
        "maxTokens": 2000,
        "temperature": 0.0,
        "systemPrompt": "You are an expert mathematician. Provide proofs, derivations, and verify via computation where applicable.",
        "outputFormat": "Proof/derivation + final answer"
      }
    },
    "reasoning": {
      "basic": {
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        "maxTokens": 900,
        "temperature": 0.2,
        "systemPrompt": "You are a reasoning assistant. Break down problems into small steps and validate each step.",
        "outputFormat": "Step-wise reasoning + conclusion"
      },
      "intermediate": {
        "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "maxTokens": 1300,
        "temperature": 0.1,
        "systemPrompt": "You are a reasoning specialist. Provide clear chain-of-thought-style steps but avoid exposing raw chain-of-thought in final output. Verify final claims.",
        "outputFormat": "Reasoned steps + verified conclusion"
      },
      "advanced": {
        "modelId": "anthropic.claude-sonnet-4-20250514-v1:0",
        "maxTokens": 1600,
        "temperature": 0.0,
        "systemPrompt": "You are an expert reasoner. Provide rigorous reasoning and indicate confidence and sources.",
        "outputFormat": "Rigorous reasoning + confidence"
      }
    },
    "english_grammar": {
      "basic": {
        "modelId": "anthropic.claude-3-haiku-20240307-v1:0",
        "maxTokens": 600,
        "temperature": 0.2,
        "systemPrompt": "You are an English grammar tutor. Provide corrections, brief rules, and examples.",
        "outputFormat": "Correction + brief explanation"
      },
      "intermediate": {
        "modelId": "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "maxTokens": 900,
        "temperature": 0.1,
        "systemPrompt": "You are an English editor and tutor. Provide corrected text, explanation, and usage notes.",
        "outputFormat": "Corrected text + explanation"
      },
      "advanced": {
        "modelId": "anthropic.claude-sonnet-4-20250514-v1:0",
        "maxTokens": 1200,
        "temperature": 0.0,
        "systemPrompt": "You are an expert English linguist. Provide authoritative corrections, alternatives, and style notes.",
        "outputFormat": "Corrected + stylistic suggestions"
      }
    },
    "science": {
      "basic": { "modelId": "anthropic.claude-3-haiku-20240307-v1:0", "maxTokens": 800, "temperature": 0.2, "systemPrompt": "You are a science tutor. Keep explanations simple and factual.", "outputFormat": "Answer + simple explanation" },
      "intermediate": { "modelId": "anthropic.claude-sonnet-4-20250514-v1:0", "maxTokens": 1200, "temperature": 0.1, "systemPrompt": "You are a science assistant. Provide sourced explanations and reasoning.", "outputFormat": "Answer + citations" },
      "advanced": { "modelId": "anthropic.claude-sonnet-4-20250514-v1:0", "maxTokens": 1600, "temperature": 0.0, "systemPrompt": "You are an expert scientist. Provide thorough explanations, citations, and confidence.", "outputFormat": "Detailed answer + citations" }
    },
    "history": {
      "basic": { "modelId": "anthropic.claude-3-haiku-20240307-v1:0", "maxTokens": 800, "temperature": 0.2, "systemPrompt": "You are a history tutor. Provide concise historical facts and timelines.", "outputFormat": "Facts + timeline" },
      "intermediate": { "modelId": "anthropic.claude-sonnet-4-20250514-v1:0", "maxTokens": 1200, "temperature": 0.1, "systemPrompt": "You are a historian. Provide sourced explanations and context.", "outputFormat": "Answer + citations" },
      "advanced": { "modelId": "anthropic.claude-sonnet-4-20250514-v1:0", "maxTokens": 1600, "temperature": 0.0, "systemPrompt": "You are an expert historian. Provide deep contextual analysis and sources.", "outputFormat": "Deep analysis + sources" }
    }
  }
};

export const appConfig = {
  port: parseInt(process.env.PORT || '3000'),
  logLevel: process.env.LOG_LEVEL || 'info',
  nodeEnv: process.env.NODE_ENV || 'development',
  
  // AWS Configuration
  aws: {
    region: process.env.AWS_REGION || 'ap-south-1',
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY,
  },
  
  // Model Configuration
  defaultModelId: process.env.BEDROCK_MODEL_ID || 'anthropic.claude-3-sonnet-20240229-v1:0',
  embeddingModelId: process.env.EMBED_MODEL_ID || 'amazon.titan-embed-text-v1',
  
  // Cache Configuration
  cache: {
    serveSimilarity: parseFloat(process.env.CACHE_SERVE_SIMILARITY || '0.94'),
    softSimilarity: parseFloat(process.env.CACHE_SOFT_SIMILARITY || '0.82'),
    ttl: parseInt(process.env.CACHE_TTL || '86400'),
  },
  
  // External Services
  upstash: {
    url: process.env.UPSTASH_URL,
    token: process.env.UPSTASH_TOKEN,
  },
  
  redis: {
    url: process.env.REDIS_URL || 'redis://localhost:6379',
    password: process.env.REDIS_PASSWORD,
  },
  
  serpapi: {
    key: process.env.SERPAPI_KEY,
  }
};

/**
 * Classification Configuration - Subject Keywords and Intent Patterns
 * Used by Classifier for heuristic fallback when LLM is unavailable
 */
export const classificationConfig = {
  /**
   * Subject keyword matching configuration
   */
  subjectKeywords: {
    math: {
      keywords: [
        'math', 'equation', 'integral', 'derivative', 'algebra', 'calculus',
        'geometry', 'trigonometry', 'sine', 'cosine', 'tangent', 'sin', 'cos', 'tan',
        'arithmetic', 'number', 'calculate', 'solve', 'sum', 'difference', 'product',
        'quotient', 'fraction', 'decimal', 'percentage', 'ratio', 'proportion',
        'angle', 'triangle', 'circle', 'square', 'polynomial', 'logarithm',
        'exponent', 'power', 'root', 'sqrt', 'function', 'graph', 'plot',
        'matrix', 'vector', 'statistics', 'probability', 'mean', 'median',
        'permutation', 'combination', 'factorial', 'random', 'expected',
        'variance', 'standard deviation', 'distribution', 'sample'
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
        'preposition', 'interjection', 'syntax', 'spelling', 'capitalization',
        'gerund', 'participle', 'infinitive', 'modal', 'passive', 'active'
      ],
      symbols: [] 
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
        'nim', 'mastermind', 'cryptarithmetic', 'weighing', 'probability',
        'random', 'chosen', 'flipped', 'conditional probability', 'bayes',
        'expected value', 'outcomes', 'sample space', 'event'
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
  },

  /**
   * Intent detection patterns
   */
  intentPatterns: {
    factual_retrieval: [
      'what is', 'who is', 'where is', 'when', 'how long', 'how many',
      'define', 'meaning', 'what are', 'list', 'name', 'identify',
      'what does', 'capital of', 'country', 'continent'
    ],
    step_by_step_explanation: [
      'how to', 'steps', 'process', 'procedure', 'explain how',
      'solve', 'calculate', 'find', 'derive', 'prove',
      'show me', 'work through', 'walk me'
    ],
    comparative_analysis: [
      'compare', 'contrast', 'difference between', 'vs', 'versus',
      'similarities', 'differences', 'which is', 'better', 'advantage'
    ],
    problem_solving: [
      'problem', 'puzzle', 'riddle', 'challenge', 'figure out',
      'resolve', 'answer', 'solution', 'help with'
    ],
    reasoning_puzzle: [
      'why', 'reason', 'logic', 'because', 'cause', 'effect',
      'what if', 'suppose', 'assume', 'imply', 'deduce'
    ],
    verification_check: [
      'correct', 'right', 'wrong', 'check', 'verify', 'is this',
      'am i right', 'is this correct', 'validate'
    ],
    summarize: [
      'summarize', 'summary', 'condense', 'shorten', 'brief', 'gist',
      'main points', 'overview', 'recap', 'abstract'
    ],
    change_tone: [
      'change tone', 'make it sound', 'tone of voice', 'formal', 'informal',
      'friendly', 'professional', 'casual', 'serious', 'humorous', 'optimistic',
      'pessimistic', 'assertive', 'passive', 'empathetic', 'direct', 'indirect'
    ],
    proofread: [
      'proofread', 'grammar check', 'correct grammar', 'typos', 'spelling',
      'punctuation', 'errors', 'review text', 'fix mistakes', 'edit'
    ],
    make_email_professional: [
      'make email professional', 'professionalize email', 'rewrite email formally',
      'improve email for work', 'business email', 'formal email'
    ]
  },

  /**
   * Level detection indicators
   */
  levelIndicators: {
    advanced: [
      // Formal/rigorous keywords
      'prove', 'proof', 'theorem', 'derivation', 'derive', 'rigorous',
      'formalize', 'formal', 'axiom', 'lemma', 'corollary',
      
      // Research/academic keywords
      'complex analysis', 'advanced', 'research', 'dissertation',
      'sophisticated', 'intricate', 'elaborate', 'comprehensive study',
      
      // Mathematical complexity
      'infinitely', 'infinite', 'convergence', 'divergence', 'asymptotic',
      'characterize', 'generalize', 'abstract', 'canonical',
      
      // Logic & reasoning
      'formalize', 'characterize', 'minimal', 'necessary and sufficient',
      'if and only if', 'iff', 'contrapositive', 'contradiction',
      
      // Complexity indicators
      'multiple constraints', 'optimize', 'minimize', 'maximize',
      'given arbitrary', 'for all', 'there exists', 'such that'
    ],
    intermediate: [
      'compare', 'contrast', 'explain', 'analyze', 'describe', 'discuss',
      'why', 'how does', 'what is the difference', 'steps', 'process',
      'relationship', 'connection', 'cause', 'effect', 'evaluate',
      'examine', 'interpret', 'illustrate', 'demonstrate', 'calculate',
      'solve', 'find', 'determine'
    ],
    basic: [
      'what is', 'define', 'who is', 'where is', 'when', 'list',
      'name', 'identify', 'state', 'what are', 'simple', 'basic',
      'tell me', 'give me'
    ]
  },

  /**
   * Subject mapping for non-standard LLM responses
   */
  subjectMapping: {
    'reasoning_puzzle': 'reasoning',
    'logic_puzzle': 'reasoning',
    'game_theory': 'reasoning',
    'english': 'english_grammar',
    'english_grammer': 'english_grammar',
    'language': 'english_grammar',
    'grammar': 'english_grammar'
  },

  /**
   * Expected format templates by intent
   */
  expectedFormats: {
    factual_retrieval: 'Direct answer with bullet points or concise explanation',
    step_by_step_explanation: 'Numbered steps with explanations and final answer (LaTeX for math)',
    comparative_analysis: 'Markdown table or structured comparison with key differences',
    problem_solving: 'Problem breakdown → approach → detailed solution → verification',
    reasoning_puzzle: 'Logical reasoning steps with clear conclusions',
    verification_check: 'Yes/No answer with explanation and corrections if needed',
    summarize: 'A concise summary of the provided text',
    change_tone: 'The rewritten text with the requested tone',
    proofread: 'The corrected text with grammar and spelling fixes',
    make_email_professional: 'A professionally rewritten version of the email'
  }
};

/**
 * Model Configuration Service - LangChain compatible
 */
export class ModelConfigService {
  private tutorConfig: TutorConfig;

  constructor() {
    this.tutorConfig = tutorConfigData;
  }

  /**
   * Get model configuration based on classification and subscription
   */
  getModelConfig(classification: Classification, userSubscription: string = 'free'): ModelConfig {
    const { subject, level } = classification;
    
    // Get subject-specific config, fallback to general
    const subjectConfig = this.tutorConfig.subjects[subject] || this.tutorConfig.subjects.general;
    
    // Get level-specific config, fallback to basic
    const levelConfig = subjectConfig[level] || subjectConfig.basic;
    
    // Apply subscription plan multiplier
    const plan = this.tutorConfig.plans[userSubscription] || this.tutorConfig.plans[this.tutorConfig.defaultSubscription];
    const adjustedMaxTokens = Math.floor(levelConfig.maxTokens * plan.maxTokensMultiplier);
    
    return {
      ...levelConfig,
      maxTokens: adjustedMaxTokens,
    };
  }

  /**
   * Get model registry entry for a given model ID
   */
  getModelRegistryEntry(modelId: string): ModelRegistryEntry | undefined {
    return this.tutorConfig.modelRegistry[modelId];
  }

  /**
   * Get classifier configuration
   */
  getClassifierConfig(): ModelConfig & { modelId: string } {
    return this.tutorConfig.classifier;
  }

  /**
   * Get reranker configuration
   */
  getRerankerConfig(): ModelConfig & { modelId: string } {
    return this.tutorConfig.reranker;
  }

  /**
   * Get all available subjects
   */
  getAvailableSubjects(): string[] {
    return Object.keys(this.tutorConfig.subjects);
  }

  /**
   * Get available levels for a subject
   */
  getAvailableLevels(subject: string): string[] {
    const subjectConfig = this.tutorConfig.subjects[subject];
    return subjectConfig ? Object.keys(subjectConfig) : ['basic', 'intermediate', 'advanced'];
  }

  /**
   * Get preferred models for a subscription plan
   */
  getPreferredModels(userSubscription: string): string[] {
    const plan = this.tutorConfig.plans[userSubscription] || this.tutorConfig.plans[this.tutorConfig.defaultSubscription];
    return plan.preferredModels;
  }

  /**
   * Get classification configuration (subject keywords, intent patterns, etc.)
   */
  getClassificationConfig() {
    return classificationConfig;
  }
}

export const modelConfigService = new ModelConfigService();
export default tutorConfigData;