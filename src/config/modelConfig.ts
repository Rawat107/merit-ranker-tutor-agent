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
      "preferredModels": ["anthropic.claude-3-sonnet-20240229-v1:0", "openai.gpt-oss-20b-1:0"],
      "maxTokensMultiplier": 1.0
    },
    "premium": {
      "description": "Higher quality — use stronger models for final verification and complex reasoning.",
      "preferredModels": ["anthropic.claude-3-5-sonnet-20241022-v2:0", "openai.gpt-oss-120b-1:0"],
      "maxTokensMultiplier": 1.3
    }
  },
  "modelRegistry": {
    "gpt-3.5-turbo": { "bedrockId": "gpt-3.5-turbo", "invokeId": "gpt-3.5-turbo", "inferenceProfileArn": null, "region": "ap-south-1" },
    "amazon.nova-micro-v1:0": { "bedrockId": "amazon.nova-micro-v1:0", "invokeId": "amazon.nova-micro-v1:0", "inferenceProfileArn": null, "region": "ap-south-1" },
    "amazon.rerank-v1:0": { "bedrockId": "amazon.rerank-v1:0", "invokeId": "amazon.rerank-v1:0", "inferenceProfileArn": null, "region": "ap-south-1" },
    "anthropic.claude-3-sonnet-20240229-v1:0": { "bedrockId": "anthropic.claude-3-sonnet-20240229-v1:0", "invokeId": "anthropic.claude-3-sonnet-20240229-v1:0", "inferenceProfileArn": null, "region": "ap-south-1" },
    "anthropic.claude-3-5-sonnet-20241022-v2:0": { "bedrockId": "anthropic.claude-3-5-sonnet-20241022-v2:0", "invokeId": "anthropic.claude-3-5-sonnet-20241022-v2:0", "inferenceProfileArn": null, "region": "ap-south-1" },
    "openai.gpt-oss-20b-1:0": { "bedrockId": "openai.gpt-oss-20b-1:0", "invokeId": "openai.gpt-oss-20b-1:0", "inferenceProfileArn": null, "region": "ap-south-1" },
    "openai.gpt-oss-120b-1:0": { "bedrockId": "openai.gpt-oss-120b-1:0", "invokeId": "openai.gpt-oss-120b-1:0", "inferenceProfileArn": null, "region": "ap-south-1" }
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
        "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
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
        "modelId": "openai.gpt-oss-20b-1:0",
        "maxTokens": 1600,
        "temperature": 0.0,
        "systemPrompt": "You are an expert tutor. Provide authoritative, citation-backed answers, check reasoning, and flag uncertainty.",
        "outputFormat": "Authoritative answer with sources, reasoning steps, and confidence score."
      }
    },
    "math": {
      "basic": {
        "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
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
        "modelId": "openai.gpt-oss-120b-1:0",
        "maxTokens": 2000,
        "temperature": 0.0,
        "systemPrompt": "You are an expert mathematician. Provide proofs, derivations, and verify via computation where applicable.",
        "outputFormat": "Proof/derivation + final answer"
      }
    },
    "reasoning": {
      "basic": {
        "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
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
        "modelId": "openai.gpt-oss-120b-1:0",
        "maxTokens": 1600,
        "temperature": 0.0,
        "systemPrompt": "You are an expert reasoner. Provide rigorous reasoning and indicate confidence and sources.",
        "outputFormat": "Rigorous reasoning + confidence"
      }
    },
    "english_grammar": {
      "basic": {
        "modelId": "anthropic.claude-3-sonnet-20240229-v1:0",
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
        "modelId": "openai.gpt-oss-20b-1:0",
        "maxTokens": 1200,
        "temperature": 0.0,
        "systemPrompt": "You are an expert English linguist. Provide authoritative corrections, alternatives, and style notes.",
        "outputFormat": "Corrected + stylistic suggestions"
      }
    },
    "science": {
      "basic": { "modelId": "anthropic.claude-3-sonnet-20240229-v1:0", "maxTokens": 800, "temperature": 0.2, "systemPrompt": "You are a science tutor. Keep explanations simple and factual.", "outputFormat": "Answer + simple explanation" },
      "intermediate": { "modelId": "openai.gpt-oss-20b-1:0", "maxTokens": 1200, "temperature": 0.1, "systemPrompt": "You are a science assistant. Provide sourced explanations and reasoning.", "outputFormat": "Answer + citations" },
      "advanced": { "modelId": "openai.gpt-oss-20b-1:0", "maxTokens": 1600, "temperature": 0.0, "systemPrompt": "You are an expert scientist. Provide thorough explanations, citations, and confidence.", "outputFormat": "Detailed answer + citations" }
    },
    "history": {
      "basic": { "modelId": "anthropic.claude-3-sonnet-20240229-v1:0", "maxTokens": 800, "temperature": 0.2, "systemPrompt": "You are a history tutor. Provide concise historical facts and timelines.", "outputFormat": "Facts + timeline" },
      "intermediate": { "modelId": "openai.gpt-oss-20b-1:0", "maxTokens": 1200, "temperature": 0.1, "systemPrompt": "You are a historian. Provide sourced explanations and context.", "outputFormat": "Answer + citations" },
      "advanced": { "modelId": "openai.gpt-oss-20b-1:0", "maxTokens": 1600, "temperature": 0.0, "systemPrompt": "You are an expert historian. Provide deep contextual analysis and sources.", "outputFormat": "Deep analysis + sources" }
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
}

export const modelConfigService = new ModelConfigService();
export default tutorConfigData;