export interface Classification {
  subject: string;
  level: 'basic' | 'intermediate' | 'advanced';
  confidence: number;
  intent?: string; // NEW
  expectedFormat?: string;
}

export interface Document {
  id: string;
  text: string;
  metadata: Record<string, any>;
  score?: number;
}

export interface AITutorResponse {
  answer: string;
  sources?: Document[];
  confidence?: number;
  classification: Classification;
  cached?: boolean;
  reasoning?: string;
  metadata?: {
    stage?: string;
    message?: string;
    [key: string]: any;
  };
}

export interface StreamingCallbacks {
  onToken: (token: string) => void;
  onError: (error: Error) => void;
  onComplete: () => void;
}

export interface LLMOptions {
  maxTokens?: number;
  temperature?: number;
  systemPrompt?: string;
  timeoutMs?: number;
}

export interface ModelConfig {
  modelId: string;
  maxTokens: number;
  temperature: number;
  systemPrompt: string;
  outputFormat: string;
}

export interface ModelRegistryEntry {
  bedrockId: string;
  invokeId: string;
  inferenceProfileArn: string | null;
  region: string;
}

export interface PlanConfig {
  description: string;
  preferredModels: string[];
  maxTokensMultiplier: number;
}

export interface TutorConfig {
  defaultSubscription: string;
  plans: Record<string, PlanConfig>;
  modelRegistry: Record<string, ModelRegistryEntry>;
  classifier: ModelConfig & { modelId: string };
  reranker: ModelConfig & { modelId: string };
  subjects: Record<string, Record<string, ModelConfig>>;
}

export interface ChatRequest {
  message: string;
  subject?: string;
  level?: string;
  userSubscription?: string;
  sessionId?: string;
  language?: string;
  examPrep?: boolean;
}

export interface CacheEntry {
  response: string;
  embedding: number[];
  metadata: Record<string, any>;
  timestamp: number;
}

export interface SemanticCacheResult {
  entry: CacheEntry;
  similarity: number;
}

export interface RerankerResult {
  document: Document;
  score: number;
  reason?: string;
}

export interface WebSearchResult {
  title: string;
  url: string;
  snippet: string;
  relevance?: number;
}

export interface ClassificationWithIntent extends Classification {
  intent?: string;
  expectedFormat?: string;
}