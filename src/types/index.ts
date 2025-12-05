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
  suggestedNext?: string;
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
  imageGenerator: ModelConfig & { modelId: string; region?: string };
  subjects: Record<string, Record<string, ModelConfig>>;
  prompt_refinement?: Record<string, ModelConfig & { modelId: string }>;

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

export interface SlideOutlineRequest {
  title: string;
  description?: string;
  noOfSlides: number;
  level: 'basic' | 'intermediate' | 'advanced';
  language: 'en' | 'hi';
  designStyle: string;
  colorTheme: string;
  userId: string;
}

export interface SlideOutline {
  slideNumber: number;
  title: string;
  keyPoints: string[];
  speakerNotes?: string;
}

export interface PresentationOutlineResponse {
  slideId: string;
  userId: string;
  title: string;
  status: 'GENERATING' | 'READY' | 'FAILED';
  noOfSlides: number;
  level: string;
  language: string;
  designStyle: string;
  colorTheme: string;
  outline: SlideOutline[];
  webSearchResults?: string;
  createdAt: string;
  updatedAt: string;
}

export interface PresentationSlideContent {
  slideNumber: number;
  title: string;
  content: string;
  keyPoints: string[];
  imageUrl?: string;
  imageAlt?: string;
  speakerNotes?: string;
}

export interface PresentationFinalResponse {
  slideId: string;
  userId: string;
  title: string;
  status: 'READY';
  slidesContent: PresentationSlideContent[];
  totalSlides: number;
  createdAt: string;
  updatedAt: string;
}

// LangChain BaseMessage type (re-export from @langchain/core/messages)
export { BaseMessage } from '@langchain/core/messages';

export * from '../compression/lingua_compressor';
