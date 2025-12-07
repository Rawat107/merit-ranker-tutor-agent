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

/**
 * Assessment Types and Schemas
 * Used by blueprintGenerator, promptRefiner, batchRunner for quiz, mock_test, test_series
 */
export type AssessmentCategory = 'quiz' | 'mock_test' | 'test_series';
export type AssessmentDifficulty = 'EASY' | 'MEDIUM' | 'HARD';
export type TopicLevel = 'easy' | 'medium' | 'hard' | 'mix';

export interface TopicRequest {
  topicName: string;
  level: TopicLevel[];
  noOfQuestions: number;
}

export interface AssessmentRequest {
  examTags: string[];
  subject: string;
  totalQuestions: number;
  topics?: TopicRequest[];
  userId?: string;
}

export interface AssessmentQuestion {
  questionId: string;
  question: string;
  options: string[];
  correctAnswer: string;
  explanation?: string; // Not included for test_series
  marks?: number;
  negativeMarks?: number;
  section?: string;
  difficulty: 'basic' | 'intermediate' | 'advanced';
  topic: string;
  subject?: string;
  format?: 'standard' | 'math' | 'science' | 'reasoning' | 'diagram';
}

export interface AssessmentSection {
  name: string;
  questionCount: number;
  marks?: number;
  time?: number;
}

export interface AssessmentResponse {
  success: boolean;
  assessmentId: string;
  name: string;
  category: AssessmentCategory;
  subject: string;
  topic?: string;
  difficulty: AssessmentDifficulty;
  exams?: string[];
  totalQuestions: number;
  durationMinutes?: number;
  sections?: AssessmentSection[];
  questions: AssessmentQuestion[];
  explanationsIncluded: boolean;
  meta?: Record<string, any>;
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

/**
 * ============================================
 * EDUCATOR MODULE INTERFACES
 * ============================================
 */

/**
 * Topic Validator Configuration and Results
 */
export interface ValidatorConfig {
  maxQuestionsPerNewTopic?: number;
  distributionStrategy?: "round-robin" | "priority" | "proportional";
  allowBlueprintWhenTopicsProvided?: boolean;
}

export interface ValidationResult {
  isValid: boolean;
  action: "bypass_blueprint" | "generate_blueprint" | "adjust_topics";
  topics: Array<{
    topicName: string;
    level: string[];
    noOfQuestions: number;
  }> | null;
  reason: string;
  metadata: Record<string, any>;
}

/**
 * Research Module Interfaces
 */
export interface EnrichedTopic {
  topicName: string;
  level: string[];
  noOfQuestions: number;
  research: {
    web: Document[];
    kb: Document[];
  };
}

export interface BlueprintInput {
  examTags: string[];
  subject: string;
  totalQuestions: number;
  topics: Array<{
    topicName: string;
    level: string[];
    noOfQuestions: number;
  }>;
}

export interface EnrichedBlueprint {
  examTags: string[];
  subject: string;
  totalQuestions: number;
  topics: EnrichedTopic[];
}

/**
 * Prompt Refiner Interfaces
 */
export interface RefinedPrompt {
  topic: string;
  prompt: {
    noOfQuestions: number;
    patterns: {
      [key: string]: string[]; // e.g., "easy": [...], "medium": [...], "hard": [...]
    };
    numberRanges: {
      min: number;
      max: number;
      decimals: boolean;
    };
    optionStyle: string;
    avoid: string[];
    context: string; // Compressed summary from research
  };
}

/**
 * Batch Runner Interfaces
 */
export type GeneratedQuestion = AssessmentQuestion;

export interface GeneratedQuestionOutput {
  slotId: number;
  q: string;
  options: string[];
  answer: number; // Index of correct option (1-based)
  explanation?: string; // Optional, excluded for test_series
}

export interface QuestionBatchResultItem {
  topic: string;
  questions: GeneratedQuestionOutput[];
  rawResponse?: string;
  error?: string;
}

export interface QuestionBatchResult {
  items: QuestionBatchResultItem[];
  totalQuestions: number;
  successCount: number;
  errorCount: number;
  duration: number;
}
