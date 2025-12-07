import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatRequest, Classification, Document } from '../types/index.js';

/**
 * Build classifier system prompt
 */
export function buildClassifierSystemPrompt(subjects: string[]): string {
  const subjectList = subjects.join(', ');
  return `You are an expert query classifier for an AI tutoring system.

Your task is to classify student queries into:
1. Subject: ${subjectList}
2. Level: basic, intermediate, advanced
3. Intent: what type of response format the user expects
4. Expected Format: how the answer should be structured

**IMPORTANT: Subject must be EXACTLY one of these:** ${subjectList}
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
- **reasoning_puzzle**: Logic/reasoning questions (Why? Reason? Logic?)
- **verification_check**: Check/validate questions (Is this correct? Verify?)
- **summarize**: Requests to shorten or condense a given text.
- **change_tone**: Requests to rewrite a text with a different tone (e.g., formal, friendly).
- **proofread**: Requests to fix grammar, spelling, and punctuation.
- **make_email_professional**: Requests to rewrite an email to be more professional.

Expected Format Examples:
- Factual: "Direct answer with bullet points"
- Steps: "Numbered steps 1,2,3... with LaTeX for math"
- Compare: "Markdown table for comparison"
- Problem: "Problem breakdown → approach → solution"
- Reasoning: "Logical chain with conclusions and winning strategy"
- Verify: "Yes/No with explanation and corrections if needed"

Provide a confidence score (0.0 to 1.0) indicating your certainty.

You MUST respond with ONLY a valid JSON object. Do not include any text before or after the JSON.
The JSON must have these fields: subject, level, confidence, reasoning (optional), intent, expectedFormat.

Example response: {{"subject": "math", "level": "intermediate", "confidence": 0.9, "intent": "step_by_step_explanation", "expectedFormat": "Numbered steps with LaTeX formulas and verification", "reasoning": "Trigonometry problem asking for solution steps"}}`;
}


export function buildHindiResponseWrapper(systemPrompt: string): string {
  return `${systemPrompt}

IMPORTANT: User prefers responses in HINDI (हिंदी). 
Please respond in Hindi/Hinglish format.
Use simple Hindi mixed with English technical terms where needed.
Example: "यह quadratic equation है जिसका solution x = 2 है।"`;
}


export function buildMCQGenerationPrompt(
  topic: string,
  count: number = 10,
  difficulty: 'basic' | 'intermediate' | 'advanced' = 'intermediate',
  subject: string = 'general'
): string {
  return `You are an expert question paper setter for ${subject}.

Create exactly ${count} Multiple Choice Questions on the topic: "${topic}"

Difficulty Level: ${difficulty.toUpperCase()}

For each MCQ, follow this format:

Q1. [Question text]
A) Option 1
B) Option 2
C) Option 3 (Correct)
D) Option 4

[Explanation]: [Why C is correct, why others are wrong]

---

Requirements:
- Questions must be clear and unambiguous
- Options should be plausible but clearly different
- Include mix of conceptual, calculation, and application questions
- Mark correct answer with (Correct) tag
- Provide brief explanation after each question
- Ensure all questions are unique

Generate ${count} questions following this format exactly.`;
}

export function buildNoteGenerationPrompt(
  topic: string,
  sourceText?: string,
  includeExamples: boolean = true
): string {
  return `You are an expert study note creator for government exam preparation (SSC, Banking, Railways).

Topic: ${topic}
${sourceText ? `Reference: ${sourceText.substring(0, 500)}` : ''}

Create comprehensive study notes with the following structure:

## Key Concepts
[Define 2-3 main concepts related to ${topic}]

## Definition
[Clear, exam-friendly definition]

## Key Points
- Point 1
- Point 2
- Point 3
- Point 4

## Formula/Important Facts
[List relevant formulas or facts]
${includeExamples ? `
## Examples
[Provide 2-3 worked examples]

## Common Mistakes
[List 2-3 common student mistakes]

## Quick Revision
[One-liner summary]` : ''}

## Related Topics
[List 2-3 related topics students should learn]

Format: Make notes exam-focused, concise, and easy to remember.
Use bullet points where appropriate.
Include mnemonics if applicable.`;
}

/**
 * Build enhanced prompt template with LangChain
 * Includes: converted query + reranker top 1 result (shot) + intent + response structure
 */
export function buildEnhancedPrompt(
  req: ChatRequest,
  classification: Classification,
  topDocument: Document | null,
  userPrefs?: Record<string, any>
): ChatPromptTemplate {

  const shotExample = topDocument
    ? `\n[EXAMPLE FROM KB]\nSource: ${topDocument.id}\nContent: ${topDocument.text.substring(0, 300)}...\n`
    : '';

  const intentGuidance = buildIntentGuidance(classification);
  const prefsBlock = userPrefs
    ? `Language: ${userPrefs.language}\nExam: ${userPrefs.exam}\nStyle: ${userPrefs.tutorStyle}\n`
    : '';

  return ChatPromptTemplate.fromMessages([
    [
      'system',
      `You are an expert tutor specialized in ${classification.subject} at ${classification.level} level.
        ${prefsBlock}
        ${intentGuidance}
        Provide accurate, helpful answers. Cite sources when using provided examples.
        Always format responses according to the intent guidance above.`
    ],
    [
      'human',
      `Question: {query}
      ${shotExample}
      Please provide a response following the intent guidance.`
    ]
  ]);
}


export function buildMockTestPrompt(
  subject: string,
  topics: string[],
  questionCount: number = 20,
  duration: number = 60 // minutes
): string {
  return `Create a mock test for ${subject} exam preparation.

  Topics Covered: ${topics.join(', ')}
  Total Questions: ${questionCount}
  Suggested Duration: ${duration} minutes

  Structure:
  - ${Math.floor(questionCount * 0.5)} MCQ (1 mark each)
  - ${Math.floor(questionCount * 0.3)} Short Answer (2 marks)
  - ${Math.floor(questionCount * 0.2)} Long Answer (3 marks)

  For each question:
  Q1. [Question text]
  [Type: MCQ/Short Answer/Long Answer]
  [Category: Topic name]
  [Marks: X]
  ${true ? `[Answer Key with explanation]` : ''}

  Instructions:
  - Mix easy, medium, and difficult questions
  - Ensure good coverage of all topics
  - Avoid duplicate concepts
  - Make questions exam-relevant
  - Include previous year question style
  - Ensure all questions are unique

  Generate the complete mock test in structured format.`;
}

export function buildQuizEvaluationPrompt(
  question: string,
  studentAnswer: string,
  correctAnswer: string,
  subject: string = 'general'
): string {
  return `You are an expert exam evaluator for ${subject}.

  Evaluate this student's answer:

  QUESTION: ${question}

  STUDENT'S ANSWER: ${studentAnswer}

  CORRECT ANSWER: ${correctAnswer}

  Provide evaluation in this format:

  ## Marks: X/Y
  [Give partial marks if answer is partially correct]

  ## Analysis:
  - What the student got right
  - What the student missed
  - Common mistake (if any)

  ## Correct Explanation:
  [Clear explanation of why this is the correct answer]

  ## Learning Points:
  [2-3 key learnings for the student]

  ## Tips for Similar Questions:
  [Specific strategy for answering similar questions]

  Be encouraging but constructive.`;
}

/**
 * Build intent-specific response guidance
 */
function buildIntentGuidance(classification: Classification): string {
  const intent = (classification as any).intent || 'factual_retrieval';

  const guidanceMap: Record<string, string> = {
    'factual_retrieval': 'Provide a direct, concise answer with bullet points if multiple items. Include source citations.',
    'step_by_step_explanation': 'Break the response into numbered steps (1., 2., 3., ...). Explain each step clearly. End with "Final Answer:"',
    'comparative_analysis': 'Create a structured comparison using markdown tables. List similarities first, then key differences.',
    'problem_solving': 'Restate the problem → Identify approach → Work through solution step-by-step → Verify the answer',
    'reasoning_puzzle': 'Present reasoning as a clear logical chain. Number each deduction. End with definitive conclusion.',
    'verification_check': 'Begin with TRUE/FALSE verdict. Immediately explain why. Provide correct answer if incorrect.'
  };

  return `**Response Intent: ${intent}**\n${guidanceMap[intent] || guidanceMap['factual_retrieval']}`;
}

/**
 * Build final evaluation prompt with all context and conversation history
 * Used by evaluatePrompt to generate final response
 * 
 * IMPORTANT: Returns plain text to avoid conflicts with LaTeX curly braces
 */
export function buildEvaluationPrompt(
  userQuery: string,
  classification: Classification,
  topDocument: Document | null,
  intent: string,
  userPrefs?: Record<string, string>,
  conversationHistory?: string,
  userName?: string | null
): {
  systemPrompt: string;
  conversationHistory: string;
  reference: string;
  currentQuery: string;
} {

  // ✅ REFERENCE ONLY - will be compressed
  const referenceBlock = topDocument
    ? (() => {
      const url = (topDocument.metadata && (topDocument.metadata as any).url) || '';
      const title = (topDocument.metadata && (topDocument.metadata as any).title) || topDocument.id;
      const linkPart = url ? ` (Link: ${url})` : '';
      const header = `\n[REFERENCE MATERIAL]\nSource: ${title}${linkPart}\n`;
      const body = `${topDocument.text || ''}\n`;
      return header + body;
    })()
    : '';

  const responseFormat = getResponseFormatByIntent(intent);

  const prefsBlock = userPrefs
    ? `Language: ${userPrefs.language} | Exam: ${userPrefs.exam} | Style: ${userPrefs.tutorStyle}\n`
    : '';

  const greetingInstruction = userName
    ? `The user's name is ${userName}. Greet them naturally by name when appropriate and maintain consistency with previous conversations.`
    : 'If the user introduces themselves or mentions their name in the conversation history, greet them naturally and remember it throughout the conversation.';

  const systemPrompt = `You are a professional educator for government exams in India. You are specializing in ${classification.subject}. Your role is to help students prepare for these exams by providing clear, accurate, and actionable answers, and by guiding them to think about what to ask next.

Difficulty Level: ${classification.level.toUpperCase()}

${prefsBlock}
CONVERSATION GUIDELINES:
- ${greetingInstruction}
- Reference previous messages when relevant to maintain context and continuity
- Be natural and conversational while staying accurate and helpful
- If asked about information from earlier in the conversation, recall it accurately
- When citing sources from web search, include the clickable URL if available

CRITICAL: Do NOT repeat or include the conversation history in your response. Use it only as context to understand the current query.

RESPONSE FORMAT:
${responseFormat}

CRITICAL: Always follow the response format above. Make your answer clear, accurate, and actionable.

After your main answer, suggest ONE relevant next question the student could ask to continue learning.`;

  //  CONVERSATION HISTORY - will be compressed
  const historyBlock = conversationHistory
    ? `\n[CONVERSATION HISTORY]\n${conversationHistory}\n`
    : '';

  //  CURRENT QUERY - will NOT be compressed
  const currentQueryBlock = `\n[CURRENT QUERY]\n${userQuery}\n\nProvide your response in the format specified above, using the conversation history and reference material as needed. Remember to include a single suggested next question at the end.`;

  return {
    systemPrompt,
    conversationHistory: historyBlock,
    reference: referenceBlock,
    currentQuery: currentQueryBlock
  };
}

export function buildStandaloneRewritePrompt(): string {
  return `You are an expert at converting context-dependent queries into standalone, self-contained questions for an AI tutor.

Your task: Rewrite ONLY the CURRENT QUERY to make it fully understandable without needing conversation history. The rewritten query must be a clear, direct question.

**STRICT RULES (Follow exactly):**
1. Focus SOLELY on the "Current Query" line. Analyze it independently first.
2. Use "Conversation History" ONLY to resolve pronouns or vague references (e.g., "it" → specific noun from history; "their" → entity mentioned). If no pronouns/references in Current Query, IGNORE history completely and return Current Query as-is (after minor cleanup).
3. Remove casual greetings (e.g., "Hi [name],", "Hello,") to make it a pure question, but keep the core intent.
4. Do NOT summarize, repeat, or add from history unless resolving a direct reference. Do NOT change the meaning or add unrelated details.
5. Correct obvious typos/spelling only if they obscure meaning (e.g., "know value" → "tell me the value"), but keep original phrasing otherwise.
6. Output ONLY the rewritten query as plain text. NO explanations, quotes, JSON, labels (e.g., no "Rewritten: "), or extra sentences. If already standalone, return exactly (cleaned).

**Input Format:**
Current Query: [the user's latest message]
Conversation History: [previous messages - use only for reference resolution]

**Examples:**

Example 1:
Current Query: "What about their economy?"
Conversation History: Previous: Tell me about Japan. Assistant: Japan's capital is Tokyo...
Output: What is Japan's economy like?

Example 2:
Current Query: "Can you explain photosynthesis?"
Conversation History: [any]
Output: Can you explain photosynthesis?

Example 3:
Current Query: "Hi Vaibhav, can you know value of 234 pi"
Conversation History: [unrelated or none]
Output: What is the value of 234 π?

**Final Note:** If Current Query has no references to history, output a cleaned version of it directly. Always produce a valid, standalone question.`;
}


/**
 * Get response format based on user intent
 */
export function getResponseFormatByIntent(intent: string): string {
  const formats: Record<string, string> = {
    'factual_retrieval': `
- Start with "Answer:" 
- Provide direct answer in 1-2 sentences
- Add key points as bullet list
- End with source if applicable`,

    'step_by_step_explanation': `
1. Begin with "Steps:"
2. Number each step clearly
3. Explain the "why" for each step
4. Use LaTeX for formulas (\\( ... \\))
5. End with "Final Answer:"`,

    'comparative_analysis': `
- Create Markdown table if 2-4 items
- Format: | Item | Aspect 1 | Aspect 2 |
- List similarities first (section: "Similarities")
- Then list key differences (section: "Key Differences")
- Highlight what makes each distinct`,

    'problem_solving': `
- **Problem:** [Restate problem]
- **Approach:** [Method to solve]
- **Solution:** [Step-by-step work]
- **Verification:** [Check answer]
- **Final Answer:** [Result]`,

    'reasoning_puzzle': `
- **Given:** [List facts/assumptions]
- **Reasoning Chain:**
  1. [First deduction] → [Why?]
  2. [Second deduction] → [Why?]
- **Conclusion:** [Final answer with confidence]`,

    'verification_check': `
- **Verdict:** [TRUE/FALSE/CORRECT/INCORRECT]
- **Explanation:** [Why this verdict]
- **Correct Answer:** [If wrong, provide correct version]
- **What Was Right:** [Acknowledge correct parts]`,
    'summarize': `
- **Summary:** [Provide a concise summary of the text]`,
    'change_tone': `
- **Rewritten Text:** [Provide the text rewritten in the requested tone]`,
    'proofread': `
- **Corrected Text:** [Provide the text with grammar, spelling, and punctuation fixed]`,
    'make_email_professional': `
- **Professional Email:** [Provide a professionally rewritten version of the email]`,

    'mcq_generation': `
  - Begin with "MCQ Questions:"
  - Format each as: Q1. [Question], A) [opt], B) [opt], C) [opt], D) [opt]
  - Mark correct answer
  - End with explanation for each`,

    'note_generation': `
  - Begin with "## Key Concepts"
  - Structure: Definition → Key Points → Examples → Revision
  - Use bullet points and numbers
  - Keep notes concise and exam-focused`,

    'mock_test_generation': `
  - Begin with "Mock Test - [Subject]"
  - Include MCQ + Short Answer + Long Answer mix
  - Provide marks for each question
  - Include answer keys with brief explanations`,

    'quiz_evaluation': `
  - Begin with "Evaluation Report"
  - Provide: Marks, Analysis, Correct Explanation, Learning Points
  - Be constructive and encouraging`
  };

  return formats[intent] || formats['factual_retrieval'];
}

/**
 * Build system prompt for model selection
 */
export function getSystemPromptByClassification(classification: Classification): string {
  const subject = classification.subject;
  const level = classification.level;

  const systemPrompts: Record<string, Record<string, string>> = {
    math: {
      basic: 'You are a math tutor. Provide clear step-by-step solutions with formulas.',
      intermediate: 'You are an expert math tutor. Verify all calculations and show reasoning.',
      advanced: 'You are a master mathematician. Provide rigorous proofs and derivations.'
    },
    reasoning: {
      basic: 'Break down logic problems into simple steps.',
      intermediate: 'Provide clear chain-of-thought reasoning.',
      advanced: 'Provide rigorous logical analysis with confidence levels.'
    },
    english_grammer: {
      basic: 'Correct grammar mistakes and explain rules simply.',
      intermediate: 'Provide corrections with detailed explanations.',
      advanced: 'Provide expert grammatical analysis with style suggestions.'
    },
    general: {
      basic: 'Provide helpful, concise answers.',
      intermediate: 'Provide detailed, well-cited answers.',
      advanced: 'Provide authoritative, well-reasoned answers.'
    }
  };

  return systemPrompts[subject]?.[level] || systemPrompts.general[level] || systemPrompts.general.basic;
}


export function buildPresentationOutlinePrompt(
  request: any,
  searchResults: string
): string {
  return `You are an expert presentation designer. Create a detailed outline for a presentation.

Title: "${request.title}"
Description: ${request.description || "Not provided"}
Number of Slides: ${request.noOfSlides}
Level: ${request.level}
Language: ${request.language}
Design Style: ${request.designStyle}

Web Search Context:
${searchResults}

Create a JSON array with exactly ${request.noOfSlides} slide objects. Each slide must have:
- slideNumber: number (1 to ${request.noOfSlides})
- title: string (engaging slide title)
- keyPoints: string[] (3-5 concise points)
- speakerNotes: string (brief speaker talking points)

Return ONLY valid JSON array, no markdown formatting.

Example format:
[
  {
    "slideNumber": 1,
    "title": "Introduction to the Topic",
    "keyPoints": ["Key concept 1", "Key concept 2", "Key concept 3"],
    "speakerNotes": "Welcome the audience and introduce the main topic"
  }
]`;
}

/**
 * Build slide content generation prompt
 */
export function buildSlideContentPrompt(slide: any, context: string): string {
  return `Generate detailed content for a presentation slide.

Slide Title: "${slide.title}"
Key Points: ${slide.keyPoints.join(", ")}
Context: ${context.substring(0, 500)}

Create professional, engaging slide content. The content should:
1. Have a clear main message
2. Support the key points
3. Be concise (150-250 words)
4. Include a relevant image description for Unsplash search

Return JSON with:
{
  "content": "markdown content here with ## headings and bullet points",
  "imageSearchQuery": "specific search query for Unsplash (2-4 words)"
}`;
}



/**
 * Assessment type definition
 */
export type AssessmentType = "quiz" | "mock_test" | "test_series";

/**
 * Build assessment generation prompt
 * Handles quiz, mock test, and test series with proper formatting
 */
export function buildAssessmentGenerationPrompt(params: {
  category: AssessmentType;
  subject: string;
  topic?: string;
  difficulty: "EASY" | "MEDIUM" | "HARD";
  questionCount: number;
  numberOfTests?: number;
  examPattern?: string;
  sectionsPattern?: Array<{
    name: string;
    questions: number;
    marks?: number;
    time?: number;
  }>;
  needExplanations: boolean;
  exams?: string[];
  durationMinutes?: number;
}): string {
  // Get category-specific rules
  const categoryRules = getCategoryRules(params.category, params.needExplanations);

  // Build sections description
  let sectionsDescription = "";
  if (params.sectionsPattern && params.sectionsPattern.length > 0) {
    sectionsDescription = `\nSections and Pattern:\n${params.sectionsPattern
      .map(
        (s) =>
          `- ${s.name}: ${s.questions} questions${s.marks ? ` (${s.marks} marks)` : ""}${s.time ? ` (${s.time} min)` : ""}`
      )
      .join("\n")}`;
  } else if (params.examPattern) {
    sectionsDescription = `\nExam Pattern: ${params.examPattern}`;
  } else {
    sectionsDescription = `\nPattern: Auto-detect common pattern for ${params.exams?.join(", ") || "competitive exams"}`;
  }

  // Build duration note
  const durationNote = params.durationMinutes
    ? `\nSuggested Duration: ${params.durationMinutes} minutes`
    : "";

  // Build exams note
  const examsNote = params.exams && params.exams.length > 0
    ? `\nTarget Exams: ${params.exams.join(", ")}`
    : "";

  // Test series specific parameters
  const testSeriesNote =
    params.category === "test_series" && params.numberOfTests
      ? `\nGenerate ${params.numberOfTests} separate full-length tests, each with ${params.questionCount} questions.`
      : "";

  return `You are an assessment generator for competitive exams.

TASK: Generate ${params.category === "test_series" ? `${params.numberOfTests} separate` : "a"} ${params.category.replace(/_/g, " ")} ${params.category === "test_series" ? "tests" : ""}

Category: ${params.category}
Subject: ${params.subject}
${params.topic ? `Topic: ${params.topic}` : ""}
Difficulty: ${params.difficulty}
Questions per ${params.category === "test_series" ? "test" : "assessment"}: ${params.questionCount}
${testSeriesNote}
${sectionsDescription}
${examsNote}
${durationNote}

CATEGORY RULES:
${categoryRules}

OUTPUT REQUIREMENTS:
${getOutputFormat(params.category, params.needExplanations)}

IMPORTANT GUIDELINES:
- Use clear, unambiguous language
- All questions must be multiple-choice with 4 options (A, B, C, D)
- Ensure questions are non-repetitive within each test
- Options should be plausible but clearly distinguishable
- Maintain consistent difficulty level throughout
- If explanations are included, keep them concise (2-3 lines)
${params.category === "test_series" ? "- Do NOT include any explanations for test_series, only Q, options, and correct answer" : ""}
${params.category === "mock_test" ? "- Structure sections properly with correct marks and timing" : ""}

Now generate the assessment:`;
}

/**
 * Get category-specific rules
 */
function getCategoryRules(
  category: AssessmentType,
  needExplanations: boolean
): string {
  switch (category) {
    case "quiz":
      return `- Create a small-to-medium practice set
- Questions should be straightforward but comprehensive
- Pattern is flexible based on the subject
- ${needExplanations ? "Include brief 2-3 line explanations after each question" : "No explanations required"}
- Ideal for quick practice and concept reinforcement`;

    case "mock_test":
      return `- Create a single, full-length test that simulates the real exam
- Follow the proper exam pattern with sections, marks, and timing
- Include realistic difficulty distribution
- ${needExplanations ? "Provide detailed solutions after all questions (or mark solutions section)" : "No explanations required"}
- Ensure proper marks allocation per question
- Include negative marking rules if applicable`;

    case "test_series":
      return `- Create multiple separate full-length tests
- Each test must follow the exact exam pattern specified
- All tests should have same structure but different questions
- ONLY output questions, options, and correct answers
- Do NOT include any explanations, rationales, or detailed solutions
- Each test is independent and complete`;

    default:
      return "Invalid category";
  }
}

/**
 * Get output format instructions
 */
function getOutputFormat(
  category: AssessmentType,
  needExplanations: boolean
): string {
  const baseFormat = `For each question:
- Question ID (Q1, Q2, etc.)
- Question text
- Options (A, B, C, D)
- Correct option`;

  switch (category) {
    case "quiz":
      return `${baseFormat}${
        needExplanations ? `
- Explanation (2-3 lines, if applicable)` : ""
      }

Example format:
Q1. What is 50% of 200?
A) 50
B) 100
C) 150
D) 200
Correct: B
${needExplanations ? `Explanation: 50% means 1/2, so 50% of 200 = 200/2 = 100.` : ""}`;

    case "mock_test":
      return `${baseFormat}${
        needExplanations ? `
- Detailed Explanation/Solution` : ""
      }

Structure:
[SECTION NAME – Total Marks: X, Time: Y minutes]
Q1. [question] | Options A-D | Correct: [X]
Q2. [question] | Options A-D | Correct: [X]
...
${
  needExplanations
    ? `

[SOLUTIONS / EXPLANATIONS SECTION]
Q1. [Detailed explanation/solution]
Q2. [Detailed explanation/solution]
...`
    : ""
}`;

    case "test_series":
      return `${baseFormat}

Format for each test:
TEST 1
[SECTION NAME – Total Marks: X, Time: Y minutes]
Q1. [question] | A) ... | B) ... | C) ... | D) ... | Correct: [X]
Q2. [question] | A) ... | B) ... | C) ... | D) ... | Correct: [X]
...

TEST 2
[Same structure, different questions]
...

DO NOT include explanations, solutions, or rationales anywhere in the output.`;

    default:
      return baseFormat;
  }
}

/**
 * Build blueprint generation prompt for topic creation
 * Used when topics are NOT provided by the user
 */
export function buildBlueprintGenerationPrompt(
  examTags: string[],
  subject: string,
  totalQuestions: number,
  difficultyLevel: string,
  maxQuestionsPerTopic: number
): string {
  return `You are an expert educator creating assessment topics for ${examTags.join(', ')} exam preparation.

Subject: ${subject}
Total Questions Required: ${totalQuestions}
Difficulty Level: ${difficultyLevel}
Max Questions per Topic: ${maxQuestionsPerTopic}

Generate a topic breakdown following these rules:
1. Create 3-5 relevant topics for this subject and exam
2. Each topic should have at most ${maxQuestionsPerTopic} questions
3. Distribute all ${totalQuestions} questions across topics exactly (sum must equal ${totalQuestions})
4. For each topic, assign difficulty levels from: ["easy"], ["medium"], ["hard"], ["mix"], ["easy", "medium"], etc.
5. Topics must be relevant to ${examTags.join(', ')} exam pattern

Respond ONLY with valid JSON in this exact format:
{
  "topics": [
    { "topicName": "Topic Name", "level": ["mix"], "noOfQuestions": 10 },
    { "topicName": "Another Topic", "level": ["easy", "medium"], "noOfQuestions": 10 }
  ]
}`;
}

/**
 * Build batch question generation prompt
 * Used by batchRunner to generate questions with patterns from research
 */
export function buildBatchQuestionPrompt(
  topic: string,
  context: string,
  patterns: Record<string, string[]>,
  noOfQuestions: number,
  numberRanges: { min: number; max: number; decimals: boolean },
  optionStyle: string,
  avoid: string[],
  includeExplanation: boolean
): string {
  const patternSections = Object.entries(patterns).map(([level, patternList]) => {
    return `\n${level.toUpperCase()}:\n${patternList.map((p, i) => `  ${i + 1}. ${p}`).join('\n')}`;
  }).join('\n');

  return `You are an expert question generator for competitive exams.

TOPIC: ${topic}
CONTEXT: ${context}

DIFFICULTY LEVELS & PATTERNS:${patternSections}

REQUIREMENTS:
- Generate exactly ${noOfQuestions} questions
- Distribute questions across all difficulty levels (${Object.keys(patterns).join(', ')})
- Follow the patterns listed above for each difficulty level
- Use number ranges: min=${numberRanges.min}, max=${numberRanges.max}, decimals=${numberRanges.decimals}
- Option style: ${optionStyle}
- Avoid: ${avoid.join(', ')}
${includeExplanation ? '- Include detailed explanation for each question' : '- Do NOT include explanations'}

OUTPUT FORMAT (JSON only, no markdown):
{
  "questions": [
    {
      "question": "Full question text here",
      "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
      "correctAnswer": "Option 2",${includeExplanation ? '\n      "explanation": "Detailed explanation here",' : ''}
      "difficulty": "easy"
    }
  ]
}

Generate ${noOfQuestions} high-quality questions now.`;
}

/**
 * Build prompt refinement prompt for extracting question patterns from research
 * Used by promptRefiner to analyze research and create structured prompts
 */
export function buildPromptRefinementPrompt(
  examTags: string[],
  subject: string,
  topicName: string,
  topicLevel: string[],
  noOfQuestions: number,
  researchContext: string
): string {
  const levelString = topicLevel.join(', ');
  
  return `You are an expert exam question pattern analyzer for ${examTags.join(', ')} exams.

TASK: Analyze the research below and extract question patterns for the topic "${topicName}".

TOPIC DETAILS:
- Topic: ${topicName}
- Subject: ${subject}
- Exam: ${examTags.join(', ')}
- Difficulty Levels Required: ${levelString}
- Number of Questions: ${noOfQuestions}

RESEARCH CONTEXT:
${researchContext}

INSTRUCTIONS:
1. Extract specific question patterns from the research for each difficulty level: ${levelString}
2. Each pattern should describe a type of question commonly asked
3. Focus on calculation methods, problem types, and common variations
4. Include 3-5 patterns per difficulty level
5. Create a brief context summary (2-3 sentences) about what this topic covers in ${examTags.join(', ')}

OUTPUT FORMAT (JSON only, no markdown):
{
  "topic": "${topicName}",
  "prompt": {
    "noOfQuestions": ${noOfQuestions},
    "patterns": {
${topicLevel.map(lvl => `      "${lvl}": ["pattern 1", "pattern 2", "pattern 3", "pattern 4", "pattern 5"]`).join(',\n')}
    },
    "numberRanges": {
      "min": 1,
      "max": 1000,
      "decimals": false
    },
    "optionStyle": "${examTags[0]}",
    "avoid": ["repetitive numbers", "same question structure", "obvious answers"],
    "context": "Brief 2-3 sentence summary of what ${topicName} covers in ${examTags.join(', ')}"
  }
}

Respond with ONLY the JSON object.`;
}
