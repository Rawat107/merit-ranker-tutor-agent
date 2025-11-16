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

Example response: {"subject": "math", "level": "intermediate", "confidence": 0.9, "intent": "step_by_step_explanation", "expectedFormat": "Numbered steps with LaTeX formulas and verification", "reasoning": "Trigonometry problem asking for solution steps"}`;
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
  userPrefs?: Record<string, any>,
  conversationHistory?: string,
  userName?: string | null
): string {
  
  const shotExample = topDocument
    ? (() => {
        const url = (topDocument.metadata && (topDocument.metadata as any).url) || '';
        const title = (topDocument.metadata && (topDocument.metadata as any).title) || topDocument.id;
        const linkPart = url ? ` (Link: ${url})` : '';
        const header = `\n[REFERENCE MATERIAL]\nSource: ${title}${linkPart}\n`;
        const body = `${topDocument.text.substring(0, 240)}\n`;
        return header + body;
      })()
    : '';

  const responseFormat = getResponseFormatByIntent(intent);
  const prefsBlock = userPrefs
    ? `Language: ${userPrefs.language} | Exam: ${userPrefs.exam} | Style: ${userPrefs.tutorStyle}\n`
    : '';

  // Format conversation history if provided
  const historyBlock = conversationHistory 
    ? `\n[CONVERSATION HISTORY]\n${conversationHistory}\n`
    : '';

  // Build the system message with natural instructions
  const greetingInstruction = userName 
    ? `The user's name is ${userName}. Greet them naturally by name when appropriate and maintain consistency with previous conversations.`
    : 'If the user introduces themselves or mentions their name in the conversation history, greet them naturally and remember it throughout the conversation.';

  const systemMessage = `You are a professional educator for government exams in India. You are specializing in ${classification.subject}. Your role is to help students prepare for these exams by providing clear, accurate, and actionable answers, and by guiding them to think about what to ask next.
Difficulty Level: ${classification.level.toUpperCase()}
${prefsBlock}

CONVERSATION GUIDELINES:
- ${greetingInstruction}
- Reference previous messages when relevant to maintain context and continuity
- Be natural and conversational while staying accurate and helpful
- If asked about information from earlier in the conversation, recall it accurately
- When citing sources from web search, include the clickable URL if available

RESPONSE FORMAT:
${responseFormat}

CRITICAL: Always follow the response format above. Make your answer clear, accurate, and actionable.

After your main answer, suggest ONE relevant next question the student could ask to continue learning. Format this as:
Suggested Next Question: <your suggested question here>`;

  // Build the human message with the actual query embedded
  const humanMessage = `${historyBlock}Current User Query: ${userQuery}${shotExample}

Provide your response in the format specified above, using the conversation history and reference material as needed. Remember to include a single suggested next question at the end.`;

  // Return as plain text prompt
  return `${systemMessage}\n\n${humanMessage}`;
}

/**
 * Get response format based on user intent
 */
function getResponseFormatByIntent(intent: string): string {
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
- **What Was Right:** [Acknowledge correct parts]`
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
