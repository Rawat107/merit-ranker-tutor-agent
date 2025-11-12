import { ChatPromptTemplate } from '@langchain/core/prompts';
import { ChatRequest, Classification, Document } from '../types/index.js';

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
 * Build final evaluation prompt with all context
 * Used by evaluatePrompt to generate final response
 */
export function buildEvaluationPrompt(
  userQuery: string,
  classification: Classification,
  topDocument: Document | null,
  intent: string,
  userPrefs?: Record<string, any>
): ChatPromptTemplate {
  
  const shotExample = topDocument
    ? `\n[REFERENCE]\n[[${topDocument.id}]] ${topDocument.text.substring(0, 400)}\n`
    : '';

  const responseFormat = getResponseFormatByIntent(intent);
  const prefsBlock = userPrefs
    ? `Language: ${userPrefs.language} | Exam: ${userPrefs.exam} | Style: ${userPrefs.tutorStyle}\n`
    : '';

  return ChatPromptTemplate.fromMessages([
    [
      'system',
      `You are a world-class tutor specializing in ${classification.subject}.
Difficulty Level: ${classification.level.toUpperCase()}
${prefsBlock}

RESPONSE FORMAT:
${responseFormat}

CRITICAL: Always follow the response format above. Make your answer clear, accurate, and actionable.`
    ],
    [
      'human',
      `User Query: {query}${shotExample}

Provide your response in the format specified above, using the reference material as support.`
    ]
  ]);
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