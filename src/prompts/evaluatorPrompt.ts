import { ChatRequest, Classification, Document } from '../types/index.js';

export function buildTutorPrompt(
  req: ChatRequest,
  classification: Classification,
  docs: Document[],
  userPrefs?: Record<string, any>
): string {
  const contextBlock = docs.map(d => `[[${d.id}]] ${d.text}`).join('\n\n');
  const prefs = userPrefs
    ? `Language: ${userPrefs.language}\nExam: ${userPrefs.exam}\nTutorStyle: ${userPrefs.tutorStyle}\nVerification: ${userPrefs.verification}\n`
    : '';
  return `SYSTEM: Subject=${classification.subject} Level=${classification.level}
${prefs}
USER: ${req.message}
CONTEXT:
${contextBlock}
INSTRUCTIONS: Provide a helpful, accurate answer citing document IDs when used.`;
}