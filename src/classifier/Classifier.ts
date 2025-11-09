import { Classification } from '../types/index.js';
import { ILLM } from '../llm/ILLM.js';
import { modelConfigService } from '../config/modelConfig.js';
import pino from 'pino';

export class Classifier {
  private subjects = modelConfigService.getAvailableSubjects();
  constructor(private logger: pino.Logger, private llm?: ILLM) {}

  async classify(query: string): Promise<Classification> {
    try {
      if (this.llm) return await this.llmClassify(query);
      return this.heuristicClassify(query);
    } catch {
      return this.heuristicClassify(query);
    }
  }

  private async llmClassify(query: string): Promise<Classification> {
    const cfg = modelConfigService.getClassifierConfig();
    const resp = await this.llm!.generate(this.buildPrompt(query), {
      maxTokens: cfg.maxTokens,
      temperature: cfg.temperature,
      systemPrompt: cfg.systemPrompt,
    });
    return this.parseResponse(resp);
  }

  private heuristicClassify(query: string): Classification {
    const q = query.toLowerCase();
    let subject: string = 'general';
    if (this.has(q, ['math','equation','integral','derive'])) subject = 'math';
    else if (this.has(q, ['physics','chemistry','biology'])) subject = 'science';
    else if (this.has(q, ['grammar','sentence','correct'])) subject = 'english_grammar';
    else if (this.has(q, ['news','recent','current'])) subject = 'current_affairs';
    else if (this.has(q, ['logic','reason','prove'])) subject = 'reasoning';

    let level: 'basic'|'intermediate'|'advanced' = 'basic';
    if (this.has(q, ['prove','rigorous','theorem','derivation'])) level = 'advanced';
    else if (this.has(q, ['compare','explain','steps','why'])) level = 'intermediate';
    return { subject, level, confidence: 0.7 };
  }

  private buildPrompt(query: string) {
    return `Classify query to subject and level.\nSubjects: ${this.subjects.join(', ')}\nLevels: basic, intermediate, advanced\nQuery: "${query}"\nJSON: {"subject":"<subject>","level":"<level>"}`;
  }

  private parseResponse(resp: string): Classification {
    try {
      const parsed = JSON.parse(resp.trim().replace(/```json|```/g, ''));
      const subject = this.subjects.includes(parsed.subject) ? parsed.subject : 'general';
      const level: 'basic'|'intermediate'|'advanced' = ['basic','intermediate','advanced'].includes(parsed.level) ? parsed.level : 'basic';
      return { subject, level, confidence: 0.9 };
    } catch {
      return { subject: 'general', level: 'basic', confidence: 0.5 };
    }
  }

  private has(q: string, terms: string[]) { return terms.some(t => q.includes(t)); }
}
