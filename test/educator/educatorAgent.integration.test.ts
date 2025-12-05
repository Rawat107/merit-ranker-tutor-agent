// Minimal runner: Blueprint → topics only → PromptRefiner (with web search)
import pino from 'pino';
import fs from 'fs';
import path from 'path';

import { BlueprintGenerator } from '../../src/educator/generator/blueprintGenerator';
import { PromptRefiner } from '../../src/educator/generator/promptRefiner';
import { webSearchTool } from '../../src/tools/webSearch';
import type { Classification } from '../../src/types';

const logger = pino({ name: 'EducatorBlueprintRefineTest', level: 'info' });

async function main() {
  const userQuery =
    process.argv[2] ?? 'Create some history questions about ancient civilizations';

  // Subject/level can be overridden via CLI args 3 and 4
  const subject = (process.argv[3] as Classification['subject']) ?? 'history';
  const level = (process.argv[4] as Classification['level']) ?? 'intermediate';

  const classification: Classification = {
    subject,
    level,
    confidence: 0.9,
  };

  logger.info({ userQuery, subject, level }, '[RUN] Starting blueprint + refine (topics only)');

  // 1) Generate blueprint (full), but we'll only use topics for refinement
  const blueprintGen = new BlueprintGenerator(logger);
  const blueprint = await blueprintGen.generateBlueprint(userQuery, classification);

  // For visibility: print the full blueprint once, like your sample
  console.log('\nGenerated Blueprint:');
  console.log(JSON.stringify(blueprint, null, 2));

  const topics = blueprint.topics || [];
  if (!topics.length) {
    console.error('[ERROR] No topics produced by blueprint generator');
    process.exit(1);
  }

  // 2) Fetch ONLY web search results (no KB) as requested
  const tavilyKey = process.env.TAVILY_API_KEY || '';
  const webResults = await webSearchTool(userQuery, subject, tavilyKey, logger);

  // 3) Refine prompts using topics + web search; awsKbResults = []
  const refiner = new PromptRefiner(logger);
  const refined = await refiner.refinePrompts(
    userQuery,
    topics,
    webResults,
    [], // no KB as requested
    subject,
    level
  );

  // 4) Output ONLY the requested array: { "Topic": "FULL PROMPT..." }
  const simplified = refined.map((r) => ({ [r.topicName]: r.prompt }));

  console.log('\nrefinedPrompts:');
  console.log(JSON.stringify(simplified, null, 2));

  // Also save to a file for quick inspection
  const outPath = path.join(process.cwd(), 'educator-refined-prompts.json');
  fs.writeFileSync(outPath, JSON.stringify(simplified, null, 2), 'utf8');

  logger.info(
    { topics: topics.length, prompts: simplified.length, outputFile: outPath },
    '[RUN] Done'
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
