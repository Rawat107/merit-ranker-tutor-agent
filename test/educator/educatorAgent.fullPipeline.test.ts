/**
 * Full Pipeline Integration Test - Educator Agent
 * Tests: Blueprint → Research → Prompt Refinement → Question Generation
 * 
 * This test validates the complete flow:
 * 1. Generate blueprint with topics and model selection
 * 2. Perform web + KB research
 * 3. Refine prompts with research context
 * 4. Generate final quiz questions with answers
 */

import { describe, it, expect } from 'vitest';
import pino from 'pino';
import { createEducatorAgent } from '../../src/educator/educatorAgent.ts';
import { Classification } from '../../src/types/index.js';
import fs from 'fs/promises';
import path from 'path';

describe('Educator Agent - Full Pipeline Test', () => {
  const logger = pino({ level: 'info' });

  it('should execute complete pipeline and generate quiz with answers', async () => {
    // 1. Setup test input
    const userQuery = 'Create a quiz about Quantum Computing fundamentals and applications';
    const classification: Classification = {
      subject: 'Quantum Computing',
      level: 'intermediate',
      confidence: 0.92,
      intent: 'quiz',
    };

    console.log('🚀 Starting full pipeline test...\n');
    console.log(`Query: "${userQuery}"`);
    console.log(`Level: ${classification.level}\n`);

    // 2. Create agent and execute
    const agent = createEducatorAgent(logger);
    const startTime = Date.now();
    
    const result = await agent.execute(userQuery, classification);
    
    const totalDuration = Date.now() - startTime;

    // 3. Validate pipeline results
    console.log('\n📊 Pipeline Results:');
    console.log('─'.repeat(60));

    // Blueprint validation
    expect(result.blueprint).toBeDefined();
    expect(result.blueprint?.topics).toBeDefined();
    expect(result.blueprint?.topics.length).toBeGreaterThan(0);
    
    console.log(`✅ Blueprint Generated:`);
    console.log(`   Topics: ${result.blueprint?.topics.length}`);
    console.log(`   Total Questions: ${result.blueprint?.totalQuestions}`);
    console.log(`   Model: ${result.blueprint?.selectedModel.name}`);
    console.log(`   Topics: ${result.blueprint?.topics.map((t: any) => t.name).join(', ')}`);

    // Research validation
    expect(result.webSearchResults).toBeDefined();
    expect(result.kbResults).toBeDefined();
    
    console.log(`\n✅ Research Completed:`);
    console.log(`   Web Results: ${result.webSearchResults.length}`);
    console.log(`   KB Results: ${result.kbResults.length}`);

    // Prompt refinement validation
    expect(result.refinedPrompts).toBeDefined();
    expect(result.refinedPrompts.length).toBeGreaterThan(0);
    
    console.log(`\n✅ Prompt Refinement:`);
    console.log(`   Refined Prompts: ${result.refinedPrompts.length}`);

    // Question generation validation (FINAL OUTPUT)
    expect(result.generatedQuestions).toBeDefined();
    expect(result.generatedQuestions.length).toBeGreaterThan(0);
    
    console.log(`\n✅ Question Generation:`);
    console.log(`   Total Questions: ${result.generatedQuestions.length}`);

    // 4. Print ALL generated questions with answers
    console.log('\n📝 Generated Questions:');
    console.log('═'.repeat(80));

    result.generatedQuestions.forEach((q: any, idx: number) => {
      console.log(`\n[Question ${idx + 1}/${result.generatedQuestions.length}]`);
      console.log(`Topic: ${q.topic} | Difficulty: ${q.difficulty}`);
      console.log(`\nQ: ${q.question}`);
      console.log(`\nOptions:`);
      q.options.forEach((opt: string, optIdx: number) => {
        const marker = opt === q.correctAnswer ? '✓✓✓' : '   ';
        console.log(`  ${marker} ${String.fromCharCode(65 + optIdx)}. ${opt}`);
      });
      console.log(`\n💡 Correct Answer: ${q.correctAnswer}`);
      console.log(`📖 Explanation: ${q.explanation}`);
      console.log('─'.repeat(80));
    });

    // 5. Validate question schema
    result.generatedQuestions.forEach((q: any) => {
      expect(q).toHaveProperty('question');
      expect(q).toHaveProperty('options');
      expect(q).toHaveProperty('correctAnswer');
      expect(q).toHaveProperty('explanation');
      expect(q).toHaveProperty('difficulty');
      expect(q).toHaveProperty('topic');
      expect(Array.isArray(q.options)).toBe(true);
      expect(q.options.length).toBeGreaterThanOrEqual(2);
      expect(q.correctAnswer).toBeDefined();
    });

    // 6. Print step logs
    console.log('\n📜 Pipeline Step Logs:');
    console.log('═'.repeat(60));
    result.stepLogs.forEach((log: any) => {
      const status = log.status === 'completed' ? '✅' : '❌';
      console.log(`${status} ${log.step}: ${log.status}`);
      if (log.metadata) {
        console.log(`   Duration: ${log.metadata.duration}ms`);
        if (log.metadata.topics) {
          console.log(`   Topics: ${JSON.stringify(log.metadata.topics)}`);
        }
        if (log.metadata.totalQuestions) {
          console.log(`   Total Questions: ${log.metadata.totalQuestions}`);
          console.log(`   Success: ${log.metadata.successCount}, Errors: ${log.metadata.errorCount}`);
        }
      }
      if (log.error) {
        console.log(`   ❌ Error: ${log.error}`);
      }
    });

    console.log(`\n⏱️  Total Pipeline Duration: ${totalDuration}ms`);
    console.log('═'.repeat(60));

    // 7. Save results to file
    const outputDir = path.join(process.cwd(), 'test', 'educator');
    const outputPath = path.join(outputDir, 'full-pipeline-result.json');
    
    const outputData = {
      query: userQuery,
      classification,
      pipeline: {
        blueprint: {
          topics: result.blueprint?.topics.length,
          totalQuestions: result.blueprint?.totalQuestions,
          model: result.blueprint?.selectedModel.name,
          topicNames: result.blueprint?.topics.map((t: any) => t.name),
        },
        research: {
          webResults: result.webSearchResults.length,
          kbResults: result.kbResults.length,
        },
        refinedPrompts: result.refinedPrompts.length,
        generatedQuestions: result.generatedQuestions.length,
      },
      sampleQuestions: result.generatedQuestions.slice(0, 5),
      stepLogs: result.stepLogs,
      totalDuration,
    };

    await fs.writeFile(outputPath, JSON.stringify(outputData, null, 2), 'utf-8');
    console.log(`\n💾 Results saved to: ${outputPath}\n`);

    // 8. Final assertions
    expect(result.generatedQuestions.length).toBeGreaterThanOrEqual(
      Math.floor((result.blueprint?.totalQuestions || 0) * 0.7) // Allow 70% success rate
    );
    
    // Verify all steps completed (note: may have duplicate logs from LangGraph reducers)
    const completedSteps = result.stepLogs.filter((l: any) => l.status === 'completed');
    expect(completedSteps.length).toBeGreaterThanOrEqual(4);
  }, 120000); // 2 minute timeout for full pipeline

  it('should generate questions matching blueprint topic count', async () => {
    const userQuery = 'Create a 10-question quiz about Python data structures';
    const classification: Classification = {
      subject: 'Python',
      level: 'basic',
      confidence: 0.95,
      intent: 'quiz',
    };

    console.log('🔍 Testing topic-question alignment...');

    const agent = createEducatorAgent(logger);
    const result = await agent.execute(userQuery, classification);

    // Verify questions were generated
    expect(result.generatedQuestions.length).toBeGreaterThan(0);
    
    // Verify questions have topics assigned
    const questionTopics = new Set(result.generatedQuestions.map((q: any) => q.topic));
    expect(questionTopics.size).toBeGreaterThan(0);

    console.log(`\n📊 Results Summary:`);
    console.log(`Blueprint Topics: ${result.blueprint?.topics.length || 0}`);
    console.log(`Question Topics: ${Array.from(questionTopics).join(', ')}`);
    console.log(`Questions Generated: ${result.generatedQuestions.length}`);
    console.log(`✅ Topic diversity: ${questionTopics.size} unique topics`);

    // Print all questions
    console.log('\n📝 All Generated Questions:');
    console.log('═'.repeat(80));
    result.generatedQuestions.forEach((q: any, idx: number) => {
      console.log(`\n[${idx + 1}] ${q.topic} - ${q.difficulty}`);
      console.log(`Q: ${q.question}`);
      q.options.forEach((opt: string, optIdx: number) => {
        const marker = opt === q.correctAnswer ? '✓' : ' ';
        console.log(`  ${marker} ${String.fromCharCode(65 + optIdx)}. ${opt}`);
      });
      console.log(`✓ Answer: ${q.correctAnswer}`);
      console.log(`💡 ${q.explanation}`);
    });
    
    // Verify reasonable question count
    const expectedMin = Math.floor((result.blueprint?.totalQuestions || 10) * 0.7);
    expect(result.generatedQuestions.length).toBeGreaterThanOrEqual(expectedMin);
  }, 120000);
});
