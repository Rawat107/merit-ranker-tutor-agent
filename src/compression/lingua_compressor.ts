import { LLMLingua2 } from '@atjsh/llmlingua-2';
import { Tiktoken } from "js-tiktoken/lite";
import o200k_base from "js-tiktoken/ranks/o200k_base";
import pino from 'pino';

// Define interface for compression options
export interface CompressionOptions {
    rate?: number;
    targetToken?: number;
    useSentenceLevel?: boolean;
}

/**
 * LinguaCompressor - Singleton service for prompt compression using LLMLingua-2
 */
export class LinguaCompressor {
    private static instance: LinguaCompressor;
    private compressor: any = null;
    private initializationPromise: Promise<void> | null = null;
    private logger: pino.Logger;

    // Configuration - optimized for CPU usage
    private readonly modelName = "atjsh/llmlingua-2-js-tinybert-meetingbank";
    private readonly device = 'cpu';  // CPU-only, safe for ECS
    private readonly dtype = 'fp32';  // Standard precision

    private constructor(logger?: pino.Logger) {
        this.logger = logger || pino({ level: 'info' });
    }

    public static getInstance(logger?: pino.Logger): LinguaCompressor {
        if (!LinguaCompressor.instance) {
            LinguaCompressor.instance = new LinguaCompressor(logger);
        }
        return LinguaCompressor.instance;
    }

    /**
     * Initializes the compressor model. This is called automatically by compress(),
     * but can be called manually to pre-load the model at startup.
     */
    public async init(): Promise<void> {
        if (this.compressor) {
            this.logger.debug('LinguaCompressor already initialized, skipping');
            return;
        }

        if (this.initializationPromise) {
            this.logger.debug('LinguaCompressor initialization in progress, waiting...');
            return this.initializationPromise;
        }

        this.initializationPromise = (async () => {
            const startTime = Date.now();
            try {
                this.logger.info(
                    {
                        model: this.modelName,
                        device: this.device,
                        dtype: this.dtype
                    },
                    '[LinguaCompressor] Initializing compression model...'
                );

                // Initialize OpenAI tokenizer for accurate token counting
                const oaiTokenizer = new Tiktoken(o200k_base);
                this.logger.debug('[LinguaCompressor] Tokenizer initialized');

                // Load the BERT model for compression
                const result = await LLMLingua2.WithBERTMultilingual(this.modelName, {
                    transformerJSConfig: {
                        device: this.device,
                        dtype: this.dtype,
                    },
                    oaiTokenizer: oaiTokenizer
                });

                this.compressor = result.promptCompressor;

                const initTime = Date.now() - startTime;
                this.logger.info(
                    { initializationTime: `${initTime}ms` },
                    '[LinguaCompressor] ✅ Compression model loaded successfully'
                );
            } catch (error) {
                const initTime = Date.now() - startTime;
                this.logger.error(
                    {
                        error,
                        initializationTime: `${initTime}ms`,
                        model: this.modelName
                    },
                    '[LinguaCompressor] ❌ Failed to initialize compression model'
                );
                this.initializationPromise = null; // Reset so we can retry
                throw error;
            }
        })();

        return this.initializationPromise;
    }

    /**
     * Compresses the given prompt using LLMLingua-2.
     * 
     * @param prompt The text prompt to compress
     * @param options Compression options:
     *   - rate: Compression ratio (0-1), default 0.5 = 50% compression
     *   - targetToken: Target token count (alternative to rate)
     *   - useSentenceLevel: Use sentence-level compression
     * @returns The compressed prompt string
     */
    public async compress(prompt: string, options: CompressionOptions = {}): Promise<string> {
        const startTime = Date.now();

        if (!this.compressor) {
            this.logger.debug('[LinguaCompressor] Compressor not initialized, initializing now...');
            await this.init();
        }

        // Safety check: Estimate token count (approx 4 chars per token)
        // TinyBERT model has a 512 token limit. If we exceed this, ONNX runtime will crash.
        // We use a conservative estimate of 1800 characters (~450 tokens) to be safe.
        // Chunking logic for long prompts
        if (prompt.length > 1800) {
            this.logger.info(
                { promptLength: prompt.length },
                '[LinguaCompressor] Prompt exceeds context window, splitting into chunks...'
            );

            // Split into chunks of ~1500 chars to be safe
            const chunkSize = 1500;
            const chunks: string[] = [];
            for (let i = 0; i < prompt.length; i += chunkSize) {
                chunks.push(prompt.substring(i, i + chunkSize));
            }

            this.logger.debug({ chunkCount: chunks.length }, '[LinguaCompressor] Processing chunks...');

            const compressedChunks = await Promise.all(chunks.map(async (chunk, index) => {
                try {
                    // Recursive call for each chunk
                    return await this.compress(chunk, options);
                } catch (error) {
                    this.logger.warn({ chunkIndex: index, error }, '[LinguaCompressor] Failed to compress chunk, using original');
                    return chunk;
                }
            }));

            return compressedChunks.join(' ');
        }

        try {
            const originalLength = prompt.length;

            // Map options to snake_case expected by the library
            const libraryOptions: any = {};
            if (options.rate !== undefined) libraryOptions.rate = options.rate;
            if (options.targetToken !== undefined) libraryOptions.target_token = options.targetToken;
            if (options.useSentenceLevel !== undefined) libraryOptions.use_sentence_level = options.useSentenceLevel;

            // Default to 50% compression if nothing specified
            if (libraryOptions.rate === undefined && libraryOptions.target_token === undefined) {
                libraryOptions.rate = 0.5;
            }

            this.logger.debug(
                {
                    originalLength,
                    compressionRate: libraryOptions.rate || 'auto',
                    targetToken: libraryOptions.target_token || 'none'
                },
                '[LinguaCompressor] Starting compression...'
            );

            const compressed = await this.compressor.compress_prompt(prompt, libraryOptions);

            const compressionTime = Date.now() - startTime;
            const compressedLength = compressed.length;
            const actualRatio = (compressedLength / originalLength);
            const tokenSavings = Math.round((1 - actualRatio) * 100);

            this.logger.info(
                {
                    originalLength,
                    compressedLength,
                    ratio: actualRatio.toFixed(2),
                    tokenSavings: `${tokenSavings}%`,
                    compressionTime: `${compressionTime}ms`
                },
                '[LinguaCompressor] ✅ Compression complete'
            );

            return compressed;
        } catch (error) {
            const compressionTime = Date.now() - startTime;
            this.logger.error(
                {
                    error,
                    compressionTime: `${compressionTime}ms`,
                    promptLength: prompt.length
                },
                '[LinguaCompressor] ❌ Compression failed, falling back to original prompt'
            );
            // CRITICAL FIX: Return original prompt instead of throwing, to prevent app crash
            return prompt;
        }
    }
}

// Export a singleton instance
export const linguaCompressor = LinguaCompressor.getInstance();
