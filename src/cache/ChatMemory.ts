import { RedisClientType } from 'redis';
import pino from 'pino';

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  timestamp?: number;
}

/**
 * Clean, simple chat memory manager using Redis
 * 
  * STORAGE BEHAVIOR:
 * - Messages accumulate in Redis LIST until 24h TTL expires
 * - NO automatic trimming/dequeuing (old messages stay until TTL)
 * - Only last N messages loaded into prompts (default: 10)
 * - TTL refreshes on each save (24h from last activity)
 * 
 * MEMORY MANAGEMENT:
 * - For long sessions (100+ messages), consider periodic cleanup
 * - Use trim() method to keep list size manageable
 * - Monitor Redis memory usage for active sessions
 */
export class ChatMemory {
  private client: RedisClientType;
  private logger: pino.Logger;
  private readonly ttl = 86400; // 24 hours
  private readonly maxStoredMessages = 5; // Keep only last 5 messages total

  constructor(client: RedisClientType, logger: pino.Logger) {
    this.client = client;
    this.logger = logger;
  }

  /**
   * Load conversation history for a user
   * @param userId - Unique user/session identifier
   * @param limit - Maximum number of messages to retrieve (default: 10 = last 5 exchanges)
   */
  async load(userId: string, limit: number = 5): Promise<ChatMessage[]> {
    try {
      const key = `history:${userId}`;
      const rawMessages = await this.client.lRange(key, -limit, -1);
      
      const messages = rawMessages.map((raw) => {
        try {
          return JSON.parse(raw) as ChatMessage;
        } catch (e) {
          this.logger.warn({ raw, error: e }, 'Failed to parse message, skipping');
          return null;
        }
      }).filter((msg): msg is ChatMessage => msg !== null);

      this.logger.debug({ userId, count: messages.length }, 'Loaded chat history');
      return messages;
    } catch (error) {
      this.logger.error({ error, userId }, 'Failed to load chat history');
      return [];
    }
  }

  /**
   * Save a new exchange (user message + assistant response)
   * Automatically trims history to maxStoredMessages to prevent unbounded growth
   * 
   * @param userId - Unique user/session identifier
   * @param userMsg - User's input message
   * @param botMsg - Assistant's response
   */
  async save(userId: string, userMsg: string, botMsg: string): Promise<void> {
    try {
      const key = `history:${userId}`;
      const timestamp = Date.now();

      const userMessage: ChatMessage = { 
        role: 'user', 
        content: userMsg,
        timestamp 
      };
      
      const assistantMessage: ChatMessage = { 
        role: 'assistant', 
        content: botMsg,
        timestamp: timestamp + 1
      };

      await this.client.rPush(key, JSON.stringify(userMessage));
      await this.client.rPush(key, JSON.stringify(assistantMessage));
      
      // Auto-trim to prevent unbounded growth
      const length = await this.client.lLen(key);
      if (length > this.maxStoredMessages) {
        await this.client.lTrim(key, -this.maxStoredMessages, -1);
        this.logger.debug(
          { userId, trimmedTo: this.maxStoredMessages, previousLength: length }, 
          'Auto-trimmed chat history'
        );
      }
      
      // Set expiry to 24 hours from last activity
      await this.client.expire(key, this.ttl);

      this.logger.debug({ userId, totalMessages: Math.min(length + 2, this.maxStoredMessages) }, 'Saved chat exchange to history');
    } catch (error) {
      this.logger.error({ error, userId }, 'Failed to save chat history');
    }
  }

  /**
   * Format messages into a readable conversation history string
   * @param messages - Array of chat messages
   * @param extractName - If true, attempts to extract user's name from messages
   * @returns Formatted history string and optionally extracted name
   */
  format(messages: ChatMessage[], extractName: boolean = false): { 
    historyText: string; 
    userName: string | null;
  } {
    let userName: string | null = null;

    const formattedLines = messages.map((msg) => {
      const role = msg.role === 'user' ? 'User' : 'Assistant';
      
      // Extract name if requested and not yet found
      if (extractName && msg.role === 'user' && !userName) {
        const nameMatch = msg.content.match(/(?:my name is|I am|I'm|call me)\s+([A-Z][a-z]{1,50})/i);
        if (nameMatch) {
          userName = nameMatch[1];
        }
      }

      return `${role}: ${msg.content}`;
    });

    return {
      historyText: formattedLines.join('\n'),
      userName
    };
  }

  /**
   * Clear conversation history for a user
   * @param userId - Unique user/session identifier
   */
  async clear(userId: string): Promise<void> {
    try {
      const key = `history:${userId}`;
      await this.client.del(key);
      this.logger.info({ userId }, 'Cleared chat history');
    } catch (error) {
      this.logger.error({ error, userId }, 'Failed to clear chat history');
    }
  }

  /**
   * Get the number of messages in history for a user
   * @param userId - Unique user/session identifier
   */
  async count(userId: string): Promise<number> {
    try {
      const key = `history:${userId}`;
      return await this.client.lLen(key);
    } catch (error) {
      this.logger.error({ error, userId }, 'Failed to count chat history');
      return 0;
    }
  }

  /**
   * Manually trim history to a specific size
   * Useful for cleanup or custom memory management
   * 
   * @param userId - Unique user/session identifier
   * @param maxMessages - Maximum number of messages to keep
   */
  async trim(userId: string, maxMessages: number = 20): Promise<void> {
    try {
      const key = `history:${userId}`;
      const length = await this.client.lLen(key);
      
      if (length > maxMessages) {
        await this.client.lTrim(key, -maxMessages, -1);
        this.logger.info(
          { userId, trimmedFrom: length, trimmedTo: maxMessages }, 
          'Manually trimmed chat history'
        );
      }
    } catch (error) {
      this.logger.error({ error, userId }, 'Failed to trim chat history');
    }
  }
}
