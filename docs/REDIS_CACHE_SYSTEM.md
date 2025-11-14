# Redis Caching System with LangChain + AWS Bedrock

## Overview

This implementation provides a comprehensive caching solution using **LangChain Redis** patterns with **AWS Bedrock Titan embeddings** for semantic similarity matching.

## Features

### 1. **Exact Match Cache** (Traditional Key-Value)

- **Purpose**: Lightning-fast O(1) lookups for identical queries
- **Storage**: Redis key-value pairs
- **TTL**: 24 hours
- **Use Case**: Repeated identical questions

### 2. **Semantic Cache** (Embedding-based Similarity)

- **Purpose**: Match semantically similar queries even with different wording
- **Embeddings**: AWS Bedrock Titan (`amazon.titan-embed-text-v1`)
- **Similarity Threshold**: 0.94 (high confidence)
- **Storage**: Redis with vector embeddings
- **TTL**: 24 hours
- **Use Case**: "What is the capital of France?" matches "Tell me France's capital"

### 3. **Chat History** (Conversation Context)

- **Purpose**: Maintain conversation context for follow-up questions
- **Storage**: Last 2-3 turns (6 messages total)
- **TTL**: 24 hours
- **Use Case**: Context-aware responses in multi-turn conversations

## Architecture

```
┌─────────────────────────────────────────────────────┐
│           UnifiedCacheManager                        │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌────────────────────┐  ┌────────────────────┐   │
│  │ RedisLangChainCache│  │RedisLangChainHistory│   │
│  ├────────────────────┤  ├────────────────────┤   │
│  │ - Exact Match      │  │ - Session Tracking  │   │
│  │ - Semantic Match   │  │ - Message History   │   │
│  │ - Quality Scoring  │  │ - Context Format    │   │
│  └────────────────────┘  └────────────────────┘   │
│                                                      │
└─────────────────────────────────────────────────────┘
                          │
                          ▼
                    Redis Server
              (Exact + Semantic Storage)
```

## Cache Lookup Strategy

1. **Check Exact Match** (fastest - ~1ms)
   - Direct key-value lookup
   - Returns immediately if found

2. **Check Semantic Cache** (slower - ~100ms)
   - Generate query embedding using Bedrock Titan
   - Compare with all stored embeddings for subject
   - Return if similarity > 0.94

3. **Cache Miss**
   - Generate fresh response
   - Store in both caches
   - Auto-cleanup maintains only top 3 responses per subject

## Configuration

### Environment Variables

```env
# Redis Configuration
REDIS_URL=redis://localhost:6379
REDIS_PASSWORD=your_redis_password

# AWS Bedrock for Embeddings
AWS_REGION=ap-south-1
AWS_ACCESS_KEY_ID=your_access_key
AWS_SECRET_ACCESS_KEY=your_secret_key
```

### Cache Settings

```typescript
TTL: 86400 seconds (24 hours)
Max Entries Per Subject: 3
Semantic Threshold: 0.94 (94% similarity)
```

## API Endpoints

### Get Cache Statistics

```bash
GET http://localhost:3000/cache/stats
```

**Response:**

```json
{
  "success": true,
  "data": {
    "cache": {
      "exactCache": { "entries": 15 },
      "semanticCache": { "entries": 15 },
      "subjects": [
        {
          "subject": "math",
          "entries": 3,
          "topScores": ["0.95", "key123", "0.92", "key456"]
        }
      ],
      "ttl": 86400,
      "maxEntriesPerSubject": 3
    },
    "chatHistory": {
      "activeSessions": 5
    },
    "timestamp": "2025-11-14T..."
  }
}
```

### Clear All Caches

```bash
POST http://localhost:3000/cache/clear
```

**Response:**

```json
{
  "success": true,
  "message": "All caches cleared successfully"
}
```

## Usage Example

### Automatic Caching (Built-in)

The cache is automatically used in the evaluation pipeline:

1. **User asks**: "What is 2+2?"
2. **System checks**: Exact cache → Semantic cache
3. **If miss**: Generate response, store in both caches
4. **If hit**: Serve cached response instantly

### Cache Hit Logs

```
[RedisCache] ✅ Exact match cache HIT
  query: "What is 2+2?"
  subject: "math"
  hitCount: 3
  lookupTime: 2ms

[RedisCache] ✅ Semantic cache HIT
  query: "What's two plus two?"
  matchedQuery: "What is 2+2?"
  similarity: 0.967
  lookupTime: 89ms
```

## Benefits

### Performance

- **Exact Match**: ~2ms response time
- **Semantic Match**: ~90ms (vs 3-6 seconds for LLM call)
- **95%+ faster** for cached queries

### Cost Reduction

- Eliminates redundant LLM API calls
- Reduces token usage
- Lower Bedrock inference costs

### Quality Improvements

- Maintains best responses (quality scoring)
- Auto-cleanup removes low-quality answers
- Consistent responses for similar questions

## Storage Structure

### Redis Keys

```
tutor:exact:{hash}              # Exact match cache
tutor:semantic:{subject}:{hash} # Semantic cache entries
tutor:subject:{subject}         # Quality-sorted set for cleanup
tutor:chat:{sessionId}          # Chat history
```

### Cache Entry Structure

**Exact Match:**

```json
{
  "query": "What is 2+2?",
  "response": "2+2 equals 4...",
  "subject": "math",
  "level": "basic",
  "confidence": 0.95,
  "modelUsed": "claude-3-sonnet",
  "timestamp": 1731600000000,
  "hitCount": 5
}
```

**Semantic Match:**

```json
{
  "query": "What is 2+2?",
  "response": "2+2 equals 4...",
  "embedding": [0.123, 0.456, ...],  // 1536 dimensions
  "subject": "math",
  "level": "basic",
  "confidence": 0.95,
  "modelUsed": "claude-3-sonnet",
  "timestamp": 1731600000000,
  "hitCount": 3,
  "qualityScore": 0.95
}
```

## Monitoring

### Key Metrics

1. **Cache Hit Rate**
   - Exact: `hits / (hits + misses)`
   - Semantic: `semantic_hits / total_lookups`

2. **Storage Usage**
   - Total entries per subject
   - Memory consumption

3. **Latency**
   - Exact lookup: <5ms
   - Semantic lookup: 50-150ms
   - Fresh generation: 3000-6000ms

### Health Checks

```bash
# Check Redis connectivity
redis-cli -h localhost -p 6379 PING

# Monitor cache size
redis-cli -h localhost -p 6379 DBSIZE

# View cache keys
redis-cli -h localhost -p 6379 KEYS "tutor:*"
```

## Troubleshooting

### Issue: Semantic cache not working

**Solution:**

1. Verify AWS credentials are set
2. Check Bedrock Titan model access
3. Ensure embedding generation succeeds

### Issue: Cache not expiring

**Solution:**

1. Verify TTL is set: `redis-cli TTL tutor:exact:{key}`
2. Check Redis configuration for eviction policy
3. Restart Redis if needed

### Issue: Too many cache entries

**Solution:**

- Auto-cleanup maintains max 3 per subject
- Manually clear: `POST /cache/clear`
- Adjust `maxEntriesPerSubject` in config

## Future Enhancements

1. **Redis Vector Search** - Native vector similarity (when available)
2. **Cache Warming** - Pre-populate common queries
3. **User Feedback** - Update quality scores based on user ratings
4. **Analytics Dashboard** - Visual cache performance metrics
5. **Multi-Region Replication** - Distributed cache for global users

## Dependencies

```json
{
  "@langchain/redis": "^latest",
  "@langchain/aws": "^1.0.0",
  "ioredis": "^5.3.2",
  "redis": "^latest"
}
```

## License

Part of the AI Tutor Service project.
