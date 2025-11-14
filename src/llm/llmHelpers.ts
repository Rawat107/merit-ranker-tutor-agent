/**
 * Shared LLM helper utilities to avoid code duplication
 * Used by BasicLLM, IntermediateLLM, and AdvancedLLM
 */

/**
 * Extract text content from LangChain response
 * Handles both string and array of content blocks
 */
export function extractContent(responseContent: any): string {
  if (typeof responseContent === 'string') {
    return responseContent;
  }
  
  if (Array.isArray(responseContent)) {
    return responseContent
      .map(block => {
        if (typeof block === 'string') return block;
        if (block && typeof block === 'object' && 'text' in block) return block.text;
        return JSON.stringify(block);
      })
      .join('');
  }
  
  return JSON.stringify(responseContent);
}

/**
 * Extract text content from streaming chunk
 * Returns empty string for non-text chunks
 */
export function extractStreamChunkContent(chunkContent: any): string {
  if (typeof chunkContent === 'string') {
    return chunkContent;
  }
  
  if (Array.isArray(chunkContent)) {
    return chunkContent
      .map(block => {
        if (typeof block === 'string') return block;
        if (block && typeof block === 'object' && 'text' in block) return block.text;
        return '';
      })
      .join('');
  }
  
  return '';
}
