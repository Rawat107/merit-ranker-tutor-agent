/**
 * Response Quality Validator
 * Detects low-quality, failed, or unhelpful responses that shouldn't be cached
 */

export interface ValidationResult {
  isValid: boolean;
  reason?: string;
  score: number; // 0-1, where 1 is highest quality
}

/**
 * Validates if a response is worth caching
 * Returns false for failed/unhelpful responses
 */
export function validateResponse(response: string): ValidationResult {
  const trimmed = response.trim().toLowerCase();
  
  // Empty or too short responses
  if (trimmed.length < 10) {
    return {
      isValid: false,
      reason: 'Response too short (less than 10 characters)',
      score: 0
    };
  }

  // Check for failure indicators (case-insensitive)
  const failurePatterns = [
    // Direct refusals (first-person only - "I cannot")
    /unfortunately[,\s]+i\s+(don't|do not|cannot|can't|am unable)/i,
    /i\s+(don't|do not|cannot|can't)\s+have\s+(enough\s+)?information/i,
    /i\s+(cannot|can't|am unable to)\s+help\s+with/i,
    /i\s+(cannot|can't|am unable to)\s+(answer|provide|assist)/i,
    
    // Apologies for inability (first-person)
    /sorry[,\s]+(but\s+)?i\s+(don't|cannot|can't)/i,
    /apologies[,\s]+(but\s+)?i\s+(don't|cannot|can't)/i,
    /i apologize[,\s]+(but\s+)?i\s+(don't|cannot|can't)/i,
    
    // Not enough context (first-person)
    /i\s+need\s+more\s+(information|context|details)/i,
    /could\s+you\s+please\s+provide\s+more/i,
    /insufficient\s+information/i,
    
    // Out of scope (first-person)
    /outside\s+(my|the)\s+(scope|domain|expertise)/i,
    /not\s+within\s+my\s+(capabilities|scope)/i,
    /i\s+am\s+not\s+(designed|built|trained)\s+to/i,
    
    // System/processing errors (NOT user-facing error explanations)
    /^error\s+occurred/i,
    /^something\s+went\s+wrong/i,
    /^failed\s+to\s+(process|generate|retrieve)/i,
    
    // Empty or placeholder responses
    /^(n\/a|na|null|undefined|none)$/i,
    /^i\s+don'?t\s+know\.?$/i,
    /^no\s+(answer|response)\.?$/i,
  ];

  // Test against all failure patterns
  for (const pattern of failurePatterns) {
    if (pattern.test(trimmed)) {
      return {
        isValid: false,
        reason: `Matched failure pattern: ${pattern.source}`,
        score: 0.2
      };
    }
  }

  // Check for very generic unhelpful responses
  const unhelpfulPatterns = [
    /^(yes|no|maybe|ok|okay)\.?$/i,
    /^i\s+understand\.?$/i,
    /^got\s+it\.?$/i,
  ];

  for (const pattern of unhelpfulPatterns) {
    if (pattern.test(trimmed)) {
      return {
        isValid: false,
        reason: `Response too generic: ${pattern.source}`,
        score: 0.3
      };
    }
  }

  // Quality indicators (positive signals)
  const qualityIndicators = [
    /\b(answer|solution|explanation|because|therefore|steps?|formula)\b/i,
    /\b(first|second|third|finally|conclusion)\b/i,
    /[0-9]+\.|step\s+[0-9]+/i, // Numbered steps
    /\$\$.+\$\$|\\\(.+\\\)/s, // LaTeX formulas
    /\*\*[^*]+\*\*/i, // Bold markdown
    /```/i, // Code blocks
  ];

  let qualityScore = 0.5; // Base score for passing initial checks
  
  for (const indicator of qualityIndicators) {
    if (indicator.test(response)) {
      qualityScore += 0.1;
    }
  }

  // Cap at 1.0
  qualityScore = Math.min(qualityScore, 1.0);

  // Response is valid if it has decent quality
  const isValid = qualityScore >= 0.5;

  return {
    isValid,
    reason: isValid ? undefined : 'Low quality score',
    score: qualityScore
  };
}

/**
 * Quick check - just returns boolean
 */
export function isResponseValid(response: string): boolean {
  return validateResponse(response).isValid;
}

/**
 * Get a human-readable validation report
 */
export function getValidationReport(response: string): string {
  const result = validateResponse(response);
  
  if (result.isValid) {
    return `✓ Valid response (score: ${result.score.toFixed(2)})`;
  } else {
    return `✗ Invalid response (score: ${result.score.toFixed(2)}): ${result.reason}`;
  }
}
