# Test Response Validation System
# This script tests various responses to verify the validation logic

Write-Host "=== Response Validation Test ===" -ForegroundColor Cyan
Write-Host ""

# Test cases with expected results
$testCases = @(
    @{
        Name = "Valid Response - Detailed Answer"
        Response = "To solve this quadratic equation, follow these steps:\n\n1. First, identify coefficients: a=2, b=5, c=3\n2. Apply the quadratic formula: x = (-b ± √(b²-4ac)) / 2a\n3. Calculate: x = (-5 ± √(25-24)) / 4\n4. Final Answer: x = -1 or x = -1.5"
        Expected = "VALID"
    },
    @{
        Name = "Invalid - Unfortunately Response"
        Response = "Unfortunately, I don't have enough information to answer your question. Could you please provide more context?"
        Expected = "INVALID"
    },
    @{
        Name = "Invalid - Cannot Help"
        Response = "I cannot help with that topic as it's outside my scope."
        Expected = "INVALID"
    },
    @{
        Name = "Invalid - Sorry Apology"
        Response = "Sorry, but I can't provide an answer to that question."
        Expected = "INVALID"
    },
    @{
        Name = "Valid Response - Math Explanation"
        Response = "The derivative of f(x) = x² is calculated using the power rule:\n\nf'(x) = 2x\n\nBecause we bring down the exponent and reduce it by 1."
        Expected = "VALID"
    },
    @{
        Name = "Invalid - Too Short"
        Response = "Yes."
        Expected = "INVALID"
    },
    @{
        Name = "Invalid - Need More Info"
        Response = "I need more information to answer this properly."
        Expected = "INVALID"
    },
    @{
        Name = "Valid Response - With LaTeX"
        Response = "The quadratic formula is: \$\$x = \frac{-b \pm \sqrt{b^2 - 4ac}}{2a}\$\$ This gives us the roots of any quadratic equation."
        Expected = "VALID"
    }
)

Write-Host "Testing $($testCases.Count) response scenarios..." -ForegroundColor Yellow
Write-Host ""

foreach ($test in $testCases) {
    Write-Host "Test: $($test.Name)" -ForegroundColor White
    Write-Host "Expected: $($test.Expected)" -ForegroundColor Gray
    
    $body = @{
        response = $test.Response
    } | ConvertTo-Json
    
    # Note: This requires a test endpoint that uses validateResponse()
    # For now, this is a template - you'd need to add an endpoint like /test/validate
    
    Write-Host "Response Preview: $($test.Response.Substring(0, [Math]::Min(60, $test.Response.Length)))..." -ForegroundColor DarkGray
    Write-Host ""
}

Write-Host "=== Test Complete ===" -ForegroundColor Cyan
Write-Host ""
Write-Host "NOTE: The actual validation happens in the server during caching." -ForegroundColor Yellow
Write-Host "Check server logs for validation results:" -ForegroundColor Yellow
Write-Host "  - 'Response passed quality check' (with score)" -ForegroundColor Green
Write-Host "  - 'Skipped caching - response failed quality check' (with reason)" -ForegroundColor Red
Write-Host ""
Write-Host "Test a real conversation to see validation in action:" -ForegroundColor Cyan
Write-Host '  curl -X POST http://localhost:3000/evaluate -H "Content-Type: application/json" -d "{\"message\":\"test\",\"sessionId\":\"test-123\"}"' -ForegroundColor Gray
