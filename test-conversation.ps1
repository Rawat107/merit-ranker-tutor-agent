# Test conversation flow with memory
$baseUrl = "http://localhost:3000"
$sessionId = [guid]::NewGuid().ToString()

Write-Host "Testing conversation with sessionId: $sessionId" -ForegroundColor Cyan
Write-Host ""

# Message 1: Introduce myself
Write-Host "=== Message 1: Introducing myself ===" -ForegroundColor Yellow
$body1 = @{
    message = "Hi! My name is Vaibhav and I'm preparing for SSC CGL exam"
    sessionId = $sessionId
} | ConvertTo-Json

$response1 = Invoke-RestMethod -Uri "$baseUrl/chat" -Method POST -Body $body1 -ContentType "application/json"
Write-Host "Classification: $($response1.classification.subject) - $($response1.classification.level)" -ForegroundColor Green
Write-Host "Sources found: $($response1.sources.Count)" -ForegroundColor Green
Write-Host ""

# Message 2: Ask about my name
Write-Host "=== Message 2: Asking 'What is my name?' ===" -ForegroundColor Yellow
Start-Sleep -Seconds 2

$body2 = @{
    message = "What is my name?"
    sessionId = $sessionId
} | ConvertTo-Json

$response2 = Invoke-RestMethod -Uri "$baseUrl/chat" -Method POST -Body $body2 -ContentType "application/json"
Write-Host "Classification: $($response2.classification.subject) - $($response2.classification.level)" -ForegroundColor Green
Write-Host "Sources found: $($response2.sources.Count)" -ForegroundColor Green
Write-Host ""

# Message 3: Ask about previous topic
Write-Host "=== Message 3: Asking 'What exam am I preparing for?' ===" -ForegroundColor Yellow
Start-Sleep -Seconds 2

$body3 = @{
    message = "What exam am I preparing for?"
    sessionId = $sessionId
} | ConvertTo-Json

$response3 = Invoke-RestMethod -Uri "$baseUrl/chat" -Method POST -Body $body3 -ContentType "application/json"
Write-Host "Classification: $($response3.classification.subject) - $($response3.classification.level)" -ForegroundColor Green
Write-Host "Sources found: $($response3.sources.Count)" -ForegroundColor Green
Write-Host ""

Write-Host "Conversation test complete! Check server logs to see:" -ForegroundColor Cyan
Write-Host "  - Conversation history being loaded" -ForegroundColor White
Write-Host "  - User name extraction (should show 'Vaibhav')" -ForegroundColor White
Write-Host "  - Message count increasing with each request" -ForegroundColor White
Write-Host ""
Write-Host "To see the actual answers, use the /evaluate/stream endpoint with the retrieved documents." -ForegroundColor Gray
