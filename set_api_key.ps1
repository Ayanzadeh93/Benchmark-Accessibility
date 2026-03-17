<# 
Set the OpenAI API key for the current session.

Usage:
  1. Set the OPENAI_API_KEY environment variable outside of git-tracked files, e.g.:
       - In PowerShell profile
       - In a local .env file (ignored by git)
  2. Optionally call this script to echo the current key status.

This script intentionally does NOT contain any real API keys.
#>

if (-not $env:OPENAI_API_KEY) {
    Write-Warning "OPENAI_API_KEY is not set. Please set it in your environment (not in git-tracked files) and re-run."
} else {
    Write-Host "OPENAI_API_KEY is set for this session."
    Write-Host "Key length: $($env:OPENAI_API_KEY.Length) characters"
}
