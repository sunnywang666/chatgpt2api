#requires -Version 7.0
[CmdletBinding()]
param(
    [string]$Model,
    [string]$BaseUrl = "https://app.hugsweetglobal.com/ai/codex/v1",
    [string]$StateRoot = (Join-Path $PSScriptRoot ".codex-client-state"),
    [string]$FixtureRoot = (Join-Path $PSScriptRoot "acceptance-fixture"),
    [string]$Resume,
    [switch]$ListModels,
    [string]$Prompt = "Fix only calculator.py so test_calculator.py passes. Run python -m unittest -v and report the result."
)

$ErrorActionPreference = "Stop"
if ([string]::IsNullOrWhiteSpace($env:CODEX_PROVIDER_API_KEY)) {
    throw "Set CODEX_PROVIDER_API_KEY in this PowerShell session. It is injected only into the Codex child."
}

$python = Get-Command python -ErrorAction SilentlyContinue
if ($null -eq $python) { $python = Get-Command py -ErrorAction Stop }
$pythonArgs = if ($python.Name -like "py*") { @("-3") } else { @() }
$modelArgs = @()
if ($Model) { $modelArgs = @("--model", $Model) }

& $python.Source @pythonArgs (Join-Path $PSScriptRoot "list_models.py") --base-url $BaseUrl @modelArgs
if ($LASTEXITCODE -ne 0 -or $ListModels) { exit $LASTEXITCODE }
if ([string]::IsNullOrWhiteSpace($Model)) { throw "-Model is required after discovery; select an exact ID returned by GET /models." }
& $python.Source @pythonArgs (Join-Path $PSScriptRoot "prepare_acceptance.py") --state-root $StateRoot --fixture-root $FixtureRoot --base-url $BaseUrl --model $Model | Out-Null
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

# launch_codex.py reads the key from this process environment, removes that
# variable, and execs Codex with a fresh child environment. The key is never in
# a PowerShell argument list or Codex argv.
$launchArgs = @("--state-root", $StateRoot, "--fixture-root", $FixtureRoot, "--model", $Model, "--prompt", $Prompt)
if ($Resume) { $launchArgs += @("--resume", $Resume) }
& $python.Source @pythonArgs (Join-Path $PSScriptRoot "launch_codex.py") @launchArgs
exit $LASTEXITCODE
