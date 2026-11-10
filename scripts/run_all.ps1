param(
    [string]$Python = 'python',
    [int]$GeneratorWorkers = 1,
    [int]$ModelJobs = 2,
    [int]$NoPairJobs = 1,
    [switch]$DryRun,
    [switch]$SkipOfficial
)
$ErrorActionPreference = 'Stop'
$arguments = @('--workers', $GeneratorWorkers, '--model-jobs', $ModelJobs, '--no-pair-jobs', $NoPairJobs)
if ($DryRun) { $arguments += '--dry-run' }
if ($SkipOfficial) { $arguments += '--skip-official' }
& $Python (Join-Path $PSScriptRoot 'run_all.py') @arguments
if ($LASTEXITCODE -ne 0) { throw "Pipeline failed with exit code $LASTEXITCODE" }
