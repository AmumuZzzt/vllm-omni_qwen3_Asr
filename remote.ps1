# -- vllm4090 remote exec helper -------------------------------------------------
# Pipes a bash script (LF-normalized) into `bash -s` on the remote host.
# Usage:
#   .\remote.ps1 -Command 'nvidia-smi'
#   .\remote.ps1 -File .\scripts\bench.sh
#   'df -h /' | .\remote.ps1
#
# Why: Windows OpenSSH 8.1 mangles nested quotes, and PowerShell here-strings use
# CRLF -> remote bash sees "/root/vllm-work\r" and fails. We normalize to LF here.

param(
    [string]$Command,
    [string]$File,
    [string]$HostAlias = "vllm4090",
    [switch]$Quiet
)

$ErrorActionPreference = "Stop"

if ($File) {
    if (-not (Test-Path $File)) { throw "script not found: $File" }
    # explicit UTF8: PowerShell 5.1 defaults to ANSI and would mangle CJK/UTF-8
    $script = Get-Content -LiteralPath $File -Raw -Encoding UTF8
} elseif ($Command) {
    $script = $Command
} elseif ($input) {
    $script = ($input | Out-String)
} else {
    $script = ""
}

if ([string]::IsNullOrWhiteSpace($script)) { throw "empty script" }

# CRLF -> LF, strip BOM, guarantee trailing newline
$script = $script -replace "`r`n", "`n" -replace "`r", "`n"
$script = $script.TrimStart([char]0xFEFF)
if (-not $script.EndsWith("`n")) { $script += "`n" }

$bytes = [System.Text.Encoding]::UTF8.GetBytes($script)
$psi = New-Object System.Diagnostics.ProcessStartInfo
$psi.FileName = "ssh"
$psi.Arguments = "-o BatchMode=yes -o ConnectTimeout=20 $HostAlias `"bash -s`""
$psi.RedirectStandardInput = $true
$psi.RedirectStandardOutput = $true
$psi.RedirectStandardError = $true
$psi.UseShellExecute = $false

$p = [System.Diagnostics.Process]::Start($psi)
$p.StandardInput.BaseStream.Write($bytes, 0, $bytes.Length)
$p.StandardInput.Close()
$out = $p.StandardOutput.ReadToEnd()
$err = $p.StandardError.ReadToEnd()
$p.WaitForExit()

if (-not $Quiet) {
    if ($out) { Write-Output $out.TrimEnd() }
    if ($err) { Write-Error $err.TrimEnd() -ErrorAction Continue }
}
exit $p.ExitCode
