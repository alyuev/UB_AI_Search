# Runs INSIDE Windows Sandbox (a clean Windows): copies the distribution like a user would,
# starts SmartSearch.exe and writes a report to the mapped results folder.
$ErrorActionPreference = 'Continue'
$src = 'C:\dist\BilingualTextAnalyzer'
$app = 'C:\Users\WDAGUtilityAccount\Desktop\BilingualTextAnalyzer'
$out = 'C:\results\report.txt'
function Log($s) { Add-Content -Path $out -Value $s -Encoding utf8 }
Set-Content -Path $out -Value "=== sandbox test $(Get-Date)" -Encoding utf8

Log "Windows: $((Get-CimInstance Win32_OperatingSystem).Caption) $((Get-CimInstance Win32_OperatingSystem).Version)"
$wv = Get-ItemProperty 'HKLM:\SOFTWARE\WOW6432Node\Microsoft\EdgeUpdate\Clients\{F3017226-FE2A-4295-8BDF-00C3A9A7E4C5}' -ErrorAction SilentlyContinue
Log "WebView2: $(if ($wv) { $wv.pv } else { 'NOT INSTALLED' })"
Log "msvcp140 in System32: $(Test-Path C:\Windows\System32\msvcp140.dll)"
Log "vulkan-1 in System32: $(Test-Path C:\Windows\System32\vulkan-1.dll)"

Log "copying..."
Copy-Item $src $app -Recurse
Log "copied: $((Get-ChildItem $app -Recurse | Measure-Object Length -Sum).Sum / 1MB) MB"

# llama-server alone: proves its DLLs load on a clean system
$ls = Join-Path $app 'tools\llama.cpp\bin\llama-server.exe'
$p = Start-Process $ls -ArgumentList '--version' -NoNewWindow -Wait -PassThru -RedirectStandardError C:\results\llama_version.txt
Log "llama-server --version exit code: $($p.ExitCode) (0 = DLLs OK)"

Start-Process (Join-Path $app 'SmartSearch.exe')
Start-Sleep 45
$py = Get-Process pythonw -ErrorAction SilentlyContinue
Log "pythonw running: $([bool]$py); window title: '$($py.MainWindowTitle -join ',')'"

$port = $null
foreach ($pp in 8765..8775) {
  try { $r = Invoke-WebRequest "http://127.0.0.1:$pp/api/status" -UseBasicParsing -TimeoutSec 5; $port = $pp; break } catch {}
}
Log "server port: $port"
if ($port) {
  foreach ($mode in 'fts', 'semantic') {
    try {
      $r = Invoke-WebRequest "http://127.0.0.1:$port/api/search?q=love&mode=$mode&k=3&history=0" -UseBasicParsing -TimeoutSec 180
      Log "search ${mode}: HTTP $($r.StatusCode), $($r.Content.Length) bytes"
    } catch { Log "search ${mode}: FAILED $($_.Exception.Message)" }
  }
}
Add-Type -AssemblyName System.Windows.Forms, System.Drawing
$b = [System.Windows.Forms.Screen]::PrimaryScreen.Bounds
$bmp = New-Object System.Drawing.Bitmap $b.Width, $b.Height
[System.Drawing.Graphics]::FromImage($bmp).CopyFromScreen($b.Location, [System.Drawing.Point]::Empty, $b.Size)
$bmp.Save('C:\results\screen.png')
Copy-Item (Join-Path $app 'logs\*') C:\results\ -ErrorAction SilentlyContinue
Log "=== done"
