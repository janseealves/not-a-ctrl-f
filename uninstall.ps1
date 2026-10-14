param([switch]$RemoveData)
$ErrorActionPreference = "Stop"

$exe = $null
try { $exe = Join-Path (uv tool dir --bin | Select-Object -First 1) "not-a-ctrl-f.exe" } catch {}

if ($exe -and (Test-Path $exe)) {
    foreach ($args_ in @(@("stop"), @("autostart", "off"))) {
        try { & $exe @args_ } catch { Write-Host "aviso: $($args_ -join ' ') falhou: $_" }
    }
}
try { uv tool uninstall not-a-ctrl-f } catch { Write-Host "aviso: uv tool uninstall falhou: $_" }

if ($RemoveData) {
    $dir = Join-Path $env:LOCALAPPDATA "not-a-ctrl-f"
    $ans = Read-Host "Apagar $dir (índice, modelos e llama.cpp)? [s/N]"
    if ($ans -match '^(s|sim|y|yes)$') {
        Remove-Item -Recurse -Force $dir -ErrorAction SilentlyContinue
        Write-Host "Dados removidos."
    } else {
        Write-Host "Dados mantidos."
    }
}
Write-Host "Desinstalado."
