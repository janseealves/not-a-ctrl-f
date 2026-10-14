param(
    [ValidateSet("auto", "cuda", "vulkan", "cpu")][string]$Backend = "auto",
    [switch]$Autostart,
    [string]$Source
)
$ErrorActionPreference = "Stop"

function Fail($msg) {
    Write-Host "ERRO: $msg" -ForegroundColor Red
    exit 1
}

# 1. uv
Write-Host "[1/5] Verificando o uv..."
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "      uv não encontrado; instalando..."
    irm https://astral.sh/uv/install.ps1 | iex
    $env:Path = "$env:USERPROFILE\.local\bin;$env:Path"
    if (-not (Get-Command uv -ErrorAction SilentlyContinue)) { Fail "não foi possível instalar o uv" }
}

# 2. fonte
if (-not $Source) {
    if ($PSScriptRoot -and (Test-Path (Join-Path $PSScriptRoot "pyproject.toml"))) {
        $Source = $PSScriptRoot
    } else {
        $Source = "https://github.com/janseealves/not-a-ctrl-f/archive/refs/heads/main.zip"
    }
}
Write-Host "[2/5] Fonte: $Source"

# 3. instalar o app
Write-Host "[3/5] Instalando o not-a-ctrl-f..."
uv tool install --force --python 3.14 $Source
if ($LASTEXITCODE -ne 0) { Fail "uv tool install falhou (código $LASTEXITCODE)" }
uv tool update-shell
if ($LASTEXITCODE -ne 0) { Fail "uv tool update-shell falhou (código $LASTEXITCODE)" }

# 4. executável
$bin = (uv tool dir --bin | Select-Object -First 1)
if ($LASTEXITCODE -ne 0 -or -not $bin) { Fail "não foi possível localizar a pasta de executáveis do uv" }
$exe = Join-Path $bin "not-a-ctrl-f.exe"
if (-not (Test-Path $exe)) { Fail "executável não encontrado: $exe" }

# 5. componentes e início
Write-Host "[4/5] Baixando llama.cpp e modelos (backend: $Backend)..."
& $exe setup --backend $Backend
if ($LASTEXITCODE -ne 0) { Fail "setup falhou (código $LASTEXITCODE)" }

Write-Host "[5/5] Iniciando o serviço..."
if ($Autostart) {
    & $exe autostart on
    if ($LASTEXITCODE -ne 0) { Fail "autostart falhou (código $LASTEXITCODE)" }
}
& $exe start
if ($LASTEXITCODE -ne 0) { Fail "não foi possível iniciar o app (código $LASTEXITCODE); veja o log em $env:LOCALAPPDATA\not-a-ctrl-f\logs\app.log" }

Write-Host ""
Write-Host "Pronto! O serviço está rodando em segundo plano." -ForegroundColor Green
Write-Host "Para buscar: not-a-ctrl-f search `"...`"   |   para ver o índice: not-a-ctrl-f status"
Write-Host "Pastas monitoradas: edite $env:LOCALAPPDATA\not-a-ctrl-f\config.toml (watch_dirs = [...]); veja o README."
