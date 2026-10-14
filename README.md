# not-a-ctrl-f

Busca local nos seus arquivos: documentos, imagens e áudio. Combina busca por palavra-chave (SQLite FTS5) com busca semântica (EmbeddingGemma 2, rodando localmente via `llama-server.exe`), então você acha "relatório de vendas" mesmo digitando "quanto faturamos". Tudo roda na sua máquina. O uso é pela linha de comando; integração com o Explorador de Arquivos e um app desktop estão planejados.

## Requisitos

- Windows 10/11 x64
- ~2 GB de disco (llama.cpp e modelos)
- Internet só na instalação
- GPU NVIDIA opcional; sem ela o app usa a CPU

## Instalação

Em um PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://raw.githubusercontent.com/janseealves/not-a-ctrl-f/main/install.ps1 | iex"
```

Ou baixe/clone o repositório e dê duplo clique em `install.cmd`.

O instalador instala o `uv` (se faltar), o app, o llama.cpp e os modelos, e inicia o serviço em segundo plano. Opções (no PowerShell, `.\install.ps1 ...`; no `install.cmd` os argumentos são repassados):

- `-Backend auto|cuda|vulkan|cpu`: `auto` (padrão) usa CUDA se houver GPU NVIDIA, senão CPU
- `-Autostart`: inicia o app junto com o Windows
- `-Source <pasta ou url>`: instala de outra origem

## Uso

O serviço roda em segundo plano. Na primeira vez os arquivos das pastas monitoradas são indexados; a busca já funciona enquanto isso.

Comandos:

```
not-a-ctrl-f start                       # inicia em segundo plano
not-a-ctrl-f stop                        # encerra
not-a-ctrl-f run                         # roda neste terminal (--port N)
not-a-ctrl-f search relatório vendas     # busca (usa a instância em execução, se houver)
not-a-ctrl-f search contrato --type doc --limit 5
not-a-ctrl-f search reunião --keyword-only --json
not-a-ctrl-f status                      # versão, pastas, componentes, instância e índice
not-a-ctrl-f setup --backend cpu         # (re)instala llama.cpp e modelos (--force, --verify)
not-a-ctrl-f autostart on|off|status     # iniciar com o Windows
```

`search --type` aceita `doc`, `image` e `audio`. Sem instância em execução, a busca semântica sobe um `llama-server` temporário só para a consulta (mais lento); `--keyword-only` evita isso.

## Configuração

Crie `%LOCALAPPDATA%\not-a-ctrl-f\config.toml`:

```toml
watch_dirs = ["C:/Users/voce/Projetos", "D:/Fotos"]
port = 47615
```

Variáveis de ambiente (têm precedência sobre o `config.toml`):

| Variável | Efeito |
|---|---|
| `NACF_DB_PATH` | caminho do banco (`index.db`) |
| `NACF_MODELS_DIR` | pasta dos modelos GGUF |
| `NACF_LLAMA_SERVER` | caminho do `llama-server.exe` |
| `NACF_WATCH_DIRS` | pastas monitoradas, separadas por `;` |
| `NACF_PORT` | porta da API local |

Sem `watch_dirs`, são monitoradas (as que existirem): Documentos, Área de Trabalho, Downloads, Imagens, Músicas e Vídeos.

## Tipos suportados

- Documentos: `csv` `docx` `htm` `html` `ini` `json` `log` `markdown` `md` `odt` `pdf` `rst` `tex` `toml` `tsv` `txt` `xml` `yaml` `yml`
- Imagens: `bmp` `gif` `jpeg` `jpg` `png` `tif` `tiff` `webp`
- Áudio: `flac` `mp3` `ogg` `wav`

## Privacidade

Tudo é local: os arquivos, o índice e a busca semântica nunca saem da sua máquina. A API local só escuta em `127.0.0.1`. A internet é usada apenas na instalação, para baixar o `uv`, o llama.cpp e os modelos.

## Onde ficam os dados

Em `%LOCALAPPDATA%\not-a-ctrl-f`:

- `index.db`: o índice
- `models\`: modelos GGUF
- `llama.cpp\`: `llama-server.exe` e DLLs
- `logs\app.log` e `logs\llama-server.log`: logs
- `config.toml`: sua configuração (opcional)

## Solução de problemas

- Veja os logs em `%LOCALAPPDATA%\not-a-ctrl-f\logs\`.
- `not-a-ctrl-f status` mostra o que falta instalar, se o app está rodando e o estado do índice.
- Porta em uso: defina outra com `port` no `config.toml` ou `NACF_PORT`.
- Se a GPU falhar, o app volta para a CPU sozinho; para forçar a CPU, rode `not-a-ctrl-f setup --backend cpu --force`.
- Download corrompido ou modelo incompleto: `not-a-ctrl-f setup --verify` confere o sha256 e baixa de novo o que estiver errado.
- App travado: `not-a-ctrl-f stop`.

## Desinstalação

```powershell
.\uninstall.ps1               # remove o app e o início automático
.\uninstall.ps1 -RemoveData   # também apaga %LOCALAPPDATA%\not-a-ctrl-f (pede confirmação)
```

## Desenvolvimento

```
uv sync
uv run pytest
uv run not-a-ctrl-f run
```
