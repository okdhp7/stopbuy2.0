# StopBuy2.0

StopBuy2.0 is a small end-to-end purchase regret prediction system.

Flow:

1. The web screen accepts a product URL, product image, or manual product data.
2. The browser opens a WebSocket connection to the Backend.
3. The Backend opens a WebSocket connection to the AI Agent.
4. The AI Agent extracts product information, predicts purchase regret probability, and recommends alternatives when the score is above the threshold.
5. The Backend relays progress and result messages back to the browser.

## Project Structure

```text
StopBuy2.0/
  agent/
    product_extractor.py
    predictor.py
    server.py
    requirements.txt
  backend/
    main.py
    requirements.txt
  frontend/
    client/
    package.json
    vite.config.ts
  run_agent.ps1
  run_backend.ps1
```

## Run

Open two PowerShell terminals.

Terminal 1:

```powershell
cd C:\Users\user\Documents\Codex\2026-05-21\github-pr\StopBuy2.0
.\run_agent.ps1
```

Terminal 2:

```powershell
cd C:\Users\user\Documents\Codex\2026-05-21\github-pr\StopBuy2.0
.\run_backend.ps1
```

Then open:

```text
http://127.0.0.1:8010
```

## Optional Environment Variables

```powershell
$env:OPENAI_API_KEY="sk-..."
$env:LLM_MODEL_NAME="gpt-4o-mini"
$env:REGRET_THRESHOLD="0.4"
$env:AGENT_WS_URL="ws://127.0.0.1:8765"
```

Without `OPENAI_API_KEY`, image input still works with a fallback product profile, but true image understanding is skipped.

## Run With Docker

Build all images:

```powershell
cd C:\Users\user\Documents\Codex\2026-05-21\github-pr\StopBuy2.0
docker compose build
```

Start all services:

```powershell
docker compose up -d
```

Open the frontend:

```text
http://127.0.0.1:8080
```

Service ports:

```text
frontend: http://127.0.0.1:8080
backend:  http://127.0.0.1:8010
agent:    ws://127.0.0.1:8765
```

Stop services:

```powershell
docker compose down
```

## Run With Docker Dev Reload

Use the development compose file when you want source changes to be reflected automatically.

```powershell
cd C:\Users\user\Documents\Codex\2026-05-21\github-pr\StopBuy2.0
docker compose -f docker-compose.dev.yml up --build
```

Open:

```text
http://127.0.0.1:8080
```

Reload behavior:

- Frontend changes under `frontend/` are reflected by Vite HMR.
- Backend changes under `backend/` restart `uvicorn` with `--reload`.
- Agent changes under `agent/` restart `server.py` through `watchfiles`.

Stop development services:

```powershell
docker compose -f docker-compose.dev.yml down
```
