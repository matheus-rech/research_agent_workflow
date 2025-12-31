# NAHN Multi-Agent Research API

A full-stack multi-agent deep research API for neurosurgical meta-analysis opportunities, inspired by [dair-ai/m2-deep-research](https://github.com/dair-ai/m2-deep-research).

## Features

- **Multi-Agent Architecture**: Supervisor + Planning + Search + Synthesis agents
- **Multiple AI Providers**: MiniMax M2.1, OpenAI, Gemini, Anthropic Claude
- **Neural Search**: Exa API for semantic search across medical literature
- **Backward Snowballing**: Semantic Scholar integration for reference expansion
- **NAHN Signal Evaluation**: Relevance scoring and method recommendation

## API Endpoints

| Endpoint | Description |
|----------|-------------|
| `/health` | Health check |
| `/health/detailed` | Detailed status with feature availability |
| `/research-orchestrator` | Run ALL agents in parallel |
| `/multi-agent-research` | dair-ai style multi-agent research |
| `/deep-research` | OpenAI deep research |
| `/deep-research-minimax` | MiniMax M2.1 deep research |
| `/snowball` | Backward snowballing via Semantic Scholar |
| `/exa-research` | Exa semantic search |
| `/evaluate` | NAHN signal evaluation |
| `/prospero/generate` | PROSPERO template generation |

## Deployment

### Railway (Recommended)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/template)

1. Fork this repository
2. Connect to Railway
3. Add environment variables
4. Deploy

### Environment Variables

```
OPENAI_API_KEY=your_key
GEMINI_API_KEY=your_key
MINIMAX_API_KEY=your_key
ANTHROPIC_API_KEY=your_key
EXA_API_KEY=your_key
```

### Local Development

```bash
pip install -r requirements.txt
uvicorn app:app --reload
```

## Usage

```bash
# Health check
curl https://your-app.railway.app/health/detailed

# Multi-agent research
curl -X POST https://your-app.railway.app/multi-agent-research \
  -H "Content-Type: application/json" \
  -d '{
    "signal_id": "test-001",
    "title": "VP Shunt vs ETV for Hydrocephalus",
    "abstract": "Comparison of treatments...",
    "subspecialty": "hydrocephalus",
    "recommended_methods": ["pairwise", "nma"],
    "max_sources": 30
  }'
```

## Architecture

```
┌─────────────────────────────────────────┐
│         Research Orchestrator           │
└─────────────────────────────────────────┘
                    │
    ┌───────────────┼───────────────┐
    │               │               │
    ▼               ▼               ▼
┌─────────┐   ┌─────────┐   ┌─────────┐
│ Multi-  │   │ OpenAI  │   │ Gemini  │
│ Agent   │   │         │   │         │
└─────────┘   └─────────┘   └─────────┘
    │
    ├─── Planning Agent (Gemini)
    ├─── Web Search (Exa)
    └─── Synthesis (MiniMax M2.1)
```

## License

MIT
