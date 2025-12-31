# NAHN Multi-Agent Research API & n8n Workflow

A complete full-stack solution for automated neurosurgical meta-analysis opportunity detection, featuring a multi-agent deep research API and n8n workflow automation.

Inspired by [dair-ai/m2-deep-research](https://github.com/dair-ai/m2-deep-research).

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         n8n WORKFLOW                                 │
│    Triggers → Data Sources → NAHN API → Research → Notifications    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    NAHN MULTI-AGENT API                              │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│  Multi-Agent      │   │  OpenAI           │   │  Gemini           │
│  Supervisor       │   │  Deep Research    │   │  Deep Research    │
│  (dair-ai style)  │   │                   │   │                   │
└───────────────────┘   └───────────────────┘   └───────────────────┘
        │
        ├── Planning Agent (Gemini 2.0)
        ├── Web Search (Exa API)
        └── Synthesis (MiniMax M2.1)
```

## 📁 Repository Structure

```
├── app.py                      # FastAPI application
├── agents/
│   └── multi_agent_research.py # Multi-agent research module
├── n8n_workflow/
│   ├── nahn_workflow.json      # Full n8n workflow
│   ├── nahn_workflow_import.json # Clean import version
│   └── README.md               # Workflow documentation
├── requirements.txt            # Python dependencies
├── Procfile                    # Railway deployment
├── railway.json                # Railway configuration
└── .env.example                # Environment variables template
```

## 🚀 Quick Start

### 1. Deploy the API

#### Railway (Recommended)

[![Deploy on Railway](https://railway.app/button.svg)](https://railway.app/new/github)

1. Fork this repository
2. Connect to Railway
3. Add environment variables
4. Deploy

#### Local Development

```bash
pip install -r requirements.txt
uvicorn app:app --reload --port 8080
```

### 2. Import n8n Workflow

1. Open your n8n instance
2. Go to **Workflows** → **Import from File**
3. Select `n8n_workflow/nahn_workflow_import.json`
4. Set `NAHN_API_URL` environment variable to your API URL
5. Activate the workflow

## 🔌 API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /health` | Health check |
| `GET /health/detailed` | Detailed status with feature availability |
| `POST /research-orchestrator` | Run ALL agents in parallel |
| `POST /multi-agent-research` | dair-ai style multi-agent research |
| `POST /deep-research` | OpenAI deep research |
| `POST /deep-research-minimax` | MiniMax M2.1 deep research |
| `POST /snowball` | Backward snowballing via Semantic Scholar |
| `POST /exa-research` | Exa semantic search |
| `POST /evaluate` | NAHN signal evaluation |
| `POST /prospero/generate` | PROSPERO template generation |

## 🤖 Multi-Agent System

| Agent | Model | Role |
|-------|-------|------|
| **Supervisor** | MiniMax M2.1 | Orchestrates workflow, synthesizes reports |
| **Planning** | Gemini 2.0 Flash | Decomposes queries into optimized subqueries |
| **Web Search** | Exa API | Neural semantic search across medical domains |
| **Synthesis** | MiniMax M2.1 | Generates comprehensive reports with citations |

## ⚙️ Environment Variables

```bash
# Deep Research APIs (at least one required)
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
MINIMAX_API_KEY=your_minimax_key
ANTHROPIC_API_KEY=your_anthropic_key

# Search APIs
EXA_API_KEY=your_exa_key
SEMANTIC_SCHOLAR_API_KEY=your_semantic_scholar_key

# n8n Workflow
NAHN_API_URL=https://your-api.railway.app
NOTIFICATION_EMAIL=your@email.com
SLACK_WEBHOOK_URL=https://hooks.slack.com/...
```

## 📊 n8n Workflow Features

- **Daily Automated Scans**: Searches PubMed and ClinicalTrials.gov at 6 AM
- **Multi-Agent Research**: Runs all AI agents in parallel for comprehensive analysis
- **Backward Snowballing**: Expands literature coverage via reference chains
- **Smart Notifications**: Rich HTML emails and Slack block messages
- **PROSPERO Generation**: Auto-generates systematic review registration templates

See [n8n_workflow/README.md](n8n_workflow/README.md) for detailed workflow documentation.

## 📖 Usage Example

```bash
# Health check
curl https://your-api.railway.app/health/detailed

# Multi-agent research
curl -X POST https://your-api.railway.app/research-orchestrator \
  -H "Content-Type: application/json" \
  -d '{
    "signal_id": "test-001",
    "title": "VP Shunt vs ETV for Hydrocephalus",
    "abstract": "Comparison of surgical treatments...",
    "subspecialty": "hydrocephalus",
    "recommended_methods": ["pairwise", "nma"],
    "max_sources": 50
  }'
```

## 🔬 Neurosurgical Subspecialties

The system supports these subspecialties with specialized keyword detection:

- Hydrocephalus
- Spine
- Vascular
- Oncology
- Functional
- Trauma
- Pediatric

## 📚 References

- [dair-ai/m2-deep-research](https://github.com/dair-ai/m2-deep-research) - Multi-agent architecture inspiration
- [Wohlin 2014](https://www.researchgate.net/publication/261075446) - Backward snowballing methodology
- [Exa API](https://exa.ai) - Neural search
- [Semantic Scholar API](https://www.semanticscholar.org/product/api) - Citation data

## 📄 License

MIT
