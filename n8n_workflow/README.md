# NAHN n8n Workflow

This directory contains the n8n workflow for the NAHN (Neurosurgical Automated High-impact Network) Multi-Agent Research Pipeline.

## Files

| File | Description |
|------|-------------|
| `nahn_workflow.json` | Full workflow with all metadata and tags |
| `nahn_workflow_import.json` | Clean workflow for direct n8n API import |

## Workflow Overview

**Name**: NAHN Enhanced Pipeline v2.2 - Full Multi-Agent Architecture  
**Nodes**: 33  
**Version**: 2.2.0

### Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         TRIGGERS                                     │
│              Daily (6 AM) + Manual Trigger                          │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      DATA SOURCES                                    │
│         PubMed API  +  ClinicalTrials.gov API                       │
└─────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    NAHN API PROCESSING                               │
│    Signal Evaluation → Method Recommendation → Journal Targeting    │
└─────────────────────────────────────────────────────────────────────┘
                                    │
        ┌───────────────────────────┼───────────────────────────┐
        │                           │                           │
        ▼                           ▼                           ▼
┌───────────────────┐   ┌───────────────────┐   ┌───────────────────┐
│  Research         │   │  Snowball         │   │  Exa Semantic     │
│  Orchestrator     │   │  Expansion        │   │  Search           │
│  (All Agents)     │   │                   │   │                   │
└───────────────────┘   └───────────────────┘   └───────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      NOTIFICATIONS                                   │
│              Email (HTML) + Slack (Blocks)                          │
└─────────────────────────────────────────────────────────────────────┘
```

## Key Nodes

| Node | Type | Purpose |
|------|------|---------|
| Daily Trigger | Schedule | Runs at 6 AM daily |
| Manual Trigger | Manual | For testing |
| PubMed Search | HTTP Request | Search recent neurosurgical publications |
| ClinicalTrials.gov | HTTP Request | Search completed trials |
| NAHN Signal Evaluation | HTTP Request | Score and classify signals |
| Research Orchestrator | HTTP Request | Run all AI agents in parallel |
| Multi-Agent Deep Research | HTTP Request | dair-ai style multi-agent research |
| Snowball Expansion | HTTP Request | Backward snowballing |
| Exa Semantic Search | HTTP Request | Neural semantic search |
| PROSPERO Generator | HTTP Request | Generate registration templates |
| Email Notification | Email | Rich HTML notifications |
| Slack Notification | Slack | Block-formatted alerts |

## Import Instructions

### Option 1: n8n UI Import

1. Open your n8n instance
2. Go to **Workflows** → **Import from File**
3. Select `nahn_workflow_import.json`
4. Configure environment variables (see below)
5. Activate the workflow

### Option 2: n8n API Import

```bash
curl -X POST "https://your-n8n-instance/api/v1/workflows" \
  -H "X-N8N-API-KEY: your_api_key" \
  -H "Content-Type: application/json" \
  -d @nahn_workflow_import.json
```

## Required Environment Variables

Set these in your n8n instance settings:

```
NAHN_API_URL=https://your-nahn-api-url.railway.app
NCBI_API_KEY=your_ncbi_key (optional, for higher rate limits)
NOTIFICATION_EMAIL=your@email.com
SLACK_WEBHOOK_URL=https://hooks.slack.com/services/...
```

## API Endpoints Used

The workflow calls these NAHN API endpoints:

| Endpoint | Purpose |
|----------|---------|
| `/health` | Health check before processing |
| `/evaluate` | Signal evaluation and scoring |
| `/research-orchestrator` | Run all research agents |
| `/multi-agent-research` | Multi-agent deep research |
| `/snowball` | Backward snowballing |
| `/exa-research` | Semantic search |
| `/prospero/generate` | PROSPERO template generation |
| `/signals/store` | Store processed signals |

## Customization

### Modify Search Terms

Edit the PubMed Search node to change the search query:
- Default searches for: neurosurgery, brain surgery, spine surgery, etc.
- Filters for: RCTs, meta-analyses, systematic reviews
- Time range: last 7 days

### Adjust Notification Channels

- **Email**: Update the NOTIFICATION_EMAIL environment variable
- **Slack**: Update the SLACK_WEBHOOK_URL environment variable
- **Add more**: Connect additional notification nodes (Teams, Discord, etc.)

### Change Schedule

Edit the Daily Trigger node to adjust the schedule:
- Default: Every 24 hours at 6 AM
- Can be changed to any cron expression

## Troubleshooting

### Workflow not triggering
- Check that the workflow is activated
- Verify the NAHN API is running and accessible
- Check n8n execution logs

### API errors
- Verify NAHN_API_URL is correct
- Check API health: `curl $NAHN_API_URL/health`
- Review API logs for detailed errors

### No results
- PubMed may not have new publications matching criteria
- Try the Manual Trigger with test data
- Adjust search terms if needed
