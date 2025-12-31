"""
NAHN Enhanced API v2.0
Neurosurgical Automated High-impact Network with Deep Research Integration

This module provides the enhanced FastAPI backend with:
- Deep Research Agent integration (OpenAI o3-deep-research, Gemini)
- Backward Snowballing via Semantic Scholar API
- Exa Semantic Search integration
- Multi-agent supervisor architecture
"""

import os
import json
import time
import asyncio
import httpx
from typing import List, Dict, Any, Optional
from datetime import datetime
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NAHN Enhanced API v2.0",
    description="Neurosurgical Automated High-impact Network with Deep Research Integration",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================================
# Pydantic Models
# ============================================================================

class SignalInput(BaseModel):
    """Input model for signal evaluation"""
    title: str
    abstract: str
    subspecialty: Optional[str] = "general"
    study_design: Optional[str] = "unknown"
    sample_size: Optional[int] = 0
    source: Optional[str] = "unknown"
    pmid: Optional[str] = None
    nct_id: Optional[str] = None


class DeepResearchRequest(BaseModel):
    """Request model for deep research agent"""
    signal_id: str
    title: str
    abstract: str
    subspecialty: str = "general"
    recommended_methods: List[str] = []
    research_depth: str = "comprehensive"  # "quick", "standard", "comprehensive"
    include_citations: bool = True
    max_sources: int = 50


class SnowballRequest(BaseModel):
    """Request model for backward snowballing"""
    signal_id: str
    seed_pmids: List[str] = []
    seed_dois: List[str] = []
    max_depth: int = 2
    max_papers_per_level: int = 20
    relevance_threshold: float = 0.6


class ExaResearchRequest(BaseModel):
    """Request model for Exa semantic search"""
    signal_id: str
    query: str
    subspecialty: str = "general"
    search_type: str = "neural"  # "neural", "keyword", "auto"
    num_results: int = 30
    include_domains: List[str] = []
    output_schema: Optional[Dict] = None


class ProsperoRequest(BaseModel):
    """Request model for enhanced PROSPERO generation"""
    signal: Dict[str, Any]
    include_deep_research: bool = True
    include_snowball_refs: bool = True
    template_version: str = "2.0"


# ============================================================================
# Configuration
# ============================================================================

# API Keys from environment
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EXA_API_KEY = os.getenv("EXA_API_KEY", "")
SEMANTIC_SCHOLAR_API_KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")

# Neurosurgery journals database
NEUROSURGERY_JOURNALS = [
    {"name": "Journal of Neurosurgery", "impact_factor": 5.8, "subspecialties": ["all"]},
    {"name": "Neurosurgery", "impact_factor": 5.3, "subspecialties": ["all"]},
    {"name": "World Neurosurgery", "impact_factor": 2.1, "subspecialties": ["all"]},
    {"name": "Journal of Neurosurgery: Spine", "impact_factor": 3.8, "subspecialties": ["spine"]},
    {"name": "Spine", "impact_factor": 3.5, "subspecialties": ["spine"]},
    {"name": "The Spine Journal", "impact_factor": 4.1, "subspecialties": ["spine"]},
    {"name": "European Spine Journal", "impact_factor": 2.9, "subspecialties": ["spine"]},
    {"name": "Journal of Neuro-Oncology", "impact_factor": 4.2, "subspecialties": ["oncology"]},
    {"name": "Neuro-Oncology", "impact_factor": 13.0, "subspecialties": ["oncology"]},
    {"name": "Stroke", "impact_factor": 10.2, "subspecialties": ["vascular"]},
    {"name": "Journal of Cerebral Blood Flow & Metabolism", "impact_factor": 6.0, "subspecialties": ["vascular"]},
    {"name": "Epilepsia", "impact_factor": 6.0, "subspecialties": ["functional"]},
    {"name": "Movement Disorders", "impact_factor": 10.3, "subspecialties": ["functional"]},
    {"name": "Stereotactic and Functional Neurosurgery", "impact_factor": 1.8, "subspecialties": ["functional"]},
    {"name": "Fluids and Barriers of the CNS", "impact_factor": 5.4, "subspecialties": ["hydrocephalus"]},
    {"name": "Child's Nervous System", "impact_factor": 1.5, "subspecialties": ["pediatric", "hydrocephalus"]},
    {"name": "Pediatric Neurosurgery", "impact_factor": 1.2, "subspecialties": ["pediatric"]},
    {"name": "Journal of Neurotrauma", "impact_factor": 5.2, "subspecialties": ["trauma"]},
    {"name": "Brain Injury", "impact_factor": 2.2, "subspecialties": ["trauma"]},
    {"name": "Lancet Neurology", "impact_factor": 48.0, "subspecialties": ["all"]},
    {"name": "JAMA Neurology", "impact_factor": 29.0, "subspecialties": ["all"]},
    {"name": "Brain", "impact_factor": 15.0, "subspecialties": ["all"]},
    {"name": "Annals of Neurology", "impact_factor": 12.0, "subspecialties": ["all"]},
    {"name": "Neurology", "impact_factor": 12.0, "subspecialties": ["all"]},
    {"name": "Acta Neurochirurgica", "impact_factor": 2.4, "subspecialties": ["all"]}
]

# Meta-analytic methods
META_ANALYTIC_METHODS = {
    "pairwise": {
        "name": "Pairwise Meta-Analysis",
        "min_studies": 2,
        "description": "Traditional meta-analysis comparing two interventions",
        "tools": ["RevMan", "Stata", "R meta package"]
    },
    "nma": {
        "name": "Network Meta-Analysis",
        "min_studies": 3,
        "description": "Compares multiple interventions simultaneously through direct and indirect evidence",
        "tools": ["netmeta", "gemtc", "WinBUGS"]
    },
    "ipd": {
        "name": "Individual Patient Data Meta-Analysis",
        "min_studies": 2,
        "description": "Uses raw patient-level data for more precise estimates",
        "tools": ["ipdmetan", "one-stage models"]
    },
    "bayesian": {
        "name": "Bayesian Meta-Analysis",
        "min_studies": 2,
        "description": "Incorporates prior knowledge and provides probability distributions",
        "tools": ["brms", "JAGS", "Stan"]
    },
    "dose_response": {
        "name": "Dose-Response Meta-Analysis",
        "min_studies": 3,
        "description": "Models relationship between dose/exposure and outcome",
        "tools": ["dosresmeta", "metafor"]
    },
    "diagnostic": {
        "name": "Diagnostic Test Accuracy Meta-Analysis",
        "min_studies": 4,
        "description": "Synthesizes sensitivity and specificity across studies",
        "tools": ["mada", "diagmeta", "HSROC"]
    },
    "prognostic": {
        "name": "Prognostic Factor Meta-Analysis",
        "min_studies": 3,
        "description": "Synthesizes hazard ratios for prognostic factors",
        "tools": ["metamisc", "survival analysis"]
    },
    "umbrella": {
        "name": "Umbrella Review",
        "min_studies": 2,
        "description": "Systematic review of systematic reviews",
        "tools": ["AMSTAR-2", "GRADE"]
    },
    "living": {
        "name": "Living Systematic Review",
        "min_studies": 2,
        "description": "Continuously updated review as new evidence emerges",
        "tools": ["Covidence", "automated search alerts"]
    },
    "component_nma": {
        "name": "Component Network Meta-Analysis",
        "min_studies": 5,
        "description": "Evaluates individual components of complex interventions",
        "tools": ["bnma", "multinma"]
    }
}


# ============================================================================
# Helper Functions
# ============================================================================

def get_target_journals(subspecialty: str, top_n: int = 5) -> List[Dict]:
    """Get target journals for a subspecialty"""
    matching = [j for j in NEUROSURGERY_JOURNALS 
                if subspecialty in j["subspecialties"] or "all" in j["subspecialties"]]
    matching.sort(key=lambda x: x["impact_factor"], reverse=True)
    return matching[:top_n]


def recommend_methods(study_design: str, sample_size: int, subspecialty: str) -> List[Dict]:
    """Recommend meta-analytic methods based on study characteristics"""
    recommendations = []
    
    # Always recommend pairwise as baseline
    recommendations.append({
        "method": "pairwise",
        "name": META_ANALYTIC_METHODS["pairwise"]["name"],
        "rationale": "Standard approach for synthesizing evidence",
        "priority": 1
    })
    
    # Recommend NMA if multiple comparators likely
    if "rct" in study_design.lower() or "randomized" in study_design.lower():
        recommendations.append({
            "method": "nma",
            "name": META_ANALYTIC_METHODS["nma"]["name"],
            "rationale": "RCT design suitable for network comparisons",
            "priority": 2
        })
    
    # Recommend Bayesian for smaller samples
    if sample_size < 200:
        recommendations.append({
            "method": "bayesian",
            "name": META_ANALYTIC_METHODS["bayesian"]["name"],
            "rationale": "Bayesian methods handle small samples well with informative priors",
            "priority": 3
        })
    
    # Recommend IPD for large studies
    if sample_size >= 500:
        recommendations.append({
            "method": "ipd",
            "name": META_ANALYTIC_METHODS["ipd"]["name"],
            "rationale": "Large sample size justifies effort of obtaining individual patient data",
            "priority": 2
        })
    
    # Subspecialty-specific recommendations
    if subspecialty == "oncology":
        recommendations.append({
            "method": "prognostic",
            "name": META_ANALYTIC_METHODS["prognostic"]["name"],
            "rationale": "Oncology studies often report survival outcomes suitable for prognostic synthesis",
            "priority": 3
        })
    
    if subspecialty == "functional":
        recommendations.append({
            "method": "dose_response",
            "name": META_ANALYTIC_METHODS["dose_response"]["name"],
            "rationale": "DBS and stimulation studies often involve dose/parameter optimization",
            "priority": 3
        })
    
    return sorted(recommendations, key=lambda x: x["priority"])


def detect_subgroups(subspecialty: str, abstract: str) -> List[Dict]:
    """Detect potential subgroup analyses based on subspecialty and abstract"""
    subgroups = []
    abstract_lower = abstract.lower()
    
    # Common subgroups
    common_subgroups = [
        {"name": "Age groups", "keywords": ["age", "elderly", "pediatric", "adult"]},
        {"name": "Sex/Gender", "keywords": ["male", "female", "sex", "gender"]},
        {"name": "Disease severity", "keywords": ["mild", "moderate", "severe", "grade"]},
        {"name": "Comorbidities", "keywords": ["comorbid", "diabetes", "hypertension", "obesity"]}
    ]
    
    for sg in common_subgroups:
        if any(kw in abstract_lower for kw in sg["keywords"]):
            subgroups.append({
                "name": sg["name"],
                "rationale": f"Abstract mentions relevant terms: {', '.join([k for k in sg['keywords'] if k in abstract_lower])}"
            })
    
    # Subspecialty-specific subgroups
    subspecialty_subgroups = {
        "hydrocephalus": [
            {"name": "Shunt type", "keywords": ["vp shunt", "va shunt", "lp shunt"]},
            {"name": "Etiology", "keywords": ["idiopathic", "secondary", "post-hemorrhagic"]},
            {"name": "Valve type", "keywords": ["programmable", "fixed", "anti-siphon"]}
        ],
        "spine": [
            {"name": "Surgical approach", "keywords": ["anterior", "posterior", "lateral", "mis"]},
            {"name": "Spinal level", "keywords": ["cervical", "thoracic", "lumbar", "sacral"]},
            {"name": "Fusion technique", "keywords": ["plif", "tlif", "alif", "xlif"]}
        ],
        "oncology": [
            {"name": "Tumor grade", "keywords": ["grade i", "grade ii", "grade iii", "grade iv", "who grade"]},
            {"name": "Molecular markers", "keywords": ["idh", "mgmt", "1p19q", "egfr"]},
            {"name": "Extent of resection", "keywords": ["gtr", "str", "biopsy", "resection"]}
        ],
        "vascular": [
            {"name": "Aneurysm location", "keywords": ["mca", "acom", "pcom", "basilar"]},
            {"name": "Treatment modality", "keywords": ["clipping", "coiling", "flow diverter"]},
            {"name": "Rupture status", "keywords": ["ruptured", "unruptured", "sah"]}
        ],
        "functional": [
            {"name": "Target structure", "keywords": ["stn", "gpi", "vim", "anterior nucleus"]},
            {"name": "Indication", "keywords": ["parkinson", "tremor", "dystonia", "epilepsy"]},
            {"name": "Stimulation parameters", "keywords": ["frequency", "amplitude", "pulse width"]}
        ]
    }
    
    if subspecialty in subspecialty_subgroups:
        for sg in subspecialty_subgroups[subspecialty]:
            if any(kw in abstract_lower for kw in sg["keywords"]):
                subgroups.append({
                    "name": sg["name"],
                    "rationale": f"Subspecialty-specific: {', '.join([k for k in sg['keywords'] if k in abstract_lower])}"
                })
    
    return subgroups


# ============================================================================
# Deep Research Integration
# ============================================================================

async def call_openai_deep_research(request: DeepResearchRequest) -> Dict:
    """Call OpenAI o3-deep-research API"""
    if not OPENAI_API_KEY:
        logger.warning("OpenAI API key not configured")
        return {"error": "OpenAI API key not configured", "report": ""}
    
    system_message = f"""You are a professional medical researcher specializing in neurosurgery and systematic reviews.
Your task is to conduct comprehensive research on the following topic and produce a detailed, citation-rich report.

Focus areas:
- Existing systematic reviews and meta-analyses on this topic
- Key randomized controlled trials and their findings
- Methodological considerations for evidence synthesis
- Gaps in the current evidence base
- Recommendations for future research

Subspecialty context: {request.subspecialty}
Recommended methods to consider: {', '.join(request.recommended_methods) if request.recommended_methods else 'Standard meta-analytic approaches'}

Provide inline citations and structure your report with clear sections."""

    user_query = f"""Research Topic: {request.title}

Background/Abstract: {request.abstract}

Please conduct a {request.research_depth} literature review and synthesis on this topic.
Include up to {request.max_sources} relevant sources with proper citations."""

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{OPENAI_API_BASE}/responses",
                headers={
                    "Authorization": f"Bearer {OPENAI_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "o3-deep-research",
                    "input": [
                        {"role": "developer", "content": [{"type": "input_text", "text": system_message}]},
                        {"role": "user", "content": [{"type": "input_text", "text": user_query}]}
                    ],
                    "reasoning": {"summary": "auto"},
                    "tools": [
                        {"type": "web_search_preview"},
                        {"type": "code_interpreter"}
                    ]
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                # Extract report and citations
                output = result.get("output", [])
                report = ""
                citations = []
                
                for item in output:
                    if item.get("type") == "message":
                        content = item.get("content", [])
                        for c in content:
                            if c.get("type") == "text":
                                report = c.get("text", "")
                                annotations = c.get("annotations", [])
                                for ann in annotations:
                                    citations.append({
                                        "title": ann.get("title", ""),
                                        "url": ann.get("url", ""),
                                        "start_index": ann.get("start_index", 0),
                                        "end_index": ann.get("end_index", 0)
                                    })
                
                return {
                    "report": report,
                    "citations": citations,
                    "key_findings": extract_key_findings(report),
                    "methodology_recommendations": extract_methodology_recommendations(report),
                    "evidence_gaps": extract_evidence_gaps(report),
                    "cited_pmids": extract_pmids_from_citations(citations),
                    "cited_dois": extract_dois_from_citations(citations)
                }
            else:
                logger.error(f"OpenAI API error: {response.status_code} - {response.text}")
                return {"error": f"API error: {response.status_code}", "report": ""}
                
    except Exception as e:
        logger.error(f"OpenAI Deep Research error: {str(e)}")
        return {"error": str(e), "report": ""}


async def call_gemini_deep_research(request: DeepResearchRequest) -> Dict:
    """Call Google Gemini Deep Research Agent"""
    if not GEMINI_API_KEY:
        logger.warning("Gemini API key not configured")
        return {"error": "Gemini API key not configured", "report": ""}
    
    prompt = f"""Research the following neurosurgical topic comprehensively:

Title: {request.title}
Abstract: {request.abstract}
Subspecialty: {request.subspecialty}

Format the output as a technical report with:
1. Executive Summary
2. Current Evidence Base
3. Key Studies and Findings
4. Methodological Considerations
5. Evidence Gaps and Future Directions

Include citations to all sources."""

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            # Start background research
            response = await client.post(
                "https://generativelanguage.googleapis.com/v1beta/interactions",
                headers={
                    "x-goog-api-key": GEMINI_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "input": prompt,
                    "agent": "deep-research-pro-preview-12-2025",
                    "background": True
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                interaction_id = result.get("id")
                
                # Poll for completion
                for _ in range(60):  # Max 10 minutes
                    await asyncio.sleep(10)
                    status_response = await client.get(
                        f"https://generativelanguage.googleapis.com/v1beta/interactions/{interaction_id}",
                        headers={"x-goog-api-key": GEMINI_API_KEY}
                    )
                    
                    if status_response.status_code == 200:
                        status = status_response.json()
                        if status.get("status") == "completed":
                            outputs = status.get("outputs", [])
                            report = outputs[-1].get("text", "") if outputs else ""
                            return {
                                "report": report,
                                "citations": [],
                                "key_findings": extract_key_findings(report),
                                "source": "gemini"
                            }
                        elif status.get("status") == "failed":
                            return {"error": status.get("error", "Unknown error"), "report": ""}
                
                return {"error": "Research timed out", "report": ""}
            else:
                return {"error": f"API error: {response.status_code}", "report": ""}
                
    except Exception as e:
        logger.error(f"Gemini Deep Research error: {str(e)}")
        return {"error": str(e), "report": ""}


def extract_key_findings(report: str) -> List[str]:
    """Extract key findings from research report"""
    findings = []
    lines = report.split('\n')
    
    for i, line in enumerate(lines):
        line_lower = line.lower()
        if any(kw in line_lower for kw in ['key finding', 'main result', 'significant', 'demonstrated', 'showed that']):
            # Get the sentence
            sentence = line.strip()
            if len(sentence) > 20 and len(sentence) < 500:
                findings.append(sentence)
    
    return findings[:10]  # Return top 10 findings


def extract_methodology_recommendations(report: str) -> List[str]:
    """Extract methodology recommendations from report"""
    recommendations = []
    lines = report.split('\n')
    
    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in ['recommend', 'should consider', 'meta-analysis', 'systematic review', 'methodology']):
            sentence = line.strip()
            if len(sentence) > 20 and len(sentence) < 500:
                recommendations.append(sentence)
    
    return recommendations[:5]


def extract_evidence_gaps(report: str) -> List[str]:
    """Extract evidence gaps from report"""
    gaps = []
    lines = report.split('\n')
    
    for line in lines:
        line_lower = line.lower()
        if any(kw in line_lower for kw in ['gap', 'lacking', 'needed', 'future research', 'limited evidence', 'insufficient']):
            sentence = line.strip()
            if len(sentence) > 20 and len(sentence) < 500:
                gaps.append(sentence)
    
    return gaps[:5]


def extract_pmids_from_citations(citations: List[Dict]) -> List[str]:
    """Extract PMIDs from citation URLs"""
    pmids = []
    for citation in citations:
        url = citation.get("url", "")
        if "pubmed" in url.lower():
            # Extract PMID from URL
            import re
            match = re.search(r'/(\d{7,8})/?', url)
            if match:
                pmids.append(match.group(1))
    return pmids


def extract_dois_from_citations(citations: List[Dict]) -> List[str]:
    """Extract DOIs from citation URLs"""
    dois = []
    for citation in citations:
        url = citation.get("url", "")
        if "doi.org" in url.lower():
            # Extract DOI from URL
            import re
            match = re.search(r'doi\.org/(10\.\d+/[^\s]+)', url)
            if match:
                dois.append(match.group(1))
    return dois


# ============================================================================
# Snowballing Integration
# ============================================================================

async def perform_snowballing(request: SnowballRequest) -> Dict:
    """Perform backward snowballing using Semantic Scholar API"""
    all_papers = []
    visited = set()
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Process seed PMIDs
        for pmid in request.seed_pmids[:10]:  # Limit to 10 seeds
            if pmid in visited:
                continue
            visited.add(pmid)
            
            try:
                # Get paper details from Semantic Scholar
                response = await client.get(
                    f"https://api.semanticscholar.org/graph/v1/paper/PMID:{pmid}",
                    params={
                        "fields": "title,abstract,year,citationCount,references,references.title,references.paperId,references.year"
                    },
                    headers={"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}
                )
                
                if response.status_code == 200:
                    paper = response.json()
                    references = paper.get("references", [])
                    
                    # Add references to results
                    for ref in references[:request.max_papers_per_level]:
                        if ref.get("paperId") and ref.get("paperId") not in visited:
                            all_papers.append({
                                "paper_id": ref.get("paperId"),
                                "title": ref.get("title", ""),
                                "year": ref.get("year"),
                                "source_pmid": pmid,
                                "depth": 1
                            })
                            visited.add(ref.get("paperId"))
                
                # Rate limiting
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Snowballing error for PMID {pmid}: {str(e)}")
        
        # Process seed DOIs
        for doi in request.seed_dois[:10]:
            if doi in visited:
                continue
            visited.add(doi)
            
            try:
                response = await client.get(
                    f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}",
                    params={
                        "fields": "title,abstract,year,citationCount,references,references.title,references.paperId,references.year"
                    },
                    headers={"x-api-key": SEMANTIC_SCHOLAR_API_KEY} if SEMANTIC_SCHOLAR_API_KEY else {}
                )
                
                if response.status_code == 200:
                    paper = response.json()
                    references = paper.get("references", [])
                    
                    for ref in references[:request.max_papers_per_level]:
                        if ref.get("paperId") and ref.get("paperId") not in visited:
                            all_papers.append({
                                "paper_id": ref.get("paperId"),
                                "title": ref.get("title", ""),
                                "year": ref.get("year"),
                                "source_doi": doi,
                                "depth": 1
                            })
                            visited.add(ref.get("paperId"))
                
                await asyncio.sleep(0.5)
                
            except Exception as e:
                logger.error(f"Snowballing error for DOI {doi}: {str(e)}")
    
    # Sort by year (most recent first)
    all_papers.sort(key=lambda x: x.get("year") or 0, reverse=True)
    
    # Identify key references (most cited within the network)
    key_references = all_papers[:5]
    
    return {
        "papers": all_papers,
        "total_papers": len(all_papers),
        "key_references": key_references,
        "network": {
            "seed_count": len(request.seed_pmids) + len(request.seed_dois),
            "expanded_count": len(all_papers),
            "max_depth": request.max_depth
        }
    }


# ============================================================================
# Exa Search Integration
# ============================================================================

async def perform_exa_search(request: ExaResearchRequest) -> Dict:
    """Perform semantic search using Exa API"""
    if not EXA_API_KEY:
        logger.warning("Exa API key not configured")
        return {"error": "Exa API key not configured", "related_studies": []}
    
    try:
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Build search query
            search_query = f"{request.query} {request.subspecialty} neurosurgery systematic review meta-analysis"
            
            response = await client.post(
                "https://api.exa.ai/search",
                headers={
                    "x-api-key": EXA_API_KEY,
                    "Content-Type": "application/json"
                },
                json={
                    "query": search_query,
                    "type": request.search_type,
                    "numResults": request.num_results,
                    "includeDomains": request.include_domains if request.include_domains else None,
                    "contents": {
                        "text": {"maxCharacters": 1000},
                        "highlights": True
                    }
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                results = result.get("results", [])
                
                related_studies = []
                for r in results:
                    related_studies.append({
                        "title": r.get("title", ""),
                        "url": r.get("url", ""),
                        "text": r.get("text", ""),
                        "highlights": r.get("highlights", []),
                        "score": r.get("score", 0)
                    })
                
                # Generate methodology insights
                methodology_insights = generate_methodology_insights(related_studies, request.subspecialty)
                
                # Generate gap analysis
                gap_analysis = generate_gap_analysis(related_studies, request.query)
                
                return {
                    "related_studies": related_studies,
                    "methodology_insights": methodology_insights,
                    "gap_analysis": gap_analysis,
                    "search_quality": len(related_studies) / request.num_results if request.num_results > 0 else 0
                }
            else:
                return {"error": f"Exa API error: {response.status_code}", "related_studies": []}
                
    except Exception as e:
        logger.error(f"Exa search error: {str(e)}")
        return {"error": str(e), "related_studies": []}


def generate_methodology_insights(studies: List[Dict], subspecialty: str) -> str:
    """Generate methodology insights from related studies"""
    if not studies:
        return "Insufficient data for methodology insights."
    
    # Analyze study types mentioned
    study_types = []
    for study in studies:
        text = (study.get("text", "") + " " + study.get("title", "")).lower()
        if "meta-analysis" in text:
            study_types.append("meta-analysis")
        if "systematic review" in text:
            study_types.append("systematic review")
        if "rct" in text or "randomized" in text:
            study_types.append("RCT")
        if "cohort" in text:
            study_types.append("cohort")
    
    type_counts = {}
    for t in study_types:
        type_counts[t] = type_counts.get(t, 0) + 1
    
    insights = f"Analysis of {len(studies)} related studies reveals: "
    if type_counts:
        insights += f"Study types include {', '.join([f'{k} ({v})' for k, v in type_counts.items()])}. "
    
    insights += f"For {subspecialty} topics, consider prioritizing high-quality RCTs and existing meta-analyses as primary evidence sources."
    
    return insights


def generate_gap_analysis(studies: List[Dict], query: str) -> str:
    """Generate evidence gap analysis"""
    if not studies:
        return "Limited evidence available. This represents a significant gap in the literature."
    
    # Check for recent studies
    recent_count = sum(1 for s in studies if "2024" in s.get("text", "") or "2025" in s.get("text", ""))
    
    gap_analysis = f"Based on {len(studies)} related publications: "
    
    if recent_count < 3:
        gap_analysis += "Limited recent evidence (2024-2025) suggests this may be an emerging area requiring updated synthesis. "
    else:
        gap_analysis += f"Found {recent_count} recent publications indicating active research in this area. "
    
    gap_analysis += "Potential gaps include: subgroup analyses, long-term outcomes, and comparative effectiveness studies."
    
    return gap_analysis


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.now().isoformat(),
        "features": {
            "deep_research": bool(OPENAI_API_KEY or GEMINI_API_KEY),
            "snowballing": True,
            "exa_search": bool(EXA_API_KEY)
        }
    }


@app.post("/evaluate")
async def evaluate_signal(signal: SignalInput):
    """Evaluate a research signal"""
    try:
        # Get target journals
        target_journals = get_target_journals(signal.subspecialty)
        
        # Recommend methods
        recommended_methods = recommend_methods(
            signal.study_design,
            signal.sample_size,
            signal.subspecialty
        )
        
        # Detect subgroups
        subgroups = detect_subgroups(signal.subspecialty, signal.abstract)
        
        # Calculate relevance score
        relevance_score = 50  # Base score
        if "rct" in signal.study_design.lower() or "randomized" in signal.study_design.lower():
            relevance_score += 20
        if signal.sample_size >= 200:
            relevance_score += 15
        if signal.sample_size >= 500:
            relevance_score += 10
        
        return {
            "success": True,
            "subspecialty": signal.subspecialty,
            "relevance_score": min(relevance_score, 100),
            "recommended_methods": recommended_methods,
            "target_journals": target_journals,
            "subgroups": subgroups,
            "spinoff_opportunities": [],
            "network_validation": {"status": "pending"},
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Evaluation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deep-research")
async def deep_research(request: DeepResearchRequest, background_tasks: BackgroundTasks):
    """Perform deep research using OpenAI or Gemini"""
    try:
        # Try OpenAI first, fall back to Gemini
        if OPENAI_API_KEY:
            result = await call_openai_deep_research(request)
        elif GEMINI_API_KEY:
            result = await call_gemini_deep_research(request)
        else:
            return {
                "error": "No deep research API configured",
                "report": "",
                "citations": []
            }
        
        return result
        
    except Exception as e:
        logger.error(f"Deep research error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/snowball")
async def snowball_search(request: SnowballRequest):
    """Perform backward snowballing"""
    try:
        result = await perform_snowballing(request)
        return result
    except Exception as e:
        logger.error(f"Snowballing error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/exa-research")
async def exa_research(request: ExaResearchRequest):
    """Perform Exa semantic search"""
    try:
        result = await perform_exa_search(request)
        return result
    except Exception as e:
        logger.error(f"Exa research error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/signals/store")
async def store_signal(signal: Dict[str, Any]):
    """Store enhanced signal for audit trail"""
    try:
        # In production, this would store to a database
        # For now, we just acknowledge receipt
        return {
            "success": True,
            "signal_id": signal.get("signal_id"),
            "stored_at": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Store signal error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/prospero/generate")
async def generate_prospero(request: ProsperoRequest):
    """Generate enhanced PROSPERO template"""
    try:
        signal = request.signal
        
        # Build PROSPERO template
        template = f"""# PROSPERO Registration Template
## Generated by NAHN Enhanced Pipeline v2.0

### 1. Review Title
{signal.get('title', 'TBD')}

### 2. Original Language Title
{signal.get('title', 'TBD')}

### 3. Anticipated or Actual Start Date
{datetime.now().strftime('%Y-%m-%d')}

### 4. Anticipated Completion Date
TBD

### 5. Stage of Review
Preliminary searches

### 6. Named Contact
TBD

### 7. Review Question
Based on the identified signal: {signal.get('title', '')}

### 8. Searches
Databases to be searched:
- PubMed/MEDLINE
- Cochrane Library
- Embase
- Web of Science
- ClinicalTrials.gov

### 9. Condition or Domain
Subspecialty: {signal.get('subspecialty', 'Neurosurgery')}

### 10. Participants/Population
TBD based on inclusion criteria

### 11. Intervention(s), Exposure(s)
To be defined based on the research question

### 12. Comparator(s)/Control
To be defined based on the research question

### 13. Types of Study to be Included
- Randomized Controlled Trials
- Prospective Cohort Studies
- Propensity Score Matched Studies

### 14. Main Outcome(s)
TBD

### 15. Data Extraction
Standardized data extraction form will be used

### 16. Risk of Bias Assessment
- RoB 2 for RCTs
- ROBINS-I for observational studies

### 17. Strategy for Data Synthesis
Recommended methods:
"""
        
        # Add recommended methods
        for method in signal.get('recommended_methods', [])[:3]:
            template += f"- {method.get('name', method)}: {method.get('rationale', '')}\n"
        
        # Add deep research findings if available
        if request.include_deep_research and signal.get('deep_research'):
            template += f"""

### 18. Evidence Synthesis Notes (from Deep Research)
{signal.get('deep_research', {}).get('comprehensive_report', '')[:2000]}

Key Findings:
"""
            for finding in signal.get('deep_research', {}).get('key_findings', [])[:5]:
                template += f"- {finding}\n"
        
        # Add snowball references if available
        if request.include_snowball_refs and signal.get('snowball_expansion'):
            template += f"""

### 19. Additional References (from Snowballing)
Total papers identified through backward snowballing: {signal.get('snowball_expansion', {}).get('total_expanded', 0)}

Key References:
"""
            for ref in signal.get('snowball_expansion', {}).get('key_references', [])[:5]:
                template += f"- {ref.get('title', ref)}\n"
        
        template += f"""

### 20. Target Journals for Publication
"""
        for journal in signal.get('target_journals', [])[:5]:
            template += f"- {journal.get('name', journal)} (IF: {journal.get('impact_factor', 'N/A')})\n"
        
        template += f"""

### 21. Subgroup Analyses Planned
"""
        for subgroup in signal.get('subgroups', [])[:5]:
            template += f"- {subgroup.get('name', subgroup)}: {subgroup.get('rationale', '')}\n"
        
        template += f"""

---
*Generated by NAHN Enhanced Pipeline v{request.template_version}*
*Timestamp: {datetime.now().isoformat()}*
"""
        
        return {
            "success": True,
            "template": template,
            "version": request.template_version,
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"PROSPERO generation error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8080)))


# ============================================================================
# MiniMax M2.1 Deep Research Integration
# ============================================================================

# Add MiniMax API keys to configuration
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_GROUP_ID = os.getenv("MINIMAX_GROUP_ID", "")


async def call_minimax_deep_research(request: DeepResearchRequest) -> Dict:
    """Call MiniMax M2.1 with interleaved thinking for deep research"""
    if not MINIMAX_API_KEY:
        logger.warning("MiniMax API key not configured")
        return {"error": "MiniMax API key not configured", "report": ""}
    
    system_message = f"""You are a professional medical researcher specializing in neurosurgery and systematic reviews.
You have access to comprehensive medical literature and can synthesize evidence from multiple sources.

Your task is to conduct deep research on the provided topic and produce a detailed, citation-rich report.

Focus areas:
- Existing systematic reviews and meta-analyses on this topic
- Key randomized controlled trials and their findings  
- Methodological considerations for evidence synthesis
- Gaps in the current evidence base
- Recommendations for future research

Subspecialty context: {request.subspecialty}
Recommended methods to consider: {', '.join(request.recommended_methods) if request.recommended_methods else 'Standard meta-analytic approaches'}

Provide inline citations and structure your report with clear sections.
Use your interleaved thinking capability to reason through the evidence systematically."""

    user_query = f"""Research Topic: {request.title}

Background/Abstract: {request.abstract}

Please conduct a {request.research_depth} literature review and synthesis on this topic.
Include up to {request.max_sources} relevant sources with proper citations.

Structure your response as:
1. Executive Summary
2. Current Evidence Base
3. Key Studies and Findings
4. Methodological Considerations
5. Evidence Gaps
6. Recommendations"""

    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            # MiniMax M2.1 API endpoint
            response = await client.post(
                "https://api.minimax.chat/v1/text/chatcompletion_v2",
                headers={
                    "Authorization": f"Bearer {MINIMAX_API_KEY}",
                    "Content-Type": "application/json"
                },
                json={
                    "model": "MiniMax-Text-01",
                    "messages": [
                        {"role": "system", "content": system_message},
                        {"role": "user", "content": user_query}
                    ],
                    "max_tokens": 16000,
                    "temperature": 0.7,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                
                # Extract thinking blocks if present (interleaved thinking)
                thinking_blocks = []
                if "<thinking>" in content:
                    import re
                    thinking_matches = re.findall(r'<thinking>(.*?)</thinking>', content, re.DOTALL)
                    thinking_blocks = thinking_matches
                    content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
                
                return {
                    "report": content.strip(),
                    "citations": [],
                    "key_findings": extract_key_findings(content),
                    "methodology_recommendations": extract_methodology_recommendations(content),
                    "evidence_gaps": extract_evidence_gaps(content),
                    "thinking_trace": thinking_blocks,
                    "source": "minimax_m2.1",
                    "cited_pmids": [],
                    "cited_dois": []
                }
            else:
                logger.error(f"MiniMax API error: {response.status_code} - {response.text}")
                return {"error": f"API error: {response.status_code}", "report": ""}
                
    except Exception as e:
        logger.error(f"MiniMax Deep Research error: {str(e)}")
        return {"error": str(e), "report": ""}


@app.post("/deep-research-minimax")
async def deep_research_minimax(request: DeepResearchRequest):
    """Perform deep research using MiniMax M2.1 with interleaved thinking"""
    try:
        result = await call_minimax_deep_research(request)
        return result
    except Exception as e:
        logger.error(f"MiniMax deep research error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/deep-research-multi")
async def deep_research_multi_agent(request: DeepResearchRequest):
    """
    Multi-agent deep research using supervisor pattern.
    Tries MiniMax M2.1 first, falls back to OpenAI, then Gemini.
    """
    try:
        results = {}
        
        # Try MiniMax M2.1 first (interleaved thinking)
        if MINIMAX_API_KEY:
            logger.info("Attempting MiniMax M2.1 deep research...")
            minimax_result = await call_minimax_deep_research(request)
            if not minimax_result.get("error"):
                results["minimax"] = minimax_result
        
        # Try OpenAI o3-deep-research
        if OPENAI_API_KEY and not results:
            logger.info("Attempting OpenAI deep research...")
            openai_result = await call_openai_deep_research(request)
            if not openai_result.get("error"):
                results["openai"] = openai_result
        
        # Try Gemini as fallback
        if GEMINI_API_KEY and not results:
            logger.info("Attempting Gemini deep research...")
            gemini_result = await call_gemini_deep_research(request)
            if not gemini_result.get("error"):
                results["gemini"] = gemini_result
        
        if not results:
            return {
                "error": "All deep research providers failed",
                "report": "",
                "providers_tried": ["minimax", "openai", "gemini"]
            }
        
        # Return the first successful result
        primary_source = list(results.keys())[0]
        primary_result = results[primary_source]
        primary_result["primary_source"] = primary_source
        primary_result["available_sources"] = list(results.keys())
        
        return primary_result
        
    except Exception as e:
        logger.error(f"Multi-agent deep research error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Multi-Agent Deep Research Integration (dair-ai style)
# ============================================================================

# Import the multi-agent module
import sys
sys.path.insert(0, './agents')

try:
    from multi_agent_research import conduct_multi_agent_research, SupervisorAgent, PlanningAgent, WebSearchRetriever
    MULTI_AGENT_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Multi-agent module not available: {e}")
    MULTI_AGENT_AVAILABLE = False


class MultiAgentResearchRequest(BaseModel):
    """Request model for multi-agent deep research"""
    signal_id: str
    title: str
    abstract: str
    subspecialty: str = "general"
    recommended_methods: List[str] = []
    max_sources: int = 50
    parallel_searches: bool = True
    include_similar: bool = True


@app.post("/multi-agent-research")
async def multi_agent_research_endpoint(request: MultiAgentResearchRequest):
    """
    Perform deep research using the full multi-agent architecture.
    
    This endpoint uses:
    - Supervisor Agent (MiniMax M2.1 with interleaved thinking)
    - Planning Agent (Gemini for query decomposition)
    - Web Search Retriever (Exa API for neural search)
    - Synthesis with fallback chain (MiniMax -> OpenAI -> Claude)
    """
    if not MULTI_AGENT_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Multi-agent research module not available"
        )
    
    try:
        result = await conduct_multi_agent_research(
            title=request.title,
            abstract=request.abstract,
            subspecialty=request.subspecialty,
            recommended_methods=request.recommended_methods,
            max_sources=request.max_sources
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Multi-agent research error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/research-orchestrator")
async def research_orchestrator(request: DeepResearchRequest):
    """
    Unified research orchestrator that combines all available research methods.
    
    Executes in parallel:
    1. Multi-agent research (Supervisor + Planning + Search)
    2. OpenAI o3-deep-research
    3. Gemini Deep Research
    4. MiniMax M2.1 direct
    
    Returns combined results from all available sources.
    """
    results = {
        "signal_id": request.signal_id,
        "title": request.title,
        "sources": {},
        "combined_report": "",
        "all_citations": [],
        "all_key_findings": [],
        "metadata": {
            "started_at": datetime.now().isoformat(),
            "sources_attempted": [],
            "sources_succeeded": []
        }
    }
    
    # Create tasks for parallel execution
    tasks = []
    task_names = []
    
    # Multi-agent research
    if MULTI_AGENT_AVAILABLE:
        async def run_multi_agent():
            return await conduct_multi_agent_research(
                title=request.title,
                abstract=request.abstract,
                subspecialty=request.subspecialty,
                recommended_methods=request.recommended_methods,
                max_sources=request.max_sources
            )
        tasks.append(run_multi_agent())
        task_names.append("multi_agent")
        results["metadata"]["sources_attempted"].append("multi_agent")
    
    # MiniMax M2.1
    if MINIMAX_API_KEY:
        tasks.append(call_minimax_deep_research(request))
        task_names.append("minimax")
        results["metadata"]["sources_attempted"].append("minimax")
    
    # OpenAI
    if OPENAI_API_KEY:
        tasks.append(call_openai_deep_research(request))
        task_names.append("openai")
        results["metadata"]["sources_attempted"].append("openai")
    
    # Gemini
    if GEMINI_API_KEY:
        tasks.append(call_gemini_deep_research(request))
        task_names.append("gemini")
        results["metadata"]["sources_attempted"].append("gemini")
    
    if not tasks:
        return {
            "error": "No research providers configured",
            "sources": {},
            "combined_report": ""
        }
    
    # Execute all tasks in parallel
    try:
        task_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        for name, result in zip(task_names, task_results):
            if isinstance(result, Exception):
                logger.error(f"{name} failed: {str(result)}")
                results["sources"][name] = {"error": str(result)}
            elif result.get("error"):
                results["sources"][name] = {"error": result.get("error")}
            else:
                results["sources"][name] = result
                results["metadata"]["sources_succeeded"].append(name)
                
                # Aggregate citations
                if result.get("citations"):
                    results["all_citations"].extend(result["citations"])
                
                # Aggregate key findings
                if result.get("key_findings"):
                    results["all_key_findings"].extend(result["key_findings"])
        
        # Combine reports from successful sources
        combined_parts = []
        for name in results["metadata"]["sources_succeeded"]:
            source_result = results["sources"].get(name, {})
            report = source_result.get("report", "")
            if report:
                combined_parts.append(f"## Research from {name.upper()}\n\n{report[:3000]}")
        
        results["combined_report"] = "\n\n---\n\n".join(combined_parts)
        
        # Deduplicate citations
        seen_urls = set()
        unique_citations = []
        for c in results["all_citations"]:
            url = c.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_citations.append(c)
        results["all_citations"] = unique_citations[:50]
        
        # Deduplicate key findings
        results["all_key_findings"] = list(set(results["all_key_findings"]))[:20]
        
        results["metadata"]["completed_at"] = datetime.now().isoformat()
        
        return results
        
    except Exception as e:
        logger.error(f"Research orchestrator error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# Update health check to include multi-agent status
@app.get("/health/detailed")
async def detailed_health_check():
    """Detailed health check with all feature availability"""
    return {
        "status": "healthy",
        "version": "2.1.0",
        "timestamp": datetime.now().isoformat(),
        "features": {
            "deep_research": {
                "openai": bool(OPENAI_API_KEY),
                "gemini": bool(GEMINI_API_KEY),
                "minimax": bool(MINIMAX_API_KEY),
                "multi_agent": MULTI_AGENT_AVAILABLE
            },
            "snowballing": True,
            "exa_search": bool(EXA_API_KEY),
            "anthropic": bool(ANTHROPIC_API_KEY)
        },
        "endpoints": [
            "/evaluate",
            "/deep-research",
            "/deep-research-minimax",
            "/deep-research-multi",
            "/multi-agent-research",
            "/research-orchestrator",
            "/snowball",
            "/exa-research",
            "/signals/store",
            "/prospero/generate"
        ]
    }
