"""
NAHN Multi-Agent Deep Research Module
Based on dair-ai/m2-deep-research architecture

This module implements a supervisor-based multi-agent system for comprehensive
medical literature research with:
- Supervisor Agent (MiniMax M2.1 with interleaved thinking)
- Planning Agent (Gemini for query decomposition)
- Web Search Retriever (Exa API for neural search)
- Synthesis Agent (combines findings into comprehensive report)
"""

import os
import json
import asyncio
import httpx
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

class AgentConfig:
    """Configuration for multi-agent system"""
    # API Keys
    MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
    OPENAI_API_BASE = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
    EXA_API_KEY = os.getenv("EXA_API_KEY", "")
    ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
    OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY", "")
    
    # Model settings
    SUPERVISOR_MODEL = "MiniMax-Text-01"
    PLANNING_MODEL = "gemini-2.0-flash"
    SYNTHESIS_MODEL = "MiniMax-Text-01"
    
    # Search settings
    MAX_SUBQUERIES = 5
    MAX_SEARCH_RESULTS = 10
    SEARCH_TIMEOUT = 60


class QueryType(Enum):
    """Types of search queries"""
    RESEARCH = "research"
    NEWS = "news"
    AUTO = "auto"


@dataclass
class SubQuery:
    """A decomposed subquery for research"""
    query: str
    priority: int  # 1-5, 1 being highest
    query_type: QueryType
    rationale: str


@dataclass
class SearchResult:
    """A search result from Exa or other sources"""
    title: str
    url: str
    text: str
    highlights: List[str] = field(default_factory=list)
    score: float = 0.0
    source: str = "exa"


@dataclass
class ResearchReport:
    """Final research report"""
    title: str
    executive_summary: str
    key_takeaways: List[str]
    detailed_analysis: str
    methodology_notes: str
    evidence_gaps: List[str]
    citations: List[Dict[str, str]]
    thinking_trace: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


# ============================================================================
# Planning Agent
# ============================================================================

class PlanningAgent:
    """
    Planning Agent that decomposes research queries into optimized subqueries.
    Uses Gemini for intelligent query planning.
    """
    
    def __init__(self):
        self.api_key = AgentConfig.GEMINI_API_KEY
        self.model = AgentConfig.PLANNING_MODEL
    
    async def decompose_query(self, 
                              main_query: str, 
                              subspecialty: str = "general",
                              context: str = "") -> List[SubQuery]:
        """
        Decompose a research query into 3-5 optimized subqueries.
        
        Args:
            main_query: The main research question
            subspecialty: Neurosurgical subspecialty context
            context: Additional context (abstract, etc.)
        
        Returns:
            List of SubQuery objects with priorities
        """
        if not self.api_key:
            logger.warning("Gemini API key not configured, using fallback decomposition")
            return self._fallback_decomposition(main_query, subspecialty)
        
        prompt = f"""You are a medical research query planner specializing in neurosurgery.

Given the following research topic, decompose it into 3-5 optimized search subqueries.

Research Topic: {main_query}
Subspecialty: {subspecialty}
Additional Context: {context[:500] if context else 'None'}

For each subquery, provide:
1. The search query (optimized for academic/medical search)
2. Priority (1-5, where 1 is most important)
3. Query type (research, news, or auto)
4. Brief rationale

Format your response as JSON:
{{
  "subqueries": [
    {{
      "query": "search query here",
      "priority": 1,
      "query_type": "research",
      "rationale": "why this query is important"
    }}
  ]
}}

Focus on:
- Existing systematic reviews and meta-analyses
- Key RCTs and clinical trials
- Methodological considerations
- Recent developments and controversies
- Subspecialty-specific aspects"""

        try:
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent",
                    headers={"Content-Type": "application/json"},
                    params={"key": self.api_key},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.3,
                            "maxOutputTokens": 2000
                        }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    text = result.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")
                    
                    # Parse JSON from response
                    json_start = text.find("{")
                    json_end = text.rfind("}") + 1
                    if json_start >= 0 and json_end > json_start:
                        parsed = json.loads(text[json_start:json_end])
                        subqueries = []
                        for sq in parsed.get("subqueries", []):
                            subqueries.append(SubQuery(
                                query=sq.get("query", ""),
                                priority=sq.get("priority", 3),
                                query_type=QueryType(sq.get("query_type", "auto")),
                                rationale=sq.get("rationale", "")
                            ))
                        return subqueries
                
                logger.error(f"Gemini API error: {response.status_code}")
                return self._fallback_decomposition(main_query, subspecialty)
                
        except Exception as e:
            logger.error(f"Planning agent error: {str(e)}")
            return self._fallback_decomposition(main_query, subspecialty)
    
    def _fallback_decomposition(self, query: str, subspecialty: str) -> List[SubQuery]:
        """Fallback query decomposition without LLM"""
        return [
            SubQuery(
                query=f"{query} systematic review meta-analysis",
                priority=1,
                query_type=QueryType.RESEARCH,
                rationale="Find existing evidence syntheses"
            ),
            SubQuery(
                query=f"{query} randomized controlled trial RCT",
                priority=2,
                query_type=QueryType.RESEARCH,
                rationale="Find primary RCT evidence"
            ),
            SubQuery(
                query=f"{query} {subspecialty} neurosurgery outcomes",
                priority=3,
                query_type=QueryType.RESEARCH,
                rationale="Subspecialty-specific outcomes"
            ),
            SubQuery(
                query=f"{query} methodology challenges limitations",
                priority=4,
                query_type=QueryType.RESEARCH,
                rationale="Understand methodological considerations"
            )
        ]


# ============================================================================
# Web Search Retriever
# ============================================================================

class WebSearchRetriever:
    """
    Web Search Retriever using Exa API for neural semantic search.
    Finds and retrieves relevant academic content.
    """
    
    def __init__(self):
        self.api_key = AgentConfig.EXA_API_KEY
        self.medical_domains = [
            "pubmed.ncbi.nlm.nih.gov",
            "ncbi.nlm.nih.gov",
            "cochranelibrary.com",
            "jamanetwork.com",
            "thelancet.com",
            "nejm.org",
            "bmj.com",
            "nature.com",
            "sciencedirect.com",
            "springer.com",
            "wiley.com"
        ]
    
    async def search(self, 
                     query: str, 
                     query_type: QueryType = QueryType.AUTO,
                     num_results: int = 10) -> List[SearchResult]:
        """
        Perform neural semantic search using Exa API.
        
        Args:
            query: Search query
            query_type: Type of search (research, news, auto)
            num_results: Maximum number of results
        
        Returns:
            List of SearchResult objects
        """
        if not self.api_key:
            logger.warning("Exa API key not configured")
            return []
        
        try:
            async with httpx.AsyncClient(timeout=AgentConfig.SEARCH_TIMEOUT) as client:
                search_type = "neural" if query_type == QueryType.RESEARCH else "auto"
                
                response = await client.post(
                    "https://api.exa.ai/search",
                    headers={
                        "x-api-key": self.api_key,
                        "Content-Type": "application/json"
                    },
                    json={
                        "query": query,
                        "type": search_type,
                        "numResults": num_results,
                        "includeDomains": self.medical_domains,
                        "contents": {
                            "text": {"maxCharacters": 2000},
                            "highlights": True
                        }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    results = []
                    for r in result.get("results", []):
                        results.append(SearchResult(
                            title=r.get("title", ""),
                            url=r.get("url", ""),
                            text=r.get("text", ""),
                            highlights=r.get("highlights", []),
                            score=r.get("score", 0.0),
                            source="exa"
                        ))
                    return results
                else:
                    logger.error(f"Exa API error: {response.status_code}")
                    return []
                    
        except Exception as e:
            logger.error(f"Web search error: {str(e)}")
            return []
    
    async def find_similar(self, url: str, num_results: int = 5) -> List[SearchResult]:
        """Find similar content to a given URL"""
        if not self.api_key:
            return []
        
        try:
            async with httpx.AsyncClient(timeout=AgentConfig.SEARCH_TIMEOUT) as client:
                response = await client.post(
                    "https://api.exa.ai/findSimilar",
                    headers={
                        "x-api-key": self.api_key,
                        "Content-Type": "application/json"
                    },
                    json={
                        "url": url,
                        "numResults": num_results,
                        "includeDomains": self.medical_domains,
                        "contents": {
                            "text": {"maxCharacters": 1000},
                            "highlights": True
                        }
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return [
                        SearchResult(
                            title=r.get("title", ""),
                            url=r.get("url", ""),
                            text=r.get("text", ""),
                            highlights=r.get("highlights", []),
                            score=r.get("score", 0.0),
                            source="exa_similar"
                        )
                        for r in result.get("results", [])
                    ]
                return []
                
        except Exception as e:
            logger.error(f"Find similar error: {str(e)}")
            return []


# ============================================================================
# Supervisor Agent
# ============================================================================

class SupervisorAgent:
    """
    Supervisor Agent that orchestrates the multi-agent research workflow.
    Uses MiniMax M2.1 with interleaved thinking for coordination and synthesis.
    """
    
    def __init__(self):
        self.api_key = AgentConfig.MINIMAX_API_KEY
        self.model = AgentConfig.SUPERVISOR_MODEL
        self.planning_agent = PlanningAgent()
        self.search_retriever = WebSearchRetriever()
        self.conversation_history = []
    
    async def conduct_research(self,
                               title: str,
                               abstract: str,
                               subspecialty: str = "general",
                               recommended_methods: List[str] = None,
                               max_sources: int = 50) -> ResearchReport:
        """
        Conduct comprehensive research using multi-agent orchestration.
        
        Args:
            title: Research topic title
            abstract: Background/abstract
            subspecialty: Neurosurgical subspecialty
            recommended_methods: Suggested meta-analytic methods
            max_sources: Maximum sources to include
        
        Returns:
            ResearchReport with comprehensive findings
        """
        logger.info(f"Starting multi-agent research: {title[:50]}...")
        
        # Phase 1: Query Planning
        logger.info("Phase 1: Query decomposition...")
        subqueries = await self.planning_agent.decompose_query(
            main_query=title,
            subspecialty=subspecialty,
            context=abstract
        )
        logger.info(f"Generated {len(subqueries)} subqueries")
        
        # Phase 2: Parallel Web Search
        logger.info("Phase 2: Executing parallel searches...")
        all_results = []
        search_tasks = []
        
        for sq in sorted(subqueries, key=lambda x: x.priority):
            search_tasks.append(
                self.search_retriever.search(
                    query=sq.query,
                    query_type=sq.query_type,
                    num_results=AgentConfig.MAX_SEARCH_RESULTS
                )
            )
        
        # Execute searches in parallel
        search_results = await asyncio.gather(*search_tasks, return_exceptions=True)
        
        for i, results in enumerate(search_results):
            if isinstance(results, Exception):
                logger.error(f"Search {i} failed: {str(results)}")
                continue
            all_results.extend(results)
        
        logger.info(f"Retrieved {len(all_results)} total results")
        
        # Phase 3: Find similar content for top results
        logger.info("Phase 3: Expanding with similar content...")
        if all_results:
            top_results = sorted(all_results, key=lambda x: x.score, reverse=True)[:3]
            similar_tasks = [
                self.search_retriever.find_similar(r.url)
                for r in top_results if r.url
            ]
            if similar_tasks:
                similar_results = await asyncio.gather(*similar_tasks, return_exceptions=True)
                for results in similar_results:
                    if not isinstance(results, Exception):
                        all_results.extend(results)
        
        # Deduplicate results
        seen_urls = set()
        unique_results = []
        for r in all_results:
            if r.url not in seen_urls:
                seen_urls.add(r.url)
                unique_results.append(r)
        
        logger.info(f"Total unique sources: {len(unique_results)}")
        
        # Phase 4: Synthesis with Supervisor
        logger.info("Phase 4: Synthesizing findings...")
        report = await self._synthesize_report(
            title=title,
            abstract=abstract,
            subspecialty=subspecialty,
            recommended_methods=recommended_methods or [],
            search_results=unique_results[:max_sources],
            subqueries=subqueries
        )
        
        return report
    
    async def _synthesize_report(self,
                                  title: str,
                                  abstract: str,
                                  subspecialty: str,
                                  recommended_methods: List[str],
                                  search_results: List[SearchResult],
                                  subqueries: List[SubQuery]) -> ResearchReport:
        """Synthesize a comprehensive research report from gathered evidence"""
        
        # Prepare context from search results
        evidence_context = self._prepare_evidence_context(search_results)
        
        system_prompt = f"""You are a senior medical researcher and systematic review expert specializing in neurosurgery.
Your task is to synthesize a comprehensive research report based on the provided evidence.

You have access to findings from multiple search queries on the topic.
Use your interleaved thinking capability to reason through the evidence systematically.

Subspecialty: {subspecialty}
Recommended meta-analytic methods: {', '.join(recommended_methods) if recommended_methods else 'To be determined based on evidence'}

Structure your report with:
1. Executive Summary (2-3 paragraphs)
2. Key Takeaways (5-7 bullet points)
3. Detailed Analysis (organized by theme)
4. Methodological Considerations
5. Evidence Gaps and Future Directions
6. Cited Sources

Be thorough, cite sources inline, and maintain academic rigor."""

        user_prompt = f"""Research Topic: {title}

Background: {abstract}

Search Queries Executed:
{chr(10).join([f"- {sq.query} (Priority: {sq.priority})" for sq in subqueries])}

Evidence Retrieved:
{evidence_context}

Please synthesize a comprehensive research report on this topic."""

        # Try MiniMax first, then fallbacks
        report_text, thinking_trace = await self._call_synthesis_llm(system_prompt, user_prompt)
        
        # Parse the report
        return self._parse_report(
            title=title,
            report_text=report_text,
            thinking_trace=thinking_trace,
            search_results=search_results,
            subqueries=subqueries
        )
    
    def _prepare_evidence_context(self, results: List[SearchResult]) -> str:
        """Prepare evidence context from search results"""
        context_parts = []
        for i, r in enumerate(results[:30], 1):  # Limit to top 30
            highlights = " | ".join(r.highlights[:3]) if r.highlights else ""
            context_parts.append(f"""
[Source {i}]
Title: {r.title}
URL: {r.url}
Content: {r.text[:500]}...
Key Points: {highlights}
""")
        return "\n".join(context_parts)
    
    async def _call_synthesis_llm(self, system_prompt: str, user_prompt: str) -> tuple:
        """Call LLM for synthesis with fallback chain"""
        thinking_trace = []
        
        # Try MiniMax M2.1 first
        if AgentConfig.MINIMAX_API_KEY:
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    response = await client.post(
                        "https://api.minimax.chat/v1/text/chatcompletion_v2",
                        headers={
                            "Authorization": f"Bearer {AgentConfig.MINIMAX_API_KEY}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": AgentConfig.SYNTHESIS_MODEL,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            "max_tokens": 16000,
                            "temperature": 0.7
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                        
                        # Extract thinking blocks
                        if "<thinking>" in content:
                            import re
                            thinking_trace = re.findall(r'<thinking>(.*?)</thinking>', content, re.DOTALL)
                            content = re.sub(r'<thinking>.*?</thinking>', '', content, flags=re.DOTALL)
                        
                        return content.strip(), thinking_trace
            except Exception as e:
                logger.error(f"MiniMax synthesis error: {str(e)}")
        
        # Try OpenAI
        if AgentConfig.OPENAI_API_KEY:
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    response = await client.post(
                        f"{AgentConfig.OPENAI_API_BASE}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {AgentConfig.OPENAI_API_KEY}",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "gpt-4o",
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt}
                            ],
                            "max_tokens": 8000,
                            "temperature": 0.7
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        content = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                        return content, []
            except Exception as e:
                logger.error(f"OpenAI synthesis error: {str(e)}")
        
        # Try Anthropic Claude
        if AgentConfig.ANTHROPIC_API_KEY:
            try:
                async with httpx.AsyncClient(timeout=300.0) as client:
                    response = await client.post(
                        "https://api.anthropic.com/v1/messages",
                        headers={
                            "x-api-key": AgentConfig.ANTHROPIC_API_KEY,
                            "anthropic-version": "2023-06-01",
                            "Content-Type": "application/json"
                        },
                        json={
                            "model": "claude-3-5-sonnet-20241022",
                            "max_tokens": 8000,
                            "system": system_prompt,
                            "messages": [
                                {"role": "user", "content": user_prompt}
                            ]
                        }
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        content = result.get("content", [{}])[0].get("text", "")
                        return content, []
            except Exception as e:
                logger.error(f"Anthropic synthesis error: {str(e)}")
        
        return "Unable to generate synthesis - no LLM providers available", []
    
    def _parse_report(self,
                      title: str,
                      report_text: str,
                      thinking_trace: List[str],
                      search_results: List[SearchResult],
                      subqueries: List[SubQuery]) -> ResearchReport:
        """Parse the generated report into structured format"""
        
        # Extract sections from report
        sections = {
            "executive_summary": "",
            "key_takeaways": [],
            "detailed_analysis": "",
            "methodology_notes": "",
            "evidence_gaps": []
        }
        
        # Simple section extraction
        lines = report_text.split('\n')
        current_section = None
        current_content = []
        
        for line in lines:
            line_lower = line.lower().strip()
            
            if 'executive summary' in line_lower:
                current_section = 'executive_summary'
                current_content = []
            elif 'key takeaway' in line_lower:
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'key_takeaways'
                current_content = []
            elif 'detailed analysis' in line_lower or 'analysis' in line_lower:
                if current_section and current_content:
                    if current_section == 'key_takeaways':
                        sections[current_section] = [c.strip('- •') for c in current_content if c.strip()]
                    else:
                        sections[current_section] = '\n'.join(current_content)
                current_section = 'detailed_analysis'
                current_content = []
            elif 'methodolog' in line_lower:
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'methodology_notes'
                current_content = []
            elif 'evidence gap' in line_lower or 'future direction' in line_lower:
                if current_section and current_content:
                    sections[current_section] = '\n'.join(current_content)
                current_section = 'evidence_gaps'
                current_content = []
            elif current_section:
                current_content.append(line)
        
        # Handle last section
        if current_section and current_content:
            if current_section in ['key_takeaways', 'evidence_gaps']:
                sections[current_section] = [c.strip('- •') for c in current_content if c.strip()]
            else:
                sections[current_section] = '\n'.join(current_content)
        
        # Build citations from search results
        citations = [
            {"title": r.title, "url": r.url, "source": r.source}
            for r in search_results[:20]
        ]
        
        return ResearchReport(
            title=title,
            executive_summary=sections.get('executive_summary', report_text[:1000]),
            key_takeaways=sections.get('key_takeaways', [])[:10] if isinstance(sections.get('key_takeaways'), list) else [],
            detailed_analysis=sections.get('detailed_analysis', report_text),
            methodology_notes=sections.get('methodology_notes', ''),
            evidence_gaps=sections.get('evidence_gaps', [])[:5] if isinstance(sections.get('evidence_gaps'), list) else [],
            citations=citations,
            thinking_trace=thinking_trace,
            metadata={
                "subqueries_used": len(subqueries),
                "sources_retrieved": len(search_results),
                "generated_at": datetime.now().isoformat(),
                "agents_used": ["planning", "search", "supervisor"]
            }
        )


# ============================================================================
# Main Research Function
# ============================================================================

async def conduct_multi_agent_research(
    title: str,
    abstract: str,
    subspecialty: str = "general",
    recommended_methods: List[str] = None,
    max_sources: int = 50
) -> Dict[str, Any]:
    """
    Main entry point for multi-agent deep research.
    
    Args:
        title: Research topic
        abstract: Background/abstract
        subspecialty: Neurosurgical subspecialty
        recommended_methods: Suggested methods
        max_sources: Maximum sources
    
    Returns:
        Dictionary with research report and metadata
    """
    supervisor = SupervisorAgent()
    
    try:
        report = await supervisor.conduct_research(
            title=title,
            abstract=abstract,
            subspecialty=subspecialty,
            recommended_methods=recommended_methods,
            max_sources=max_sources
        )
        
        return {
            "success": True,
            "report": report.detailed_analysis,
            "executive_summary": report.executive_summary,
            "key_findings": report.key_takeaways,
            "methodology_recommendations": [report.methodology_notes] if report.methodology_notes else [],
            "evidence_gaps": report.evidence_gaps,
            "citations": report.citations,
            "thinking_trace": report.thinking_trace,
            "metadata": report.metadata,
            "source": "multi_agent_supervisor"
        }
        
    except Exception as e:
        logger.error(f"Multi-agent research failed: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "report": "",
            "source": "multi_agent_supervisor"
        }


# ============================================================================
# Test Function
# ============================================================================

async def test_multi_agent():
    """Test the multi-agent research system"""
    result = await conduct_multi_agent_research(
        title="VP Shunt vs Endoscopic Third Ventriculostomy for Idiopathic Normal Pressure Hydrocephalus",
        abstract="Comparison of surgical treatments for iNPH in elderly patients.",
        subspecialty="hydrocephalus",
        recommended_methods=["pairwise", "nma"],
        max_sources=20
    )
    
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(test_multi_agent())
