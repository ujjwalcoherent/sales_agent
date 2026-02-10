"""
LangGraph Orchestrator.
Coordinates the agent pipeline: Trend → Impact → Company → Contact → Email
"""

import logging
import json
import csv
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, START, END

from ..schemas import AgentState, PipelineResult
from ..database import get_database
from .trend_agent import run_trend_agent
from .impact_agent import run_impact_agent
from .company_agent import run_company_agent
from .contact_agent import run_contact_agent
from .email_agent import run_email_agent

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """State type for LangGraph."""
    trends: list
    impacts: list
    companies: list
    contacts: list
    outreach_emails: list
    errors: list
    current_step: str


def create_pipeline_graph():
    """
    Create the LangGraph pipeline.
    
    Flow:
    Trend Agent → Impact Agent → Company Agent → Contact Agent → Email Agent
    """
    
    # Create the graph
    workflow = StateGraph(GraphState)
    
    # Add nodes for each agent
    workflow.add_node("trend_detection", trend_node)
    workflow.add_node("impact_analysis", impact_node)
    workflow.add_node("company_finding", company_node)
    workflow.add_node("contact_finding", contact_node)
    workflow.add_node("email_generation", email_node)
    
    # Define the flow
    workflow.add_edge(START, "trend_detection")
    workflow.add_edge("trend_detection", "impact_analysis")
    workflow.add_edge("impact_analysis", "company_finding")
    workflow.add_edge("company_finding", "contact_finding")
    workflow.add_edge("contact_finding", "email_generation")
    workflow.add_edge("email_generation", END)
    
    # Compile the graph
    return workflow.compile()


async def trend_node(state: GraphState) -> GraphState:
    """Run trend detection agent."""
    logger.info("=" * 50)
    logger.info("STEP 1: TREND DETECTION")
    logger.info("=" * 50)
    
    agent_state = AgentState(**state)
    result = await run_trend_agent(agent_state)
    
    return {
        "trends": [t.model_dump() for t in result.trends],
        "impacts": state.get("impacts", []),
        "companies": state.get("companies", []),
        "contacts": state.get("contacts", []),
        "outreach_emails": state.get("outreach_emails", []),
        "errors": result.errors,
        "current_step": result.current_step
    }


async def impact_node(state: GraphState) -> GraphState:
    """Run impact analysis agent."""
    logger.info("=" * 50)
    logger.info("STEP 2: IMPACT ANALYSIS")
    logger.info("=" * 50)
    
    # Reconstruct agent state
    from ..schemas import TrendData
    trends = [TrendData(**t) for t in state.get("trends", [])]
    
    agent_state = AgentState(
        trends=trends,
        errors=state.get("errors", [])
    )
    result = await run_impact_agent(agent_state)
    
    return {
        "trends": state.get("trends", []),
        "impacts": [i.model_dump() for i in result.impacts],
        "companies": state.get("companies", []),
        "contacts": state.get("contacts", []),
        "outreach_emails": state.get("outreach_emails", []),
        "errors": result.errors,
        "current_step": result.current_step
    }


async def company_node(state: GraphState) -> GraphState:
    """Run company finder agent."""
    logger.info("=" * 50)
    logger.info("STEP 3: COMPANY FINDING")
    logger.info("=" * 50)
    
    from ..schemas import TrendData, ImpactAnalysis
    trends = [TrendData(**t) for t in state.get("trends", [])]
    impacts = [ImpactAnalysis(**i) for i in state.get("impacts", [])]
    
    agent_state = AgentState(
        trends=trends,
        impacts=impacts,
        errors=state.get("errors", [])
    )
    result = await run_company_agent(agent_state)
    
    return {
        "trends": state.get("trends", []),
        "impacts": state.get("impacts", []),
        "companies": [c.model_dump() for c in result.companies],
        "contacts": state.get("contacts", []),
        "outreach_emails": state.get("outreach_emails", []),
        "errors": result.errors,
        "current_step": result.current_step
    }


async def contact_node(state: GraphState) -> GraphState:
    """Run contact finder agent."""
    logger.info("=" * 50)
    logger.info("STEP 4: CONTACT FINDING")
    logger.info("=" * 50)
    
    from ..schemas import TrendData, ImpactAnalysis, CompanyData
    trends = [TrendData(**t) for t in state.get("trends", [])]
    impacts = [ImpactAnalysis(**i) for i in state.get("impacts", [])]
    companies = [CompanyData(**c) for c in state.get("companies", [])]
    
    agent_state = AgentState(
        trends=trends,
        impacts=impacts,
        companies=companies,
        errors=state.get("errors", [])
    )
    result = await run_contact_agent(agent_state)
    
    return {
        "trends": state.get("trends", []),
        "impacts": state.get("impacts", []),
        "companies": state.get("companies", []),
        "contacts": [c.model_dump() for c in result.contacts],
        "outreach_emails": state.get("outreach_emails", []),
        "errors": result.errors,
        "current_step": result.current_step
    }


async def email_node(state: GraphState) -> GraphState:
    """Run email generation agent."""
    logger.info("=" * 50)
    logger.info("STEP 5: EMAIL GENERATION")
    logger.info("=" * 50)
    
    from ..schemas import TrendData, ImpactAnalysis, CompanyData, ContactData
    trends = [TrendData(**t) for t in state.get("trends", [])]
    impacts = [ImpactAnalysis(**i) for i in state.get("impacts", [])]
    companies = [CompanyData(**c) for c in state.get("companies", [])]
    contacts = [ContactData(**c) for c in state.get("contacts", [])]
    
    agent_state = AgentState(
        trends=trends,
        impacts=impacts,
        companies=companies,
        contacts=contacts,
        errors=state.get("errors", [])
    )
    result = await run_email_agent(agent_state)
    
    return {
        "trends": state.get("trends", []),
        "impacts": state.get("impacts", []),
        "companies": state.get("companies", []),
        "contacts": [c.model_dump() for c in result.contacts],  # Updated with emails
        "outreach_emails": [e.model_dump() for e in result.outreach_emails],
        "errors": result.errors,
        "current_step": result.current_step
    }


async def run_pipeline(mock_mode: bool = False) -> PipelineResult:
    """
    Execute the full lead generation pipeline.
    
    Args:
        mock_mode: If True, use mock data instead of real APIs
        
    Returns:
        PipelineResult with all generated leads
    """
    start_time = datetime.utcnow()
    run_id = start_time.strftime("%Y%m%d_%H%M%S")
    
    logger.info("🚀 Starting India Trend Lead Generation Pipeline")
    logger.info(f"📅 Run ID: {run_id}")
    logger.info(f"🔧 Mock Mode: {mock_mode}")
    
    # Initialize state
    initial_state: GraphState = {
        "trends": [],
        "impacts": [],
        "companies": [],
        "contacts": [],
        "outreach_emails": [],
        "errors": [],
        "current_step": "init"
    }
    
    try:
        # Create and run the pipeline
        graph = create_pipeline_graph()
        final_state = await graph.ainvoke(initial_state)
        
        # Calculate statistics
        trends_count = len(final_state.get("trends", []))
        companies_count = len(final_state.get("companies", []))
        contacts_count = len(final_state.get("contacts", []))
        emails_count = len([c for c in final_state.get("contacts", []) if c.get("email")])
        outreach_count = len(final_state.get("outreach_emails", []))
        
        # Save outputs
        output_file = await save_outputs(final_state, run_id)
        
        # Calculate runtime
        end_time = datetime.utcnow()
        runtime = (end_time - start_time).total_seconds()
        
        logger.info("=" * 50)
        logger.info("✅ PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"📊 Trends detected: {trends_count}")
        logger.info(f"🏢 Companies found: {companies_count}")
        logger.info(f"👤 Contacts found: {contacts_count}")
        logger.info(f"📧 Emails found: {emails_count}")
        logger.info(f"✉️ Outreach emails: {outreach_count}")
        logger.info(f"⏱️ Runtime: {runtime:.2f}s")
        logger.info(f"📁 Output: {output_file}")
        logger.info("=" * 50)
        
        return PipelineResult(
            status="success",
            leads_generated=outreach_count,
            trends_detected=trends_count,
            companies_found=companies_count,
            emails_found=emails_count,
            output_file=output_file,
            errors=final_state.get("errors", []),
            run_time_seconds=runtime
        )
        
    except Exception as e:
        logger.error(f"❌ Pipeline failed: {e}")
        return PipelineResult(
            status="error",
            errors=[str(e)],
            run_time_seconds=(datetime.utcnow() - start_time).total_seconds()
        )


async def save_outputs(state: GraphState, run_id: str) -> str:
    """
    Save pipeline outputs to JSON and CSV files.
    
    Args:
        state: Final pipeline state
        run_id: Unique run identifier
        
    Returns:
        Path to output JSON file
    """
    # Ensure outputs directory exists
    outputs_dir = Path("app/outputs")
    outputs_dir.mkdir(parents=True, exist_ok=True)
    
    # Build final leads structure
    leads = []
    
    # Create lookup maps
    trend_map = {t["id"]: t for t in state.get("trends", [])}
    impact_map = {i["trend_id"]: i for i in state.get("impacts", [])}
    company_map = {c["id"]: c for c in state.get("companies", [])}
    contact_map = {c["id"]: c for c in state.get("contacts", [])}
    
    for outreach in state.get("outreach_emails", []):
        contact = contact_map.get(outreach.get("contact_id", ""), {})
        company = company_map.get(contact.get("company_id", ""), {})
        trend_id = company.get("trend_id", "")
        trend = trend_map.get(trend_id, {})
        impact = impact_map.get(trend_id, {})
        
        lead = {
            "id": outreach.get("id"),
            "trend": {
                "title": trend.get("trend_title", outreach.get("trend_title")),
                "summary": trend.get("summary", ""),
                "severity": trend.get("severity", "medium"),
                "industries": trend.get("industries_affected", [])
            },
            "company": {
                "name": company.get("company_name", outreach.get("company_name")),
                "size": company.get("company_size", "mid"),
                "industry": company.get("industry", ""),
                "website": company.get("website", ""),
                "domain": company.get("domain", "")
            },
            "contact": {
                "name": contact.get("person_name", outreach.get("person_name")),
                "role": contact.get("role", outreach.get("role")),
                "email": contact.get("email", outreach.get("email")),
                "email_confidence": contact.get("email_confidence", outreach.get("email_confidence")),
                "email_source": contact.get("email_source", ""),
                "linkedin": contact.get("linkedin_url", "")
            },
            "outreach": {
                "subject": outreach.get("subject", ""),
                "body": outreach.get("body", "")
            },
            "generated_at": outreach.get("generated_at", datetime.utcnow().isoformat())
        }
        leads.append(lead)
    
    # Save JSON
    json_file = outputs_dir / f"leads_{run_id}.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump({
            "run_id": run_id,
            "generated_at": datetime.utcnow().isoformat(),
            "total_leads": len(leads),
            "leads": leads
        }, f, indent=2, ensure_ascii=False)
    
    # Save CSV
    csv_file = outputs_dir / f"leads_{run_id}.csv"
    if leads:
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            # Header
            writer.writerow([
                "Trend", "Company", "Industry", "Website", "Domain",
                "Contact Name", "Role", "Email", "Email Confidence",
                "Subject", "Body Preview"
            ])
            # Data
            for lead in leads:
                writer.writerow([
                    lead["trend"]["title"],
                    lead["company"]["name"],
                    lead["company"]["industry"],
                    lead["company"]["website"],
                    lead["company"]["domain"],
                    lead["contact"]["name"],
                    lead["contact"]["role"],
                    lead["contact"]["email"],
                    lead["contact"]["email_confidence"],
                    lead["outreach"]["subject"],
                    lead["outreach"]["body"][:100] + "..."
                ])
    
    # Also save to database
    try:
        db = get_database()
        for lead in leads:
            db.save_lead({
                "id": lead["id"],
                "trend": lead["trend"],
                "impact": impact_map.get(trend_id, {}),
                "company": lead["company"],
                "contact": lead["contact"],
                "outreach": lead["outreach"]
            })
    except Exception as e:
        logger.warning(f"Failed to save leads to database: {e}")
    
    return str(json_file)
