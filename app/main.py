"""
India Trend Lead Generation Agent - Main Entry Point.
FastAPI server and CLI interface.
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel

from .config import get_settings
from .schemas import PipelineResult
from .database import get_database
from .agents.orchestrator import run_pipeline
from .tools.llm_tool import LLMTool

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# Suppress noisy loggers
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

# Initialize FastAPI app
app = FastAPI(
    title="India Trend Lead Generation Agent",
    description="AI-powered market trend detection and B2B lead generation for Indian companies",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for tracking pipeline runs
pipeline_status = {
    "running": False,
    "last_run": None,
    "last_result": None
}


class RunConfig(BaseModel):
    """Configuration for pipeline run."""
    mock_mode: bool = False
    max_trends: Optional[int] = None
    max_companies: Optional[int] = None


class StatusResponse(BaseModel):
    """Status response model."""
    status: str
    running: bool
    last_run: Optional[str]
    message: str


@app.on_event("startup")
async def startup_event():
    """Initialize database and check LLM availability on startup."""
    logger.info("🚀 Starting India Trend Lead Generation Agent...")
    
    # Initialize database
    try:
        db = get_database()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
    
    # Check LLM providers
    settings = get_settings()
    llm_tool = LLMTool()
    status = await llm_tool.get_provider_status()
    
    if status["ollama"]:
        logger.info(f"✅ Ollama available ({settings.ollama_model})")
    else:
        logger.warning("⚠️ Ollama not available")
    
    if status["gemini"]:
        logger.info("✅ Gemini API configured")
    else:
        logger.warning("⚠️ Gemini API not configured")
    
    if not status["any_available"]:
        logger.error("❌ No LLM providers available!")


@app.get("/", response_model=StatusResponse)
async def root():
    """Health check and status endpoint."""
    return StatusResponse(
        status="healthy",
        running=pipeline_status["running"],
        last_run=pipeline_status["last_run"],
        message="India Trend Lead Generation Agent is running"
    )


@app.get("/health")
async def health_check():
    """Detailed health check."""
    settings = get_settings()
    llm_tool = LLMTool()
    llm_status = await llm_tool.get_provider_status()
    
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "llm_providers": llm_status,
        "config": {
            "use_ollama": settings.use_ollama,
            "mock_mode": settings.mock_mode,
            "max_trends": settings.max_trends,
            "max_companies_per_trend": settings.max_companies_per_trend
        }
    }


@app.post("/run", response_model=PipelineResult)
async def run_full_pipeline(
    config: RunConfig = RunConfig(),
    background_tasks: BackgroundTasks = None
):
    """
    Run the full lead generation pipeline.
    
    This endpoint triggers:
    1. Trend detection from RSS + Tavily
    2. Impact analysis
    3. Company discovery
    4. Contact finding
    5. Email generation
    
    Returns:
        PipelineResult with generated leads
    """
    if pipeline_status["running"]:
        raise HTTPException(
            status_code=409,
            detail="Pipeline is already running. Please wait for completion."
        )
    
    pipeline_status["running"] = True
    pipeline_status["last_run"] = datetime.utcnow().isoformat()
    
    try:
        logger.info("📝 Starting pipeline run...")
        logger.info(f"   Mock mode: {config.mock_mode}")
        
        result = await run_pipeline(mock_mode=config.mock_mode)
        
        pipeline_status["last_result"] = result.model_dump()
        return result
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
    finally:
        pipeline_status["running"] = False


@app.get("/results")
async def get_latest_results():
    """Get the most recent pipeline results."""
    if pipeline_status["last_result"]:
        return pipeline_status["last_result"]
    
    # Try to get from database
    db = get_database()
    latest_run = db.get_latest_run()
    
    if latest_run:
        return latest_run
    
    return {"message": "No results available. Run the pipeline first."}


@app.get("/leads")
async def get_leads(limit: int = 50):
    """Get the most recent leads from the database."""
    db = get_database()
    leads = db.get_latest_leads(limit=limit)
    
    return {
        "count": len(leads),
        "leads": leads
    }


@app.get("/outputs/{filename}")
async def get_output_file(filename: str):
    """Download an output file."""
    outputs_dir = Path("app/outputs")
    file_path = outputs_dir / filename
    
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        path=file_path,
        filename=filename,
        media_type="application/json" if filename.endswith(".json") else "text/csv"
    )


@app.get("/outputs")
async def list_outputs():
    """List all output files."""
    outputs_dir = Path("app/outputs")
    
    if not outputs_dir.exists():
        return {"files": []}
    
    files = []
    for f in sorted(outputs_dir.iterdir(), reverse=True):
        if f.suffix in [".json", ".csv"]:
            files.append({
                "name": f.name,
                "size": f.stat().st_size,
                "modified": datetime.fromtimestamp(f.stat().st_mtime).isoformat()
            })
    
    return {"files": files[:20]}  # Return last 20 files


# CLI Runner
async def cli_main():
    """Command-line interface for running the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="India Trend Lead Generation Agent"
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Run in mock mode (no real API calls)"
    )
    parser.add_argument(
        "--server",
        action="store_true",
        help="Start the FastAPI server"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )
    
    args = parser.parse_args()
    
    if args.server:
        import uvicorn
        logger.info(f"Starting server on port {args.port}...")
        uvicorn.run(app, host="0.0.0.0", port=args.port)
    else:
        # Run pipeline directly
        print("\n" + "=" * 60)
        print("🇮🇳 INDIA TREND LEAD GENERATION AGENT")
        print("=" * 60 + "\n")
        
        if args.mock:
            print("🔧 Running in MOCK MODE (no real API calls)\n")
        
        result = await run_pipeline(mock_mode=args.mock)
        
        print("\n" + "=" * 60)
        print("📊 FINAL RESULTS")
        print("=" * 60)
        print(f"Status: {result.status}")
        print(f"Trends detected: {result.trends_detected}")
        print(f"Companies found: {result.companies_found}")
        print(f"Emails found: {result.emails_found}")
        print(f"Leads generated: {result.leads_generated}")
        print(f"Output file: {result.output_file}")
        print(f"Runtime: {result.run_time_seconds:.2f}s")
        
        if result.errors:
            print(f"\n⚠️ Errors: {len(result.errors)}")
            for error in result.errors[:5]:
                print(f"   - {error}")
        
        print("=" * 60 + "\n")


def main():
    """Entry point for CLI."""
    asyncio.run(cli_main())


if __name__ == "__main__":
    main()
