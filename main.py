# main.py — AgenticOps FastAPI Entry Point

import os
import logging
from dotenv import load_dotenv

load_dotenv()

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from graph import graph

logger = logging.getLogger("AgenticOps")

app = FastAPI(
    title="AgenticOps API",
    description="Multi-agent AI for IT Operations — Incident Detection & Resolution",
    version="1.0.0",
)


class RequestModel(BaseModel):
    time_stamp: str
    issue: str


@app.get("/")
def home():
    return {
        "message": "AgenticOps API is running",
        "endpoints": {
            "POST /run-agent": "Run the multi-agent analysis pipeline",
            "GET /health": "Health check",
        },
    }


@app.get("/health")
def health():
    return {"status": "healthy", "agents": ["commander", "metrics", "logs", "cicd", "resolver", "reporter"]}


@app.post("/run-agent")
async def run_agent(request: RequestModel):
    """
    Run the full AgenticOps multi-agent pipeline.
    
    The pipeline flow:
      commander → metrics → logs → cicd → resolver → reporter → END
    
    Each agent analyzes a different aspect of the system and contributes
    to the final incident report.
    """
    logger.info("POST /run-agent called | issue=%s | time_stamp=%s", request.issue, request.time_stamp)
    try:
        result = await graph.ainvoke({
            "issue": request.issue,
            "time_stamp": request.time_stamp,
            "agents_called": [],
        })
        logger.info("Pipeline completed | agents_called=%s", result.get("agents_called", []))

        return JSONResponse(
            status_code=200,
            content={
                "status": "success",
                "issue": request.issue,
                "time_stamp": request.time_stamp,
                "response": result.get("output", "No output generated"),
                "agents_called": result.get("agents_called", []),
                "reports": {
                    "metrics": result.get("metrics_report", ""),
                    "logs": result.get("logs_report", ""),
                    "cicd": result.get("cicd_report", ""),
                    "resolution": result.get("resolution_report", ""),
                    "final": result.get("final_report", ""),
                },
            },
        )
    except Exception as e:
        logger.exception("Pipeline failed: %s", e)
        return JSONResponse(
            status_code=500,
            content={"error": str(e), "type": type(e).__name__},
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)