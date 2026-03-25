# graph.py

from typing import TypedDict
from langgraph.graph import StateGraph
from langchain_community.chat_models import ChatOllama
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from typing_extensions import Literal

# from tools import cpu_analyzer, memory_analyzer, latency_analyzer

# Define state
class AgentState(TypedDict):
    input: str
    output: str

# Local LLM (Ollama)
llm = ChatOllama(model="llama3")

# Commander logic
async def commander_agent(state: AgentState) -> Command[Literal['metrics', 'logs', 'cicd','resolver','reporter','__end__']]:

    return Command(update="",goto="")

#Metrics Agent
async def metrics_agent(state: AgentState) -> Command(Literal['commander']):
    
    return Command(update="",goto="")

# Logs Agent
#async def logs_agent(state: AgentState) -> Command(Literal['commander']):
    
#    return Command(update="",goto="")
import json
from collections import defaultdict
from datetime import datetime

#LOG_FILE = "logs/application_logs.json"

async def logs_agent(state: AgentState) -> Command[Literal['commander']]:
    
    findings = []
    trace_map = defaultdict(list)

    try:
        with open(LOG_FILE, "r") as f:
            logs = json.load(f)

        # Group logs by trace_id
        for entry in logs:
            trace_id = entry.get("trace_id", "unknown")
            trace_map[trace_id].append(entry)

        # Analyze each trace
        for trace_id, events in trace_map.items():
            events_sorted = sorted(events, key=lambda x: x["timestamp"])

            warnings = []
            errors = []

            for e in events_sorted:
                level = e.get("level", "")
                msg = e.get("message", "")

                if level == "ERROR":
                    errors.append(msg)
                elif level == "WARN":
                    warnings.append(msg)

            # Detect suspicious patterns
            if errors:
                findings.append({
                    "trace_id": trace_id,
                    "issue": "error_detected",
                    "details": errors
                })

            if len(warnings) >= 2:
                findings.append({
                    "trace_id": trace_id,
                    "issue": "repeated_warnings",
                    "details": warnings
                })

            # Detect retry patterns (very important for your scenario)
            retry_msgs = [e for e in events_sorted if "retry" in e.get("message", "").lower()]
            if retry_msgs:
                findings.append({
                    "trace_id": trace_id,
                    "issue": "retry_pattern",
                    "count": len(retry_msgs)
                })

        # Prepare output summary
        summary = {
            "total_traces": len(trace_map),
            "issues_found": findings
        }

        return Command(
            update={"output": f"Logs Analysis: {summary}"},
            goto="commander"
        )

    except Exception as e:
        return Command(
            update={"output": f"Logs Agent failed: {str(e)}"},
            goto="commander"
        )
##


# CI/CD Agent
async def cicd_agent(state: AgentState) -> Command(Literal['commander']):
    
    return Command(update="",goto="")

# Resolution Agent
async def resolver_agent(state: AgentState) -> Command(Literal['commander']): 
    
    return Command(update="",goto="")   


# Reporting Agent
async def reporter_agent(state: AgentState) -> Command(Literal['commander', '__end__']): 
    
    return Command(update="",goto="")




# Build graph
builder = StateGraph(AgentState)

builder.add_node("commander", commander_agent)
builder.add_node("metrics", metrics_agent)
builder.add_node("logs", logs_agent)
builder.add_node("cicd", cicd_agent)
builder.add_node("resolver", resolver_agent)
builder.add_node("reporter", reporter_agent)



build.add_edge(START, "commander")

graph = builder.compile()
