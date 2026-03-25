# graph.py — AgenticOps Multi-Agent LangGraph Orchestrator

import os
import logging
from typing import Annotated
from typing_extensions import TypedDict, Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.prebuilt import create_react_agent

from langchain_community.tools.tavily_search import TavilySearchResults

from tools import (
    analyze_cpu_metrics,
    analyze_memory_metrics,
    analyze_latency_metrics,
    analyze_error_rates,
    analyze_active_sessions,
    get_failed_application_logs,
    get_error_log_timeline,
    get_cicd_failures,
    get_deployment_timeline,
    search_resolution_faqs,
    send_email_via_power_platform,
)

# ──────────────── LOGGING ────────────────
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
logger = logging.getLogger("AgenticOps")

# ──────────────── ENV & LLM ────────────────
load_dotenv()

llm = init_chat_model(
    model="gpt-5.2",
    model_provider="openai",
)


# ──────────────── STATE ────────────────
class AgentState(TypedDict):
    """Shared state that flows between all agents in the graph."""
    issue: str                     # User's original issue description
    time_stamp: str                # Timestamp from the request body
    output: str                    # Final output sent back to user
    metrics_report: str            # Metrics agent's analysis
    logs_report: str               # Logs agent's analysis
    cicd_report: str               # CI/CD agent's analysis
    resolution_report: str         # Resolver agent's recommended fixes
    final_report: str              # Reporter agent's final summary
    agents_called: list[str]       # Tracks which agents have been called
    next_agent: str                # Commander's routing decision


# ──────────────── AGENT DEFINITIONS ────────────────

# ━━━━━━━━━━ 1. COMMANDER AGENT ━━━━━━━━━━
# The commander is a deterministic orchestrator that enforces a fixed pipeline:
#   metrics → logs → cicd → resolver → reporter → END
# The first three agents (metrics, logs, cicd) gather data and populate state.
# The resolver synthesizes findings into an actionable mitigation plan.
# The reporter produces the final HTML email summary.

AGENT_SEQUENCE = ["metrics", "logs", "cicd", "resolver", "reporter"]

async def commander_agent(state: AgentState) -> Command[Literal[
    "metrics", "logs", "cicd", "resolver", "reporter", "__end__"
]]:
    """
    The Commander routes the workflow through a fixed sequence:
    metrics → logs → cicd → resolver → reporter → END
    """
    called = state.get("agents_called", [])
    logger.info("Commander invoked | agents_called=%s", called)

    for agent in AGENT_SEQUENCE:
        if agent not in called:
            logger.info("Commander routing to: %s", agent)
            return Command(
                update={
                    "next_agent": agent,
                    "agents_called": called + [agent],
                },
                goto=agent,
            )

    logger.info("Commander: all agents completed, routing to __end__")
    return Command(update={"next_agent": "__end__"}, goto="__end__")


# ━━━━━━━━━━ 2. METRICS AGENT (React) ━━━━━━━━━━
metrics_tools = [
    analyze_cpu_metrics,
    analyze_memory_metrics,
    analyze_latency_metrics,
    analyze_error_rates,
    analyze_active_sessions,
]
metrics_agent_app = create_react_agent(
    model=llm,
    tools=metrics_tools,
    prompt=(
        "You are an expert SRE specializing in telemetry and metrics analysis. "
        "Your duty is to query system metrics (CPU, Memory, Latency, Errors) and detect anomalies, spikes, and precise time windows. "
        "Return a crisp, comprehensive incident report highlighting only the deviations and critical data points. "
        "You MUST invoke your tools to retrieve authoritative metric data before answering."
    )
)

async def metrics_agent(state: AgentState) -> Command[Literal["commander"]]:
    """Metrics Agent wrapper that calls the internal Agent."""
    logger.info("Metrics agent started")
    
    input_message = f"User Query: {state.get('issue', 'Analyze system health')}\nPlease fetch and analyze the latest telemetry metrics."
    
    # Run the autonomous loop
    response = await metrics_agent_app.ainvoke({"messages": [("user", input_message)]})
    
    # Extract only the final generated report
    final_output = response["messages"][-1].content
    logger.info("Metrics agent completed")
    
    return Command(
        update={"metrics_report": final_output},
        goto="commander",
    )


# ━━━━━━━━━━ 3. LOGS AGENT (React) ━━━━━━━━━━
logs_tools = [get_failed_application_logs, get_error_log_timeline]
logs_agent_app = create_react_agent(
    model=llm,
    tools=logs_tools,
    prompt=(
        "You are a DevOps engineer analyzing application logs. "
        "Fetch failed logs, pinpoint exact error patterns, categorize by severity/endpoint, "
        "and rigorously correlate these errors with the provided metrics report. "
        "Be concise, clear, and highlight exact stack traces or method failures."
    )
)

async def logs_agent(state: AgentState) -> Command[Literal["commander"]]:
    """Logs Agent wrapper that calls the internal Agent."""
    logger.info("Logs agent started")
    
    input_message = (
        f"User Query: {state.get('issue', 'Analyze application logs')}\n"
        f"Metrics Analysis (for correlation):\n{state.get('metrics_report', 'N/A')}\n\n"
        f"Please analyze the application logs using your tools."
    )
    
    response = await logs_agent_app.ainvoke({"messages": [("user", input_message)]})
    final_output = response["messages"][-1].content
    logger.info("Logs agent completed")
    
    return Command(
        update={"logs_report": final_output},
        goto="commander",
    )


# ━━━━━━━━━━ 4. CI/CD AGENT (React) ━━━━━━━━━━
cicd_tools = [get_cicd_failures, get_deployment_timeline]
cicd_agent_app = create_react_agent(
    model=llm,
    tools=cicd_tools,
    prompt=(
        "You are a CI/CD pipeline analyst. "
        "Investigate failed builds and timeline of deployments to check if bad shipped code caused the incidents mentioned in logs/metrics. "
        "Report the exact deployment ID or commit that triggered the issue. Be crisp and direct."
    )
)

async def cicd_agent(state: AgentState) -> Command[Literal["commander"]]:
    """CI/CD Agent wrapper that calls the internal Agent."""
    logger.info("CI/CD agent started")
    
    input_message = (
        f"User Query: {state.get('issue', 'Analyze CI/CD pipelines')}\n"
        f"Metrics Report:\n{state.get('metrics_report', 'N/A')}\n"
        f"Logs Report:\n{state.get('logs_report', 'N/A')}\n\n"
        f"Please analyze the CI/CD pipelines using your tools."
    )
    
    response = await cicd_agent_app.ainvoke({"messages": [("user", input_message)]})
    final_output = response["messages"][-1].content
    logger.info("CI/CD agent completed")
    
    return Command(
        update={"cicd_report": final_output},
        goto="commander",
    )


# ━━━━━━━━━━ 5. RESOLVER AGENT (React) ━━━━━━━━━━
tavily_tool = TavilySearchResults(max_results=8)
resolver_tools = [search_resolution_faqs, tavily_tool]
resolver_agent_app = create_react_agent(
    model=llm,
    tools=resolver_tools,
    prompt=(
        "You are an incident response specialist with access to an internal FAQ knowledge base and the Tavily web search tool. "
        "First, search the internal FAQs using keywords from the findings. Fetch the top 8 documents."
        "Then, use the Tavily web search tool to find additional context, best practices, or recent solutions from the web. "
        "Combine insights from BOTH the FAQ results and Tavily web search results to produce a comprehensive, actionable mitigation plan. "
        "Clearly label which recommendations come from the internal FAQs and which come from web research. "
        "Deliver a step-by-step mitigation plan that merges internal runbook guidance with industry best practices. "
        "You MUST invoke BOTH tools before finalizing your response."
    )
)

async def resolver_agent(state: AgentState) -> Command[Literal["commander"]]:
    """Resolver Agent wrapper that calls the internal Agent."""
    logger.info("Resolver agent started")
    
    input_message = (
        f"Investigation Summary:\n\n"
        f"METRICS:\n{state.get('metrics_report', 'N/A')}\n\n"
        f"LOGS:\n{state.get('logs_report', 'N/A')}\n\n"
        f"CI/CD:\n{state.get('cicd_report', 'N/A')}\n\n"
        f"Search the FAQs based on these findings and recommend specific resolution steps."
    )
    
    response = await resolver_agent_app.ainvoke({"messages": [("user", input_message)]})
    final_output = response["messages"][-1].content
    logger.info("Resolver agent completed")
    
    return Command(
        update={"resolution_report": final_output},
        goto="commander",
    )


# ━━━━━━━━━━ 6. REPORTER AGENT (React) ━━━━━━━━━━
reporter_agent_app = create_react_agent(
    model=llm,
    tools=[],
    prompt=(
        "You are a Lead SRE formatting a final incident report. "
        "Your output MUST be a well-structured HTML email body (using <html>, <body>, <h2>, <table>, <ul>, <p> tags). "
        "Use inline CSS for styling (professional colors, padding, borders). "
        "Required Sections: Executive Summary, Root Cause Analysis, Key Findings (Metrics, Logs, CI/CD), Actionable Resolution Steps, Impact. "
        "Keep it remarkably crisp, highly readable, and authoritative. "
        "Output ONLY the raw HTML — no markdown fences, no extra text."
    )
)

async def reporter_agent(state: AgentState) -> Command[Literal["commander", "__end__"]]:
    """Reporter Agent wrapper that calls the internal Agent."""
    logger.info("Reporter agent started")
    
    input_message = (
        f"Original Query: {state.get('issue', 'System health analysis')}\n"
        f"Timestamp: {state.get('time_stamp', '')}\n\n"
        f"=== METRICS REPORT ===\n{state.get('metrics_report', 'N/A')}\n\n"
        f"=== LOGS REPORT ===\n{state.get('logs_report', 'N/A')}\n\n"
        f"=== CI/CD REPORT ===\n{state.get('cicd_report', 'N/A')}\n\n"
        f"=== RESOLUTION REPORT ===\n{state.get('resolution_report', 'N/A')}"
    )
    
    response = await reporter_agent_app.ainvoke({"messages": [("user", input_message)]})
    final_output = response["messages"][-1].content
    
    logger.info("Reporter agent completed — final report generated")

    # Programmatically send the email via Power Platform
    timestamp = state.get("time_stamp", "")
    email_result = send_email_via_power_platform.invoke({"email_html": final_output, "timestamp": timestamp})
    logger.info("Email dispatch result: %s", email_result)
    
    return Command(
        update={
             "final_report": final_output,
             "output": final_output,
        },
        goto="__end__",
    )


# ──────────────── BUILD GRAPH ────────────────

builder = StateGraph(AgentState)

# Add all nodes
builder.add_node("commander", commander_agent)
builder.add_node("metrics", metrics_agent)
builder.add_node("logs", logs_agent)
builder.add_node("cicd", cicd_agent)
builder.add_node("resolver", resolver_agent)
builder.add_node("reporter", reporter_agent)

# Entry point: always start at commander
builder.add_edge(START, "commander")

# Compile the graph
graph = builder.compile()
logger.info("LangGraph compiled successfully")
