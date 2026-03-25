# graph.py — AgenticOps Multi-Agent LangGraph Orchestrator

import os
from typing import Annotated
from typing_extensions import TypedDict, Literal

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command
from langgraph.prebuilt import create_agent
from pydantic import BaseModel, Field

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
)

# ──────────────── ENV & LLM ────────────────
load_dotenv()

llm = init_chat_model(
    model="gpt-4o-mini",
    model_provider="openai",
)


# ──────────────── STATE ────────────────
class AgentState(TypedDict):
    """Shared state that flows between all agents in the graph."""
    input: str                     # User's original query
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
# The commander is the orchestrator. It decides which agent to call next
# based on which agents have already run.
# Flow: metrics → logs → cicd → resolver → reporter → END

class RouteDecision(BaseModel):
    next_agent: Literal["metrics", "logs", "cicd", "resolver", "reporter", "__end__"] = Field(
        description="The next specialized agent to route to. "
                    "metrics: To query system telemetry and metrics data if not already done. "
                    "logs: To query application logs for failures and patterns if not already done. "
                    "cicd: To check deployment pipelines for issues if not already done. "
                    "resolver: To search FAQs for resolution steps once issues are identified. "
                    "reporter: To generate the final incident report after investigation. "
                    "__end__: If the user request is fully satisfied or irrelevant."
    )
    reasoning: str = Field(description="Reasoning for the routing decision.")

async def commander_agent(state: AgentState) -> Command[Literal[
    "metrics", "logs", "cicd", "resolver", "reporter", "__end__"
]]:
    """
    The Commander routes the workflow intelligently based on the query and current state using an LLM.
    """
    called = state.get("agents_called", [])
    
    if "reporter" in called:
        return Command(update={"next_agent": "__end__"}, goto="__end__")
        
    system_prompt = (
        "You are the Commander Agent orchestrating specialized subagents.\n"
        f"User Query: {state.get('input', '')}\n"
        f"Agents already called in this flow: {called}\n"
        "Available agents to call:\n"
        "- metrics: analyses system health/telemetry anomalies\n"
        "- logs: analyses application errors and logs\n"
        "- cicd: analyses deployment pipelines and build states\n"
        "- resolver: searches knowledge base for fixes based on findings\n"
        "- reporter: compiles all findings into a final unified report\n"
        "Decide the absolute best next agent to invoke. You MUST NOT call an agent that has already been called. "
        "Standard sequence is metrics -> logs -> cicd -> resolver -> reporter, but skip steps if irrelevant."
    )
    
    router_llm = llm.with_structured_output(RouteDecision)
    decision = await router_llm.ainvoke([
        SystemMessage(content=system_prompt), 
        HumanMessage(content="Determine the next agent to call based on the current context.")
    ])
    
    next_agent = decision.next_agent
    
    if next_agent == "__end__":
        return Command(
            update={"next_agent": "__end__"},
            goto="__end__",
        )

    return Command(
        update={
            "next_agent": next_agent,
            "agents_called": called + [next_agent],
        },
        goto=next_agent,
    )


# ━━━━━━━━━━ 2. METRICS AGENT (React) ━━━━━━━━━━
metrics_tools = [
    analyze_cpu_metrics,
    analyze_memory_metrics,
    analyze_latency_metrics,
    analyze_error_rates,
    analyze_active_sessions,
]
metrics_agent_app = create_agent(
    model=llm,
    tools=metrics_tools,
    state_modifier=(
        "You are an expert SRE specializing in telemetry and metrics analysis. "
        "Your duty is to query system metrics (CPU, Memory, Latency, Errors) and detect anomalies, spikes, and precise time windows. "
        "Return a crisp, comprehensive incident report highlighting only the deviations and critical data points. "
        "You MUST invoke your tools to retrieve authoritative metric data before answering."
    )
)

async def metrics_agent(state: AgentState) -> Command[Literal["commander"]]:
    """Metrics Agent wrapper that calls the internal Agent."""
    
    input_message = f"User Query: {state.get('input', 'Analyze system health')}\nPlease fetch and analyze the latest telemetry metrics."
    
    # Run the autonomous loop
    response = await metrics_agent_app.ainvoke({"messages": [("user", input_message)]})
    
    # Extract only the final generated report
    final_output = response["messages"][-1].content
    
    return Command(
        update={"metrics_report": final_output},
        goto="commander",
    )


# ━━━━━━━━━━ 3. LOGS AGENT (React) ━━━━━━━━━━
logs_tools = [get_failed_application_logs, get_error_log_timeline]
logs_agent_app = create_agent(
    model=llm,
    tools=logs_tools,
    state_modifier=(
        "You are a DevOps engineer analyzing application logs. "
        "Fetch failed logs, pinpoint exact error patterns, categorize by severity/endpoint, "
        "and rigorously correlate these errors with the provided metrics report. "
        "Be concise, clear, and highlight exact stack traces or method failures."
    )
)

async def logs_agent(state: AgentState) -> Command[Literal["commander"]]:
    """Logs Agent wrapper that calls the internal Agent."""
    
    input_message = (
        f"User Query: {state.get('input', 'Analyze application logs')}\n"
        f"Metrics Analysis (for correlation):\n{state.get('metrics_report', 'N/A')}\n\n"
        f"Please analyze the application logs using your tools."
    )
    
    response = await logs_agent_app.ainvoke({"messages": [("user", input_message)]})
    final_output = response["messages"][-1].content
    
    return Command(
        update={"logs_report": final_output},
        goto="commander",
    )


# ━━━━━━━━━━ 4. CI/CD AGENT (React) ━━━━━━━━━━
cicd_tools = [get_cicd_failures, get_deployment_timeline]
cicd_agent_app = create_agent(
    model=llm,
    tools=cicd_tools,
    state_modifier=(
        "You are a CI/CD pipeline analyst. "
        "Investigate failed builds and timeline of deployments to check if bad shipped code caused the incidents mentioned in logs/metrics. "
        "Report the exact deployment ID or commit that triggered the issue. Be crisp and direct."
    )
)

async def cicd_agent(state: AgentState) -> Command[Literal["commander"]]:
    """CI/CD Agent wrapper that calls the internal Agent."""
    
    input_message = (
        f"User Query: {state.get('input', 'Analyze CI/CD pipelines')}\n"
        f"Metrics Report:\n{state.get('metrics_report', 'N/A')}\n"
        f"Logs Report:\n{state.get('logs_report', 'N/A')}\n\n"
        f"Please analyze the CI/CD pipelines using your tools."
    )
    
    response = await cicd_agent_app.ainvoke({"messages": [("user", input_message)]})
    final_output = response["messages"][-1].content
    
    return Command(
        update={"cicd_report": final_output},
        goto="commander",
    )


# ━━━━━━━━━━ 5. RESOLVER AGENT (React) ━━━━━━━━━━
resolver_tools = [search_resolution_faqs]
resolver_agent_app = create_agent(
    model=llm,
    tools=resolver_tools,
    state_modifier=(
        "You are an incident response specialist. "
        "Search the FAQs using given keywords from the findings to formulate actionable resolution steps. "
        "Deliver a clear, step-by-step mitigation plan based solely on the knowledge base. "
        "You MUST invoke your search tool to find relevant runbook entries."
    )
)

async def resolver_agent(state: AgentState) -> Command[Literal["commander"]]:
    """Resolver Agent wrapper that calls the internal Agent."""
    
    input_message = (
        f"Investigation Summary:\n\n"
        f"METRICS:\n{state.get('metrics_report', 'N/A')}\n\n"
        f"LOGS:\n{state.get('logs_report', 'N/A')}\n\n"
        f"CI/CD:\n{state.get('cicd_report', 'N/A')}\n\n"
        f"Search the FAQs based on these findings and recommend specific resolution steps."
    )
    
    response = await resolver_agent_app.ainvoke({"messages": [("user", input_message)]})
    final_output = response["messages"][-1].content
    
    return Command(
        update={"resolution_report": final_output},
        goto="commander",
    )


# ━━━━━━━━━━ 6. REPORTER AGENT (React) ━━━━━━━━━━
reporter_agent_app = create_agent(
    model=llm,
    tools=[],  # The reporter doesn't need external tools, it just synthesizes
    state_modifier=(
        "You are a Lead SRE formatting a final incident report. "
        "Present a professional, executive-ready breakdown. "
        "Required Sections: Executive Summary, Root Cause Analysis, Key Findings (Metrics, Logs, CI/CD), Actionable Resolution Steps, Impact. "
        "Keep it remarkably crisp, highly readable, and authoritative."
    )
)

async def reporter_agent(state: AgentState) -> Command[Literal["commander", "__end__"]]:
    """Reporter Agent wrapper that calls the internal Agent."""
    
    input_message = (
        f"Original Query: {state.get('input', 'System health analysis')}\n\n"
        f"=== METRICS REPORT ===\n{state.get('metrics_report', 'N/A')}\n\n"
        f"=== LOGS REPORT ===\n{state.get('logs_report', 'N/A')}\n\n"
        f"=== CI/CD REPORT ===\n{state.get('cicd_report', 'N/A')}\n\n"
        f"=== RESOLUTION REPORT ===\n{state.get('resolution_report', 'N/A')}"
    )
    
    response = await reporter_agent_app.ainvoke({"messages": [("user", input_message)]})
    final_output = response["messages"][-1].content
    
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
