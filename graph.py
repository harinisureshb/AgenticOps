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
async def logs_agent(state: AgentState) -> Command(Literal['commander']):
    
    return Command(update="",goto="")

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