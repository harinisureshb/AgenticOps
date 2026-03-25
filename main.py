# app.py

from fastapi import FastAPI
from pydantic import BaseModel
# from graph import graph

app = FastAPI(title="AgenticOps API")

class RequestModel(BaseModel):
    time_stamp: str

@app.get("/")
def home():
    return {"message": "AgenticOps API is running"}

@app.post("/run-agent")
async def run_agent(request: RequestModel):
    result = await graph.ainvoke({"input": request.query})
    return {
        "query": request.query,
        "response": result["output"]
    }