# Step 1: Setup Pydantic model (Schema Validation)
from pydantic import BaseModel
from typing import List
from fastapi import FastAPI
from ai_agent import get_response_from_ai_agent

class RequestState(BaseModel):
    model_name: str
    model_provider: str
    system_prompt: str
    messages: List[str]
    allow_search: bool

# Allowed models
ALLOWED_MODEL_NAMES = [
    "llama-3.3-70b-versatile",
    "mixtral-8x7b-32768",
    "gpt-4o-mini"
]

app = FastAPI(title="LangGraph AI Agent")

# Step 2: API Endpoint
@app.post("/chat")
def chat_endpoint(request: RequestState):

    if request.model_name not in ALLOWED_MODEL_NAMES:
        return {"error": "Invalid model name. Kindly select a valid AI model."}

    llm_id = request.model_name
    query = request.messages
    allow_search = request.allow_search
    system_prompt = request.system_prompt
    provider = request.model_provider

    # Create AI agent dynamically and get response
    response = get_response_from_ai_agent(
        llm_id,
        query,
        allow_search,
        system_prompt,
        provider
    )

    return {"response": response}


# Step 3: Run app and Explore Swagger docs
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=9999)
