# Step 1: Set up API Keys for Groq and Tavily

import os
from dotenv import load_dotenv

# Load variables from .env file
load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# Step 2: Set up LLM and Tools

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.prebuilt import create_react_agent



# Step 3: Setup AI Agent with search tool functionality

from langgraph.prebuilt import create_react_agent
from langchain_core.messages.ai import AIMessage
from langchain_core.messages import HumanMessage



def get_response_from_ai_agent(llm_id, query, allow_search, system_prompt, provider):
    
    # Set default system prompt if none provided
    if system_prompt is None:
        system_prompt = "Act as an AI chatbot who is smart and friendly"

    # Initialize LLM
    if provider=="Groq":
        llm = ChatGroq(model=llm_id, api_key=GROQ_API_KEY)

    elif provider=="OpenAI":
        llm = ChatOpenAI(model=llm_id, api_key=OPENAI_API_KEY)

    # Setup tools
    tools =[TavilySearchResults(max_results=2, api_key=TAVILY_API_KEY)] if allow_search else []
    
    # Create Agent
    agent = create_react_agent(
        model=llm,
        tools=tools,
        state_modifier=system_prompt
    )

    # Prepare input messages
    if isinstance(query, str):
        # Single query string
        messages = [HumanMessage(content=query)]
    elif isinstance(query, list):
        # Already formatted messages
        messages = query
    else:
        raise ValueError("Query must be string or list of messages")
    
    # Run agent
    state = {"messages": query}
    response = agent.invoke(state)

    # Extract final AI message
    messages=response.get("messages", [])
    ai_messages=[message.content for message in messages if isinstance(message, AIMessage)]
    if not ai_messages:
        return "No response generated"
    return ai_messages[-1] # Return final reply
