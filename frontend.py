# Step 1: Setup UI with Streamlit

import streamlit as st
import requests

st.set_page_config(page_title="LangGraph Agent UI", layout="centered")
st.title(" 🤖 AI Chatbot Agent")
st.write("Create and Interact with the AI Agent")

# User System Prompt
system_prompt = st.text_area(
    "🧠 System Prompt",
    height=70,
    placeholder="Define how your AI should behave..."
)

# Models
MODEL_NAME_GROQ = ["llama-3.3-70b-versatile", "mixtral-8x7b-32768"]
MODEL_NAME_OPENAI = ["gpt-4o-mini"]

# Provider Selection
provider = st.radio("🛰 Select Model Provider", ["Groq", "OpenAI"], horizontal=True)

# Dynamic Model Dropdown
if provider == "Groq":
    model_name = st.selectbox("🤖 Select Groq Model:", MODEL_NAME_GROQ)
elif provider == "OpenAI":
    model_name = st.selectbox("🤖 Select OpenAI Model:", MODEL_NAME_OPENAI)

# Web Search Option
allow_web_search = st.checkbox("🌐 Enable Web Search")

# User Query
user_query = st.text_area(
    "Enter your query:",
    height=200,
    placeholder="Ask anything...."
)

API_URL="http://127.0.0.1:9999/chat"

# Submit Button
if st.button("Ask Agent!"):
    if user_query.strip():

        # BACKEND API CALL (fastapi)
        payload = {
            "model_name": model_name,
            "model_provider": provider,
            "system_prompt": system_prompt,
            "messages": [user_query],
            "allow_search": allow_web_search
        }


    with st.spinner("⏳ Thinking..."):
        response = requests.post(API_URL,json=payload)
        if response.status_code==200:
            response_data = response.json()
            if "error" in response_data:
                st.error(response_data["error"])
            
            else:
                st.subheader("Agent Response")
                st.markdown(f"**Final Response:** {response_data['response']}")
