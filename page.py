import os

import streamlit as st
from langgraph.prebuilt import create_react_agent

from ats.chat.prompts import chat_system_prompt
from ats.db_agent.agent import DBAgent
from ats.db_connector import Database

from langchain_openai.chat_models import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage



@st.cache_data
def get_db():
    return Database("data/processed/healthcare_dataset.csv")
db = get_db()

model = ChatOpenAI(name="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
model = model.with_structured_output(method="json_mode")

@tool
def db_tool(user_query: str):
    """Executes user's natural language query on healthcare database.
        Transforms the natural language query into a SQL query using a language model and executes it against the healthcare database.

        Args:
            user_query (str): The natural language query from the user.
    """
    db_agent = DBAgent(model=model, db=db)
    return db_agent.tool(user_query=user_query)


tools = [db_tool]
model_agent = ChatOpenAI(name="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
agent = create_react_agent(
    model_agent, tools, prompt=chat_system_prompt.format(user_name="Matthew Smith")
)

st.title("Healthcare search agent")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def convert_messages_to_langchain(messages):
    """Convert Streamlit chat messages to LangChain format."""
    langchain_messages = []
    for message in messages:
        if message["role"] == "user":
            langchain_messages.append(HumanMessage(content=message["content"]))
        elif message["role"] == "assistant":
            langchain_messages.append(AIMessage(content=message["content"]))
    return langchain_messages

# React to user input
if prompt := st.chat_input("What is up?"):
    # Display user message in chat message container
    st.chat_message("user").markdown(prompt)
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    response = agent.invoke({"messages": convert_messages_to_langchain(st.session_state.messages)})
    response = response["messages"][-1].content
    # Display assistant response in chat message container
    with st.chat_message("assistant"):
        st.markdown(response)
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
