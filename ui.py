import os

import streamlit as st
from langgraph.prebuilt import create_react_agent

from ats.chat.prompts import chat_system_prompt
from ats.db_agent.agent import DBAgent
from ats.db_connector import Database
from ats.chat.utils import convert_langchain_messages_to_openai
from ats.chat.guardrails import Guardrails

from langchain_openai.chat_models import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolCall
import pandas as pd
import json

from ats.ui_utils import show_message, show_tool_message


st.title("Healthcare search agent")

# LOAD DATA
@st.cache_data
def get_db():
    return Database("data/processed/healthcare_dataset.csv")


db = get_db()

# SIDEBAR WITH KNOBS
with st.sidebar:
    st.write("## Parameters:")
    usernames = list(db.df.Doctor.unique())
    username = st.selectbox("Please, select your name:", usernames, index=None)

    db_agent_model_name = st.selectbox(
        "LLM", options=["smart", "& smarter", "& even smarter"]
    )

    table_truncation = st.slider(
        "Maximum size of output table",
        min_value=1,
        max_value=300,
        value=200,
        help="Too big tables won't fit into models context, but they can be truncated to the threshold",
    )

    double_check = st.checkbox(
        "Double check",
        help="Take more time, may reject some queries, but should be more precise.",
    )
    

model_name_map = {
    "smart": "gpt-4o",
    "& smarter": "gpt-4.1",
    "& even smarter": "o4-mini",
}


# SETUP MODEL FOR DB_TOOL AND RAILS

model = ChatOpenAI(
    name=model_name_map[db_agent_model_name], api_key=os.getenv("OPENAI_API_KEY")
).with_structured_output(method="json_mode")
rails = Guardrails(fallback_to_llm=True, llm=model)

# SETUP TOOLS FOR CHAT AGENT


@tool
def db_tool(user_query: str):
    """Executes user's natural language query on healthcare database.
    Transforms the natural language query into a SQL query using a language model and executes it against the healthcare database.

    Args:
        user_query (str): The natural language query from the user.
    """
    db_agent = DBAgent(
        model=model, db=db, double_check=double_check, table_truncation=table_truncation
    )
    result = db_agent.tool(user_query=user_query)
    return result


tools = [db_tool]

# SETUP CHAT AGENT

model_agent = ChatOpenAI(name="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
agent = create_react_agent(
    model_agent,
    tools,
    prompt=chat_system_prompt.format(user_name=username),
    debug=True,
)

# START OF THE PAGE

if username is not None:

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # show messages in the chat
    for message in st.session_state.messages:
        show_message(message)

    # main processing
    if prompt := st.chat_input("Type your message here"):
        prompt = HumanMessage(prompt)
        to_check_with_rails = st.session_state.messages + [prompt]

        if rails.rail(
            to_check_with_rails
        ):  # if check is passed print user's message and process
            st.session_state.messages.append(prompt)
            show_message(prompt)
            response = agent.invoke(
                {"messages": st.session_state.messages}
            )  # returns full convesation

            st.session_state.messages = response["messages"]

            if response["messages"][-2].name == "db_tool":
                show_tool_message(response["messages"][-2])  # show resulting table
            show_message(response["messages"][-1])  # show llm response

        else:
            st.info(
                "System is focused only on question answering for healthcare! Please try again with another message."
            )
