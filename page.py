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
from langchain_core.messages import HumanMessage, AIMessage
import pandas as pd


@st.cache_data
def get_db():
    return Database("data/processed/healthcare_dataset.csv")


db = get_db()

model = ChatOpenAI(name="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
model = model.with_structured_output(method="json_mode")

rails = Guardrails(fallback_to_llm=True, llm=model)

# if "result_table" not in st.session_state:
#     st.session_state.result_table = None


@tool
def db_tool(user_query: str):
    """Executes user's natural language query on healthcare database.
    Transforms the natural language query into a SQL query using a language model and executes it against the healthcare database.

    Args:
        user_query (str): The natural language query from the user.
    """
    db_agent = DBAgent(model=model, db=db)
    result, result_for_llm = db_agent.tool(user_query=user_query)
    # st.session_state.result_table = result
    # print("SESSION STATE RESULT TABLE", st.session_state.result_table)
    return result_for_llm


tools = [db_tool]
model_agent = ChatOpenAI(name="gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
agent = create_react_agent(
    model_agent,
    tools,
    prompt=chat_system_prompt.format(user_name="Matthew Smith"),
    debug=True,
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


if prompt := st.chat_input("Type your message here"):
    to_check_with_rails = st.session_state.messages + [{"role": "user", "content": prompt}]
    if rails.rail(to_check_with_rails):
        st.chat_message("user").markdown(prompt)

        st.session_state.messages.append({"role": "user", "content": prompt})

        response = agent.invoke(
            {"messages": convert_messages_to_langchain(st.session_state.messages)},
        )
        response = response["messages"][-1].content
        with st.chat_message("assistant"):
            st.markdown(response)
        # print("SESSION STATE RESULT TABLE2", st.session_state.result_table)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        st.info(
            "System is focused only on question answering for healthcare! Please try again with another message."
        )

# with st.sidebar:
#     st.write("### Result table")
#     if "result_table" in st.session_state:
#         result_table = st.session_state.result_table
#         st.write(result_table)
#         # if isinstance(result_table, str):
#         #     st.write(result_table)
#         # elif isinstance(result_table, list) and len(result_table) > 0:
#         #     df = pd.DataFrame(result_table)
#         #     st.dataframe(df)
#         # else:
#         #     st.write("No results found or the result is empty.")
#     else:
#         st.write("No results yet. Run a query to see the results here.")
