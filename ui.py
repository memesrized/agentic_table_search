import os

import streamlit as st
from langgraph.prebuilt import create_react_agent

from ats.chat.prompts import chat_system_prompt
from ats.db_agent.agent import DBAgent
from ats.db_connector import Database
from ats.chat.guardrails import Guardrails

from langchain_openai.chat_models import ChatOpenAI
from langchain.tools import tool
from langchain_core.messages import HumanMessage

from ats.ui_utils import show_message, show_tool_message, model_name_map

# PARAMS:
DATA_PATH = os.getenv("DATA_PATH", "data/processed/healthcare_dataset.csv")
LLM_RETRIES = os.getenv("LLM_RETRIES", 3)
API_KEY = os.getenv("OPENAI_API_KEY")
CHAT_MODEL_NAME = os.getenv("CHAT_MODEL_NAME", "gpt-4o")

st.title("Healthcare search agent")


# LOAD DATA
@st.cache_data
def get_db():
    return Database(DATA_PATH)


db = get_db()

# SIDEBAR WITH KNOBS
with st.sidebar:
    st.write("## Parameters:")
    usernames = list(db.df.Doctor.unique())
    username = st.selectbox("Please, select your name:", usernames, index=None)

    # just to give an illusion of control to user :) (spoiler: they are mostly the same)
    db_agent_model_name = st.selectbox(
        "LLM", options=["smart", "& smarter", "& even smarter"]
    )

    # it would be nice to have a database to store temp tables for users instead of feeding it to llm
    # so they could download it as a file or check on ui if it's too big for llm
    # but I spent to much time on agents debug, so it's TODO for the next sprint
    # also I tried to put output tables into session_state, but that didn't work for some reason, so I gave up
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


# SETUP MODEL FOR DB_TOOL AND RAILS

model = ChatOpenAI(
    name=model_name_map[db_agent_model_name],
    api_key=API_KEY,
    max_retries=LLM_RETRIES,
    # yes, it's better to use pydantic models, but it's overkill for poc
    # especially when you need to experiment a lot, it add additional unnecessary complexety to handle
).with_structured_output(method="json_mode")
rails = Guardrails(fallback_to_llm=True, llm=model)

# SETUP TOOLS FOR CHAT AGENT


# I didn't manage to make this decorator work with class method, so here is this stupid workaround
@tool
def db_tool(user_query: str):
    """Executes user's natural language query on healthcare database.
    Transforms the natural language query into a SQL query using a language model and executes it against the healthcare database.

    Args:
        user_query (str): The natural language query from the user.

    Return:
        Results from database or string with error
    """
    db_agent = DBAgent(
        model=model, db=db, double_check=double_check, table_truncation=table_truncation
    )
    try:
        result = db_agent.tool(user_query=user_query)
    except Exception as e:
        return f"Tool failed: {str(e)}"
    return result


tools = [db_tool]

# SETUP CHAT AGENT

model_agent = ChatOpenAI(name=CHAT_MODEL_NAME, api_key=API_KEY, max_retries=LLM_RETRIES)
agent = create_react_agent(
    model_agent,
    tools,
    prompt=chat_system_prompt.format(user_name=username),
    debug=os.getenv("LOG_LEVEL", "").lower() == "debug",
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

        try:
            rails_check = rails.rail(to_check_with_rails)
        except Exception:
            raise
            st.info("Sorry, something went wrong, please try again later.")

        if rails_check:  # if check is passed print user's message and process
            st.session_state.messages.append(prompt)
            show_message(prompt)
            try:
                # returns full convesation
                response = agent.invoke({"messages": st.session_state.messages})

                st.session_state.messages = response["messages"]

                # show tool message to increase transparency
                # so users could detect hallucinations
                if response["messages"][-2].name == "db_tool":
                    show_tool_message(response["messages"][-2])  # show resulting table
                show_message(response["messages"][-1])  # show llm response
            except Exception:
                st.info("Sorry, something went wrong, please try again later.")
                raise

        else:
            st.info(
                "System is focused only on question answering for healthcare! Please try again with another message."
            )
