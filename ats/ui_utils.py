import json

import pandas as pd
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage


def show_tool_message(message):
    tool_result = json.loads(message.content)
    if not tool_result.get("error"):
        with st.chat_message("tool", avatar="📊"):
            st.write("Raw database results:")
            with st.popover("Click to expand data", use_container_width=True):
                tool_res = json.loads(tool_result["result"])
                st.dataframe(pd.DataFrame(tool_res))


def show_message(message):
    if message.name == "db_tool":
        show_tool_message(message)
    elif isinstance(message, HumanMessage):
        st.chat_message("user").markdown(message.content)
    elif isinstance(message, AIMessage):
        st.chat_message("assistant").markdown(message.content)
