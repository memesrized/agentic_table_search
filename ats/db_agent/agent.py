from ats.logger import get_logger
from ats.db_agent.prompts import data_context, nlq_to_sql_prompt, nlq_check_prompt
from langchain.schema import HumanMessage
import pandas as pd
from typing import Union

logger = get_logger(name="db_agent")


class DBAgent:
    def __init__(self, model, db):
        self.model = model  # should be with json mode enabled?
        self.db = db
        self.temp_table = None

    def tool(self, user_query: str) -> tuple[list, dict[str, Union[str, list[dict]]]]:
        """
        Executes user's natural language query on healthcare database.
        Transforms the natural language query into a SQL query using a language model and executes it against the healthcare database.

        Args:
            user_query (str): The natural language query from the user.
        """

        # Pipeline:
        # - Checks
        #     - Check the query
        #         - if it a read-only request
        #         - if it aligns with the database/table description
        #     - Return an error with a message if the query is not valid
        # - Generate SQL query
        # - Execute the SQL query against the database
        # - Check if text fields are extracted correctly as needed in nlq
        #     - Regenerate SQL and rerun if necessary 3 times
        #     - If still not correct, return an error with a message and results
        # - Return the result of the SQL query execution

        # TODO: move params to the class init / config
        truncation_limit = 200  # Limit for truncation of the result

        meta = {}
        logger.info(f"Received user query: {user_query}")
        # to prevent hallucinations when LLM is confinetly trying to query data that doesn't exist
        check_valid, message = self.check_nlq(user_query)
        logger.info(f"Query validation result: {check_valid}, message: {message}")
        if not check_valid:
            logger.error(f"Invalid query: {user_query}")
            logger.error(f"Reason: {message}")
            return {"error": message, "result": "[]"}

        logger.info(f"Generating SQL query")
        sql_query = self.generate_sql_query(user_query)

        logger.info(f"Executing SQL Query: {sql_query}")
        result = self.execute_sql_query(sql_query)
        logger.info(f"SQL query execution result: {result}")
        if isinstance(result, str):
            logger.error(f"Error executing SQL query: {result}")
            return {"error": result, "result": "[]"}
        
        # workaround
        # st.session_state doesn't work and doesn't allow to save a table and show to the user without LLM
        # to overcome this separate db is needed to store temporary table 
        if len(result) > truncation_limit:
            meta["truncated"] = "Original length is {}, truncated to {}.".format(
                len(result), truncation_limit
            )
            return result, {
                "result": result.head(truncation_limit).to_json(orient="records"),
                **meta,
            }

        return {"result": result.to_json(orient="records"), **meta}

    def check_nlq(self, user_query: str):
        """Check if the user query requires a read-only permission only
        i.e. doesn't plan to change data in the database
        and aligns with the database/table description.
        If not, return an error message.

        Args:
            user_query (str): The natural language query from the user.
        """

        # TODO: move to a db with RBAC and remove read-only check
        prompt = nlq_check_prompt.format(context=data_context) + user_query
        response = self.model.invoke([HumanMessage(content=prompt)])
        if response["is_valid"]:
            return True, "Query is valid."
        return False, response["message"]

    def generate_sql_query(self, user_query):
        """Generate SQL query from the user query using the model.

        Args:
            user_query (str): The natural language query from the user.

        Returns:
            str: The generated SQL query.
        """
        prompt_ = nlq_to_sql_prompt + user_query
        response = self.model.invoke([HumanMessage(content=prompt_)])
        sql_query = response["query"]
        return sql_query

    def execute_sql_query(self, sql_query):
        """Execute the SQL query against the database.

        Args:
            sql_query (str): The SQL query to execute.

        Returns:
            pd.DataFrame: The result of the SQL query execution.
        """
        result = self.db.query(sql_query)
        if isinstance(result, pd.DataFrame):
            logger.info(f"Query executed successfully. Result shape: {result.shape}")
            return result
        else:
            logger.error(f"Query execution failed. Result: {result}")
            return "Query execution failed."

    def assess_text_fields_extraction(self, result, query):
        pass

