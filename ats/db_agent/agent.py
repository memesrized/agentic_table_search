import json
from typing import Union

import pandas as pd
from langchain.schema import HumanMessage

from ats.db_agent.prompts import (
    data_context,
    nlq_check_prompt,
    nlq_to_sql_prompt,
    prompt_simple_check_sql,
    sql_context,
    prompt_regenerate_sql
)
from ats.logger import get_logger

logger = get_logger(name="db_agent")


class DBAgent:
    def __init__(self, model, db, double_check=False, table_truncation=200):
        self.model = model
        self.db = db
        self.temp_table = None
        self.double_check = double_check
        self.truncation_limit = table_truncation

    def tool(self, user_query: str) -> dict[str, Union[str, list[dict]]]:
        """
        Executes user's natural language query on healthcare database.
        Transforms the natural language query into a SQL query using a language model and executes it against the healthcare database.

        Pipeline:
        - Checks
            - Check the query
                - if it a read-only request
                - if it aligns with the database/table description
            - Return an error with a message if the query is not valid
        - Generate SQL query
        - Optionally double check the query
        - Execute the SQL query against the database
        - Return the result of the SQL query execution

        Args:
            user_query (str): The natural language query from the user.
        """
        meta = {"user_query": user_query}
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
        logger.info("Generated sql query: {}".format(sql_query))

        # It's dumb AF sometimes, for this reason it's behind a switch
        # "It filters patients based on the doctor's name, which is incorrect.
        # The query should filter based on the doctor's name to get patients associated with that doctor." © gpt-4.1
        if self.double_check:
            for i in range(3):  # TODO: move to params
                logger.debug("SQL check #{}".format(i))
                sql_check = self.simple_check_sql(user_query, sql_query)
                logger.debug("SQL check results: {}".format(json.dumps(sql_check)))
                if sql_check["is_correct"]:
                    break
                prompt = prompt_regenerate_sql.format(user_query=user_query, sql_query=sql_query, review=sql_check)
                sql_query = self.generate_sql_query(prompt)
                logger.info("Generated sql query: {}".format(sql_query))
            else:
                sql_check = self.simple_check_sql(user_query, sql_query)
                logger.debug("SQL check results: {}".format(json.dumps(sql_check)))
                if not sql_check["is_correct"]:
                    return {"error": "Can't create correct sql query", "result": "[]"}
            

        logger.info(f"Executing SQL Query: {sql_query}")
        result = self.execute_sql_query(sql_query)
        logger.info(f"SQL query execution result: {result}")
        if isinstance(result, str):
            logger.error(f"Error executing SQL query: {result}")
            return {"error": result, "result": "[]"}

        # workaround
        # st.session_state doesn't work and doesn't allow to save a table and show to the user without LLM
        # to overcome this separate db is needed to store temporary table
        if len(result) > self.truncation_limit:
            meta["truncated"] = "Original length is {}, truncated to {}.".format(
                len(result), self.truncation_limit
            )
            return result, {
                "result": result.head(self.truncation_limit).to_json(orient="records"),
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

    def generate_sql_query(self, user_query: str) -> str:
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

    def execute_sql_query(self, sql_query: str):
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

    # this just doesn't work in general
    # it should be a different approach
    def simple_check_sql(self, query, sql):
        prompt = prompt_simple_check_sql.format(
            sql=sql, query=query, data_context=data_context, sql_context=sql_context
        )
        res = self.model.invoke([HumanMessage(prompt)])
        return res
