import json
from typing import Union

import pandas as pd
from langchain.schema import HumanMessage
from retry import retry

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
        
        logger.info(f"DBAgent initialized with double_check={double_check}, truncation_limit={table_truncation}")
        logger.debug(f"Model type: {type(model).__name__}, DB type: {type(db).__name__}")

    @retry(tries=2)
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
        logger.info(f"Processing user query: '{user_query}'")
        
        # to prevent hallucinations when LLM is confidently trying to query data that doesn't exist
        logger.debug("Starting query validation")
        check_valid, message = self.check_nlq(user_query)
        
        if not check_valid:
            logger.warning(f"Query validation failed for: '{user_query}' - Reason: {message}")
            return {"error": message, "result": "[]"}
        
        logger.info("Query validation passed")

        logger.info("Starting SQL query generation")
        sql_query = self.generate_sql_query(user_query)
        logger.info(f"Generated SQL query: {sql_query}")

        # It's dumb AF sometimes, for this reason it's behind a switch
        # "It filters patients based on the doctor's name, which is incorrect.
        # The query should filter based on the doctor's name to get patients associated with that doctor." © gpt-4.1
        # 
        # this should be a separate function
        if self.double_check:
            logger.info("Double-check enabled, validating generated SQL")
            for i in range(3):  # TODO: move to params
                logger.debug(f"SQL validation attempt #{i + 1}")
                sql_check = self.simple_check_sql(user_query, sql_query)
                logger.debug(f"SQL validation results: {json.dumps(sql_check, indent=2)}")
                
                if sql_check["is_correct"]:
                    logger.info(f"SQL validation passed on attempt #{i + 1}")
                    break
                    
                logger.warning(f"SQL validation failed on attempt #{i + 1}, regenerating query")
                prompt = prompt_regenerate_sql.format(user_query=user_query, sql_query=sql_query, review=sql_check)
                sql_query = self.generate_sql_query(prompt)
                logger.info(f"Regenerated SQL query: {sql_query}")
            else:
                logger.warning("Maximum SQL validation attempts reached, performing final check")
                sql_check = self.simple_check_sql(user_query, sql_query)
                logger.debug(f"Final SQL validation results: {json.dumps(sql_check, indent=2)}")
                if not sql_check["is_correct"]:
                    logger.error("Failed to generate correct SQL query after all attempts")
                    return {"error": "Can't create correct sql query", "result": "[]"}
            

        logger.info(f"Executing SQL query: {sql_query}")
        result = self.execute_sql_query(sql_query)
        
        if isinstance(result, str):
            logger.error(f"SQL execution failed: {result}")
            return {"error": result, "result": "[]"}

        # workaround
        # st.session_state doesn't work and doesn't allow to save a table and show to the user without LLM
        # to overcome this separate db is needed to store temporary table
        if len(result) > self.truncation_limit:
            logger.info(f"Result truncated: original length {len(result)}, truncated to {self.truncation_limit}")
            meta["truncated"] = f"Original length is {len(result)}, truncated to {self.truncation_limit}."
            return result, {
                "result": result.head(self.truncation_limit).to_json(orient="records"),
                **meta,
            }

        logger.info(f"Query completed successfully, returning {len(result)} rows")
        return {"result": result.to_json(orient="records"), **meta}

    @retry(tries=2)
    def check_nlq(self, user_query: str):
        """Check if the user query requires a read-only permission only
        i.e. doesn't plan to change data in the database
        and aligns with the database/table description.
        If not, return an error message.

        Args:
            user_query (str): The natural language query from the user.
        """
        logger.debug(f"Validating natural language query: '{user_query}'")
        
        # TODO: move to a db with RBAC and remove read-only check
        prompt = nlq_check_prompt.format(context=data_context) + user_query
        logger.debug(f"Sending validation prompt to model (length: {len(prompt)} chars)")
        
        try:
            response = self.model.invoke([HumanMessage(content=prompt)])
            logger.debug(f"Model validation response: {json.dumps(response, indent=2)}")
            
            if response["is_valid"]:
                logger.debug("Query validation successful")
                return True, "Query is valid."
            else:
                logger.debug(f"Query validation failed: {response['message']}")
                return False, response["message"]
        except Exception as e:
            logger.error(f"Error during query validation: {str(e)}")
            return False, f"Validation error: {str(e)}"

    @retry(tries=2)
    def generate_sql_query(self, user_query: str) -> str:
        """Generate SQL query from the user query using the model.

        Args:
            user_query (str): The natural language query from the user.

        Returns:
            str: The generated SQL query.
        """
        logger.debug(f"Generating SQL for query: '{user_query}'")
        prompt_ = nlq_to_sql_prompt + user_query
        logger.debug(f"Sending SQL generation prompt to model (length: {len(prompt_)} chars)")
        
        try:
            response = self.model.invoke([HumanMessage(content=prompt_)])
            sql_query = response["query"]
            logger.debug(f"Model generated SQL: {sql_query}")
            return sql_query
        except Exception as e:
            logger.error(f"Error generating SQL query: {str(e)}")
            raise

    @retry(tries=2)
    def execute_sql_query(self, sql_query: str):
        """Execute the SQL query against the database.

        Args:
            sql_query (str): The SQL query to execute.

        Returns:
            pd.DataFrame: The result of the SQL query execution.
        """
        logger.debug(f"Executing SQL query: {sql_query}")
        
        try:
            result = self.db.query(sql_query)
            if isinstance(result, pd.DataFrame):
                logger.info(f"SQL query executed successfully. Result shape: {result.shape}")
                logger.debug(f"Result columns: {list(result.columns)}")
                return result
            else:
                logger.error(f"SQL query execution returned non-DataFrame result: {type(result)} - {result}")
                return "Query execution failed."
        except Exception as e:
            logger.error(f"Exception during SQL query execution: {str(e)}")
            return f"Query execution failed: {str(e)}"

    # this just doesn't work in general
    # it should be a different approach
    @retry(tries=2)
    def simple_check_sql(self, query, sql):
        """Validate the generated SQL query against the original natural language query.
        
        Args:
            query (str): The original natural language query
            sql (str): The generated SQL query to validate
            
        Returns:
            dict: Validation result from the model
        """
        logger.debug(f"Validating SQL query: {sql}")
        logger.debug(f"Against natural language query: {query}")
        
        prompt = prompt_simple_check_sql.format(
            sql=sql, query=query, data_context=data_context, sql_context=sql_context
        )
        logger.debug(f"Sending SQL validation prompt to model (length: {len(prompt)} chars)")
        
        try:
            res = self.model.invoke([HumanMessage(prompt)])
            logger.debug(f"SQL validation model response: {json.dumps(res, indent=2)}")
            return res
        except Exception as e:
            logger.error(f"Error during SQL validation: {str(e)}")
            return {"is_correct": False, "message": f"Validation error: {str(e)}"}
