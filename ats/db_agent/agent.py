from ats.logger import get_logger
from ats.db_agent.prompts import data_context, nlq_to_sql_prompt
from langchain.schema import HumanMessage
from langchain.tools import tool
import pandas as pd

logger = get_logger(name="db_agent", log_file="db_agent.log")

class DBAgent:
    def __init__(self, model, db):
        self.model = model # should be with json mode enabled?
        self.db = db

    def tool(self, user_query: str):
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
        logger.info(f"Received user query: {user_query}")
        check_valid, message = self.check_nlq(user_query)
        logger.info(f"Query validation result: {check_valid}, message: {message}")
        if not check_valid:
            logger.error(f"Invalid query: {user_query}")
            logger.error(f"Reason: {message}")
            return {"error": message}
        
        sql_query = self.generate_sql_query(user_query)
        logger.info(f"Generated SQL query: {sql_query}")
        result = self.execute_sql_query(sql_query)
        logger.info(f"SQL query execution result: {result}")
        if "error" in result:
            logger.error(f"Error executing SQL query: {result['error']}")
            return {"error": result['error']}
        return result

    def check_nlq(self, user_query: str):
        """Check if the user query requires a read-only permission only 
        i.e. doesn't plan to change data in the database 
        and aligns with the database/table description.
        If not, return an error message.

        Args:
            user_query (str): The natural language query from the user.
        """

        prompt = """Check if this natural language query:
         - doesn't plan to change data in the database, i.e. doesn't try to insert or delete or update data
         - aligns with the database/table description.
        ---
        Data description:
        {context}
        ---
        Response format is a json with the following format:
        {{
            "is_valid": true/false, # True if the query is valid, False otherwise
            "message": "Your message explaining the reason why the query is valid or not."  # skip if is_valid is True
        }}
        ---
        User query:
        """
        prompt = prompt.format(context=data_context) + user_query
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
        logger.info(f"Generated SQL Query: {sql_query}")
        return sql_query
    
    def execute_sql_query(self, sql_query):
        """Execute the SQL query against the database.

        Args:
            sql_query (str): The SQL query to execute.

        Returns:
            pd.DataFrame: The result of the SQL query execution.
        """
        logger.info(f"Executing SQL Query: {sql_query}")
        result = self.db.query(sql_query)
        if isinstance(result, pd.DataFrame):
            logger.info(f"Query executed successfully. Result shape: {result.shape}")
            return result.to_json(orient="records")
        else:
            logger.error(f"Query execution failed. Result: {result}")
            return {"error": "Query execution failed."}
        
    def assess_text_fields_extraction(self, result, query):
        pass