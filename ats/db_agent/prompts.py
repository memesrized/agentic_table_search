import json

nlq_to_sql_prompt = """You are a SQL query generator. 

Given a natural language question, generate a SQL query that retrieves the requested data from the healthcare dataset.

Write ONLY read-only SQL queries. Do not write any data manipulation or modification queries.
---
Context

Data:
{data_context}

Database and SQL:
{sql_context}
---
Response is a json with the following format:
{{
    "query": your_sql_query
}}
---
User query:
"""

data_context = """There is one and only one table named 'df' that you can use. Table length is 54966 rows.
It contains healthcare records with all necessary information described below, so you can derive insights from it.

Single record in this healthcare dataset represents one patient's complete hospital admission episode, not one patient.
It means that if a patient was admitted multiple times, there will be multiple records for that patient.

Columns in the table, use exact column names in exact case in the query:         
{
    "Patient_ID": "Unique identifier for the patient.",
    "Name": "This column represents the name of the patient associated with the healthcare record.",
    "Year_of_Birth": "Year of birth of the patient.",
    "Age": "The age of the patient at the time of admission, expressed in years.",
    "Gender": "Indicates the gender of the patient, either \"Male\" or \"Female.\"",
    "Blood_Type": "The patient's blood type, which can be one of the common blood types (e.g., \"A+\", \"O-\", etc.).",
    "Medical_Condition": "Specifies the primary medical condition or diagnosis associated with the patient, such as \"Diabetes,\" \"Hypertension,\" \"Asthma,\" and more.",
    "Date_of_Admission": "The date on which the patient was admitted to the healthcare facility.",
    "Doctor": "The name of the doctor responsible for the patient's care during their admission.",
    "Hospital": "Identifies the healthcare facility or hospital where the patient was admitted.",
    "Insurance_Provider": "Indicates the patient's insurance provider, which can be one of several options, including \"Aetna,\" \"Blue Cross,\" \"Cigna,\" \"UnitedHealthcare,\" and \"Medicare.\"",
    "Billing_Amount": "The amount of money billed for the patient's healthcare services during their admission. This is expressed as a floating-point number.",
    "Room_Number": "The room number where the patient was accommodated during their admission.",
    "Admission_Type": "Specifies the type of admission, which can be \"Emergency,\" \"Elective,\" or \"Urgent,\" reflecting the circumstances of the admission.",
    "Discharge_Date": "The date on which the patient was discharged from the healthcare facility, based on the admission date and a random number of days within a realistic range.",
    "Medication": "Identifies a medication prescribed or administered to the patient during their admission. Examples include \"Aspirin,\" \"Ibuprofen,\" \"Penicillin,\" \"Paracetamol,\" and \"Lipitor.\"",
    "Test_Results": "Describes the results of a medical test conducted during the patient's admission. Possible values include \"Normal,\" \"Abnormal,\" or \"Inconclusive,\" indicating the outcome of the test."
}

Data types: {
    "Patient_ID": "int64",
    "Name": "string",
    "Year of Birth": "int64"
    "Age": "int64",
    "Gender": "string",
    "Blood_Type": "string",
    "Medical_Condition": "string",
    "Date_of_Admission": "datetime64[ns]",
    "Doctor": "string",
    "Hospital": "string",
    "Insurance_Provider": "string",
    "Billing_Amount": "float64",
    "Room_Number": "int64",
    "Admission_Type": "string",
    "Discharge_Date": "datetime64[ns]",
    "Medication": "string",
    "Test_Results": "string"
}

VERY IMPORTANT NOTES:
- columns ["Name", "Doctor", "Hospital"] are not in lowercase, but you need to perform search in lowercase to avoid case sensitivity issues (but the result should be in original case).
- "Hospital" column contains 'broken' values in some way (e.g. "Moreno Murphy, Griffith and", here user may ask for "Moreno Murphy and Griffith" and expect to get this value, so in this case you can use two LIKE filters with AND).

Examples:
You got request from user to find patients from "Moreno Murphy and Griffith" hospital.
In reality there is only "Moreno Murphy, Griffith and" hospital.
To find it you need to create condition as "WHERE LOWER(Hospital) LIKE '%moreno%' AND LOWER(Hospital) LIKE '%murphy%' AND LOWER(Hospital) LIKE '%griffith%'"
So you need to look separately (but with "AND" logic) for each meaningful word in the hospital name provided by user.

You need to find John Doe as a patient (similar logic for doctor).
You should use "LOWER(Name) = "john doe" in where condidtion for this.

"""

sql_context = """
- Use SQLite syntax to query the table, since your query will be executed with pandasql library.
- Instead of e.g. "COUNT(*)" (or with other aggregations) as column name, you must use appropriate name like "something_count" or "something_number", etc.
- Use RANK() window function instead of LIMIT 1 to include all records that tie for the top value, cause sometimes there can be 
"""

nlq_to_sql_prompt = nlq_to_sql_prompt.format(
    data_context=data_context,
    sql_context=sql_context
)

nlq_check_prompt = """Check if this natural language query:
    - doesn't plan to change data in the database, i.e. doesn't try to insert or delete or update data in the table/database
    - temporary tables for calculation and analysis are allowed
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
