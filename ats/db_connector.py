from pandasql import sqldf
import pandas as pd
from typing import Union

# TODO: replace with sqllite3 
def load_df(path):
    df = pd.read_csv(path)
    df["Date_of_Admission"] = pd.to_datetime(df["Date_of_Admission"])
    df["Discharge_Date"] = pd.to_datetime(df["Discharge_Date"])
    return df

# simple wrapper, so it can be easily replaced with at least sqlite, but better with a normal DB
class Database:
    def __init__(self, df: Union[pd.DataFrame, str]):
        if isinstance(df, str):
            self.df = load_df(df)
        else:
            self.df = df
        
        # should be used outside of the class to get the table name
        self.table_name = "df"

    def query(self, query):
        df = self.df
        return sqldf(query, locals())
