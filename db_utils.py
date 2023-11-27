from sqlalchemy import create_engine
import pandas as pd 
import psycopg2 
from pathlib import Path
import yaml


def load_db_credentials(): 
    conf = yaml.safe_load(Path('credentials.yaml').read_text())
    return conf

class RDSDatabaseConnector():
    def __init__(self, database_credentials_dict) -> None:
        self.database_credentials_dict = database_credentials_dict

    def initialise_SQLAlchemy(self):
        login = self.database_credentials_dict
        engine = create_engine(f"{login['DATABASE_TYPE']}+{login['DBAPI']}://{login['USER']}:{login['PASSWORD']}@{login['HOST']}:{login['PORT']}/{login['DATABASE']}")
        engine.execution_options(isolation_level='AUTOCOMMIT').connect()
        engine.connect() 
        return engine 
    
    def create_df_from_database(self, table):
        try:
            connect_to_database = self.initialise_sqlalchmey_engine()
            print("Connected to the database")
        except Exception as e:
            print(e, "failed to connect")

        query = f"SELECT * FROM {table}"
        df = pd.read_sql(query, connect_to_database)
        return df
    
    def df_to_csv(self, df):
        df.to_csv('loan_payments.csv', index=False)

        

