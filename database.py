from sqlalchemy import create_engine
import pandas as pd
import numpy as np
import os

def convert_back_to_ndarray(row):
    # Remove newline characters and extra spaces
    array_str_clean = row.replace('\n', ' ').replace('  ', ' ')

    # Convert to numpy array
    # Strip the brackets and split the string into a list of numbers
    array_str_clean = array_str_clean.strip('[]')
    array = np.fromstring(array_str_clean, sep=' ')
    return array

def query_db():
    """Return wells based on criteria."""
    
    conf = {
    'host': os.environ.get('DB_HOST'),
    'port': os.environ.get('DB_PORT'),
    'database': os.environ.get('DB_NAME'),
    'user': os.environ.get('DB_USER'),
    'password': os.environ.get('DB_PASSWORD')
    }

    connection_string = f"mysql+pymysql://{conf['user']}:{conf['password']}@{conf['host']}:{conf['port']}/{conf['database']}"

    # Creating the engine for MS SQL Server
    engine = create_engine(connection_string)

    query = "SELECT * FROM offer_cleaned"
    df_db = pd.read_sql(query, engine)
    df_db['training_str_vector'] = df_db['training_str_vector'].apply(lambda row: convert_back_to_ndarray(row))
    #print(df_db)

    return df_db
