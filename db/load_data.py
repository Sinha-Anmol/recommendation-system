# db/load_data.py

import pandas as pd
from sqlalchemy import create_engine

# Load CSV data
df = pd.read_csv('./data/user_behavior.csv')

# Connect to the database
engine = create_engine('mysql+pymysql://root:@localhost/recommendation_db')

# Load data into database
df.to_sql('user_behavior', con=engine, if_exists='replace', index=False)
print("Data loaded successfully")
