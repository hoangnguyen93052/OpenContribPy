import os
import pandas as pd
from sqlalchemy import create_engine
from datetime import datetime
import requests
import json
import logging

logging.basicConfig(level=logging.INFO)

class ETLPipeline:
    def __init__(self, db_uri, source_url, target_table):
        self.db_uri = db_uri
        self.source_url = source_url
        self.target_table = target_table
        self.engine = create_engine(self.db_uri)
    
    def extract(self):
        logging.info("Starting data extraction...")
        try:
            response = requests.get(self.source_url)
            response.raise_for_status()
            data = response.json()
            df = pd.json_normalize(data)
            logging.info("Data extraction completed successfully.")
            return df
        except Exception as e:
            logging.error(f"Error during extraction: {e}")
            raise

    def transform(self, df):
        logging.info("Starting data transformation...")
        try:
            # Example transformation: Filtering and renaming columns
            transformed_df = df[['id', 'name', 'value']]
            transformed_df.columns = ['ID', 'Name', 'Value']
            transformed_df['Timestamp'] = datetime.now()
            logging.info("Data transformation completed successfully.")
            return transformed_df
        except Exception as e:
            logging.error(f"Error during transformation: {e}")
            raise

    def load(self, df):
        logging.info("Starting data loading...")
        try:
            with self.engine.connect() as connection:
                df.to_sql(self.target_table, con=connection, if_exists='replace', index=False)
            logging.info("Data loading completed successfully.")
        except Exception as e:
            logging.error(f"Error during loading: {e}")
            raise

    def run(self):
        try:
            raw_data = self.extract()
            transformed_data = self.transform(raw_data)
            self.load(transformed_data)
            logging.info("ETL process completed successfully.")
        except Exception as e:
            logging.error(f"ETL process failed: {e}")

if __name__ == "__main__":
    DB_URI = os.getenv("DATABASE_URI", "sqlite:///etl_db.sqlite")
    SOURCE_URL = os.getenv("SOURCE_URL", "https://api.example.com/data")
    TARGET_TABLE = "processed_data"

    etl_pipeline = ETLPipeline(DB_URI, SOURCE_URL, TARGET_TABLE)
    etl_pipeline.run()