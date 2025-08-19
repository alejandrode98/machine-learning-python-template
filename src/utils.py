from dotenv import load_dotenv
from sqlalchemy import create_engine
import os
import pandas as pd

load_dotenv()

def db_connect():
    engine = create_engine(os.getenv('DATABASE_URL'))
    engine.connect()
    return engine

def load_processed_data(file_path: str) -> pd.DataFrame:
    """
    Carga un dataset procesado (CSV) desde la carpeta data/processed.
    """
    return pd.read_csv(file_path)
