import os

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
import pandas as pd
import numpy as np
from dotenv import load_dotenv

load_dotenv()


def truncate_strings(df, max_lengths):
    """
    Truncate strings in DataFrame columns according to specified max lengths

    Args:
        df: pandas DataFrame
        max_lengths: dict of column names and their maximum lengths
    """
    df_copy = df.copy()
    for column, max_length in max_lengths.items():
        if column in df_copy.columns and df_copy[column].dtype == object:
            df_copy[column] = df_copy[column].astype(str).apply(lambda x: x[:max_length])
    return df_copy


def connect_to_milvus(uri, token):
    """Connect to Milvus instance"""
    try:
        connections.connect(
            alias="default",
            uri=uri,
            token=token
        )
        print("Successfully connected to Milvus")
    except Exception as e:
        print(f"Failed to connect to Milvus: {e}")
        raise


def create_buffett_collection():
    """Create a collection for Warren Buffett's quotes and embeddings"""

    # Define the dimension of your OpenAI embeddings
    dim = 1536  # OpenAI's ada-002 model uses 1536 dimensions

    # Define collection schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8000),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=500),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]

    schema = CollectionSchema(fields=fields, description="Warren Buffett Quotes Collection")
    collection_name = "buffett_quotes"

    # Create collection if it doesn't exist
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)

    collection = Collection(name=collection_name, schema=schema)

    # Create IVF_FLAT index for vector field
    index_params = {
        "metric_type": "L2",
        "index_type": "IVF_FLAT",
        "params": {"nlist": 1024}
    }
    collection.create_index(field_name="embedding", index_params=index_params)
    return collection


def insert_data_to_milvus(df, collection):
    """Insert data from DataFrame to Milvus collection"""

    # Define maximum lengths for each field
    max_lengths = {
        'category': 500,
        'content': 5000,
        'source': 500
    }

    try:
        # Truncate strings to meet Milvus limits
        df_truncated = truncate_strings(df, max_lengths)

        # Prepare data for insertion
        entities = [
            df_truncated['category'].tolist(),
            df_truncated['content'].tolist(),
            df_truncated['source'].tolist(),
            df_truncated['openai_embedding'].tolist()
        ]

        # Insert data
        collection.insert(entities)
        collection.load()

        print(f"Successfully inserted {len(df)} records into Milvus")

    except Exception as e:
        print(f"Error inserting data: {e}")
        raise


def process_and_insert_dataframes(pkl_file_path, milvus_uri, milvus_token):
    """Process and insert all dataframes from the pickle file"""

    try:
        # Read the pickle file
        df = pd.read_pickle(pkl_file_path)

        # Connect to Milvus
        connect_to_milvus(milvus_uri, milvus_token)

        # Create collection
        collection = create_buffett_collection()

        # Process and insert data
        insert_data_to_milvus(df, collection)

        print("Data insertion completed successfully")

    except Exception as e:
        print(f"Error processing data: {e}")
        raise
    finally:
        # Close connection
        connections.disconnect("default")


def main():
    # Your Milvus credentials
    MILVUS_URI = os.getenv("MILVUS_URI")
    MILVUS_TOKEN = os.getenv("MILVUS_TOKEN")

    # Read the pickle file
    pkl_file_path = '../wb_train_data_full_transcript.pkl'
    df = pd.read_pickle(pkl_file_path)

    # Connect to Milvus
    process_and_insert_dataframes(pkl_file_path, MILVUS_URI, MILVUS_TOKEN)


if __name__ == "__main__":
    main()
