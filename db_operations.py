# db_operations.py
import os
import uuid
import pandas as pd
from pymilvus import connections, Collection


def insert_data_to_milvus_only(input_data: pd.DataFrame):
    milvus_uri = os.getenv('MILVUS_URI')
    user = os.getenv('MILVUS_USER')
    password = os.getenv('MILVUS_PASS')
    connections.connect("default",
                        uri=milvus_uri,
                        user=user,
                        password=password,
                        secure=True)
    print(f"Connecting to DB: {milvus_uri}")

    # Check if the collection exists
    collection_name = "wb_convo"
    collection = Collection(name=collection_name)
    input_data['convo_id'] = [str(uuid.uuid4()) for x in range(len(input_data))]
    input_entities = [
        input_data['convo_id'].tolist(),
        input_data['category'].tolist(),
        input_data['content'].tolist(),
        input_data['source'].tolist(),
        input_data['token_count'].tolist(),
        input_data['openai_embedding'].tolist()
    ]
    ins_resp = collection.insert(input_entities)
    return ins_resp
