import os
import psycopg2
import numpy as np
from psycopg2.extras import execute_values
from sentence_transformers import SentenceTransformer
from datetime import datetime




POSTGRES_HOST = "localhost"
POSTGRES_PORT = 5432
POSTGRES_DB = "postgres"
POSTGRES_USER = "postgres"
POSTGRES_PASSWORD = "Rajneesh@2024"

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

def generate_embedding_locally(content):
    return embedding_model.encode(content).tolist()

def ingest_chat_data_to_postgres(chat_data, table_name="chat_history"):
    """
    Ingest chat data into PostgreSQL.
    
    Args:
        chat_data (list): List of chat records to ingest.
        table_name (str): Name of the table in PostgreSQL.
    """
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
        cursor = conn.cursor()

        # Create table if not exists
        cursor.execute(f"""
        CREATE TABLE IF NOT EXISTS {table_name} (
            id SERIAL PRIMARY KEY,
            query TEXT,
            response TEXT,
            feedback INT,
            feedback_text TEXT,
            query_embedding FLOAT8[],
            response_embedding FLOAT8[],
            timestamp TIMESTAMP
        );
        """)
        conn.commit()

        # Prepare data for insertion
        values = [
            (
                record["query"],
                record["response"],
                record["feedback"],
                record["feedback_text"],
                record["query_embedding"],
                record["response_embedding"],
                record["timestamp"]
            )
            for record in chat_data
        ]

        # Insert data into table
        execute_values(
            cursor,
            f"INSERT INTO {table_name} (query, response, feedback, feedback_text, query_embedding, response_embedding, timestamp) VALUES %s",
            values
        )
        conn.commit()

        cursor.close()
        conn.close()
        print("Chat data ingestion completed.")
    except Exception as e:
        print(f"Error ingesting chat data: {str(e)}")




def process_and_ingest_chat_data(chat_record):
    """
    Process and ingest a single chat record into PostgreSQL.

    Args:
        chat_record (dict): A dictionary containing query, response, feedback, and feedback_text.
                            Example:
                            {
                                "query": "user query",
                                "response": "llm generated response",
                                "feedback": 5,
                                "feedback_text": "Explain why Paris is the capital."
                            }
    """
    # Validate input
    required_keys = {"query", "response", "feedback", "feedback_text"}
    if not all(key in chat_record for key in required_keys):
        raise ValueError(f"Chat record must contain these keys: {required_keys}")

    # Generate embeddings
    query_embedding = generate_embedding_locally(chat_record["query"])
    response_embedding = generate_embedding_locally(chat_record["response"])

    # Add embeddings and timestamp to the record
    chat_record["query_embedding"] = query_embedding
    chat_record["response_embedding"] = response_embedding
    chat_record["timestamp"] = datetime.now()

    ingest_chat_data_to_postgres([chat_record])





def find_similar_chat(query, table_name="chat_history", similarity_threshold=0.5):
    """
    Find similar chat records based on the input query.

    Args:
        query (str): The input query to search for similar records.
        table_name (str): The table name in PostgreSQL.
        similarity_threshold (float): Cosine similarity threshold for matching.

    Returns:
        list: List of similar chat records.
    """
    try:
        conn = psycopg2.connect(
            host=POSTGRES_HOST,
            port=POSTGRES_PORT,
            database=POSTGRES_DB,
            user=POSTGRES_USER,
            password=POSTGRES_PASSWORD
        )
        cursor = conn.cursor()

        # Generate embedding for the query
        query_embedding = generate_embedding_locally(query)

        # Retrieve all chat records from the database
        cursor.execute(f"SELECT id, query, response, feedback, feedback_text, query_embedding FROM {table_name};")
        records = cursor.fetchall()

        similar_records = []
        for record in records:
            record_id, record_query, record_response, feedback, feedback_text, record_embedding = record
            if record_embedding is not None:
                # Compute cosine similarity
                similarity = np.dot(query_embedding, record_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(record_embedding)
                )
                if similarity >= similarity_threshold:
                    similar_records.append({
                        "id": record_id,
                        "query": record_query,
                        "response": record_response,
                        # "feedback": feedback,
                        # "feedback_text": feedback_text,
                        "similarity": similarity
                    })

        cursor.close()
        conn.close()
        return sorted(similar_records, key=lambda x: x["similarity"], reverse=True)
    except Exception as e:
        print(f"Error during similarity search: {str(e)}")
        return []



def save_response(chat_record):
    process_and_ingest_chat_data(chat_record)

def get_similar_chats(query):
    return find_similar_chat(query)


# Example Usage
if __name__ == "__main__":
    # Example chat record
    chat_record = {
        "query": "What is the capital of France?",
        "response": "The capital of France is Paris.",
        "feedback": 5,
        "feedback_text": "The response is correct"
    }

    # Ingest chat data
    process_and_ingest_chat_data(chat_record)

    # Find similar chat records
    similar_chats = find_similar_chat("What is the capital city of France?")
    print("Similar chats:", similar_chats)