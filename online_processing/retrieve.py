import openai
from pymilvus import connections
from sentence_transformers import SentenceTransformer
from pymilvus import Collection
import streamlit as st

OPENAI_API_KEY = "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"
# Milvus Configuration
MILVUS_URI = "https://in03-ba90052e3130a37.serverless.gcp-us-west1.cloud.zilliz.com"
API_KEY = "e48d006636a4c89f8f8a3db1dc94a8e05eb42f66b4616baf2dfc3376e0b7d0fee5693cc0f894269a01902d7bbf63fcc9a87792d9"  # Replace with your actual Milvus API key
COLLECTION_NAME = "cv_embeddings"

# Connect to Milvus
connections.connect(alias="default", uri=MILVUS_URI, token=API_KEY)
# Load embedding model
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
client = openai.OpenAI(api_key=OPENAI_API_KEY)

def search_candidates(query_text, top_k=5):
    """
    Retrieve the most relevant candidate CV chunks based on a natural language query.
    """
    collection = Collection(COLLECTION_NAME)
    collection.load()

    query_embedding = embed_model.encode(query_text).tolist()

    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["text", "candidate_name"]
    )

    candidates = []
    for result in results[0]:
        candidates.append({
            "name": result.entity.get("candidate_name"),
            "text": result.entity.get("text"),
            "score": result.distance
        })
    return candidates


def generate_response(query, candidates):
    """
    Uses OpenAI GPT to generate a structured response based on retrieved candidates and chat history.
    """
    if not candidates:
        return "No matching candidates found. Try refining your query."

    # Format candidates for prompt
    candidate_texts = "\n\n".join([f"- {c['name']}: {c['text']}" for c in candidates])

    # Include previous chat history for context
    chat_context = "\n".join(
        [f"User: {entry['query']}\nBot: {entry['response']}" for entry in st.session_state.chat_history])

    prompt = f"""
    Previous conversation:
    {chat_context}

    Now, the user is asking: '{query}'

    Based on the previous context, refine your response using the following candidate data:
    {candidate_texts}

    Provide a structured and relevant answer while maintaining coherence with the past conversation.
    """



    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
             "content": "You are an AI assistant that recommends job candidates while maintaining context."},
            {"role": "user", "content": prompt}
        ]
    )

    return response.choices[0].message.content

