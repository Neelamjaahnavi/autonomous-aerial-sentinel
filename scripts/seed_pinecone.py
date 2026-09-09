import os
import time
from pinecone import Pinecone, ServerlessSpec
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_core.documents import Document

api_key = os.environ.get("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY environment variable is not set.")

pc = Pinecone(api_key=api_key)
index_name = os.environ.get("PINECONE_INDEX_NAME", "drone-sentinel-sops")

existing_indexes = [index.name for index in pc.list_indexes()]

if index_name not in existing_indexes:
    print(f"Creating Pinecone index '{index_name}'...")
    pc.create_index(
        name=index_name,
        dimension=1536,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
    while not pc.describe_index(index_name).status['ready']:
        time.sleep(1)

sample_sops = [
    Document(
        page_content="SOP-101: Perimeter Breach Protocol. If an unauthorized vehicle or individual is detected within 50 meters of the primary perimeter fence, immediately log GPS coordinates, raise threat level to MEDIUM, and trigger automated visual tracking.",
        metadata={"category": "security", "sop_id": "101"}
    ),
    Document(
        page_content="SOP-102: Flight Altitude & Weather Limits. Drones must not exceed 400 feet AGL (Above Ground Level). In the event of wind gusts exceeding 25 knots or heavy precipitation, initiate emergency Return-To-Home (RTH) sequence.",
        metadata={"category": "flight_safety", "sop_id": "102"}
    ),
    Document(
        page_content="SOP-103: Thermal Anomaly Response. If visual sensors report thermal readings above 85°C on industrial infrastructure or transformers, flag immediately as a critical hazard and alert emergency dispatch.",
        metadata={"category": "hazard", "sop_id": "103"}
    )
]

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
print("Uploading sample SOP documents to Pinecone vector store...")

vector_store = PineconeVectorStore.from_documents(
    documents=sample_sops,
    embedding=embeddings,
    index_name=index_name
)

print("Pinecone vector store successfully populated!")
