import os
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage

def query_sop_database(query: str) -> str:
    """Queries Pinecone for flight safety protocols and grounding knowledge."""
    pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    index_name = os.environ.get("PINECONE_INDEX_NAME", "drone-sentinel-sops")
    vector_store = PineconeVectorStore(index_name=index_name, embedding=embeddings)
    docs = vector_store.similarity_search(query, k=2)
    
    if not docs:
        return "No specific security or operational SOP found for this scenario."
        
    context = "\n".join([d.page_content for d in docs])
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    
    rag_prompt = f"Context from SOPs:\n{context}\n\nQuery: {query}\n\nProvide an operational response:"
    return llm.invoke([HumanMessage(content=rag_prompt)]).content
