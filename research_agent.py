from typing import TypedDict, List
import uuid
from datetime import datetime
from langgraph.graph import StateGraph, END
from tavily import TavilyClient
from groq import Groq
import chromadb
from dotenv import load_dotenv
import os

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Persistent (saves to disk)
chroma_client = chromadb.PersistentClient(path="./research_db")

collection = chroma_client.get_or_create_collection(name="research_memory")

class ResearchState(TypedDict):
    query: str
    raw_results: List[str]
    retrieved_chunks: List[str]
    final_answer: str

def search_node(state: ResearchState) -> ResearchState:
    print("[search] Searching the web...")
    response = tavily.search(query=state["query"], max_results=5)
    results = [r["content"] for r in response["results"]]
    print(f"[search] Got {len(results)} results")
    return {"raw_results": results}

def store_node(state: ResearchState) -> ResearchState:
    print("[store] Storing results in ChromaDB...")
    docs = state["raw_results"]
    ids = [str(uuid.uuid4()) for _ in docs]
    metadatas = [
        {   
            "query": state["query"],
            "doc_index": str(i),
            "stored_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        for i in range(len(docs))
    ]
    collection.add(documents=docs, ids=ids, metadatas=metadatas)
    print(f"[store] Stored {len(docs)} new documents (total: {collection.count()})")
    return {}

def retrieve_node(state: ResearchState) -> ResearchState:
    print("[retrieve] Retrieving relevant chunks...")
    results = collection.query(
        query_texts=[state["query"]],
        n_results=3,
        where={"query": state["query"]}
    )
    if not results["documents"][0]:
        print("[retrieve] No exact match, falling back to full collection...")
        results = collection.query(
            query_texts=[state["query"]],
            n_results=3
        )
    chunks = results["documents"][0]
    print(f"[retrieve] Retrieved {len(chunks)} chunks")
    return {"retrieved_chunks": chunks}

def synthesize_node(state: ResearchState) -> ResearchState:
    print("[synthesize] Generating final answer...")
    context = "\n\n".join(state["retrieved_chunks"])
    prompt = f"""You are a research assistant. Based on the context below, answer the query clearly and concisely.

Query: {state["query"]}

Context:
{context}

Provide a well-structured answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024
    )
    answer = response.choices[0].message.content.strip()
    return {"final_answer": answer}

graph = StateGraph(ResearchState)

graph.add_node("search_node", search_node)
graph.add_node("store_node", store_node)
graph.add_node("retrieve_node", retrieve_node)
graph.add_node("synthesize_node", synthesize_node)

graph.set_entry_point("search_node")

graph.add_edge("search_node", "store_node")
graph.add_edge("store_node", "retrieve_node")
graph.add_edge("retrieve_node", "synthesize_node")
graph.add_edge("synthesize_node", END)

app = graph.compile()

if __name__ == "__main__":
    query = input("Enter your research query: ")
    result = app.invoke({
        "query": query,
        "raw_results": [],
        "retrieved_chunks": [],
        "final_answer": ""
    })
    print(f"\n{'='*50}")
    print(f"FINAL ANSWER:\n{result['final_answer']}")
    print(f"{'='*50}")