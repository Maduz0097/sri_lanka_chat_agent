from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core import VectorStoreIndex
from typing import List


def wikipedia_search(query: str) -> str:
    """Search Wikipedia for the given query."""
    try:
        reader = WikipediaReader()
        documents = reader.load_data(pages=[query], auto_suggest=True)
        if not documents:
            return "No relevant Wikipedia content found."
        return documents[0].text[:2000]  # Limit to 2000 chars
    except Exception as e:
        return f"Error accessing Wikipedia: {str(e)}"


def rag_search(query: str) -> str:
    """Perform RAG search using pre-indexed Sri Lanka Wikipedia content."""
    try:
        # Load Wikipedia page for Sri Lanka
        reader = WikipediaReader()
        documents = reader.load_data(pages=["Sri Lanka"])
        if not documents:
            return "No Sri Lanka Wikipedia content available."

        # Create vector store index
        index = VectorStoreIndex.from_documents(documents)

        # Query the index
        query_engine = index.as_query_engine(similarity_top_k=2)
        response = query_engine.query(query)
        return str(response)
    except Exception as e:
        return f"Error in RAG search: {str(e)}"