from llama_index.core import VectorStoreIndex
from llama_index.readers.wikipedia import WikipediaReader
from llama_index.core.tools import FunctionTool

# Try importing WikipediaToolSpec
try:
    from llama_index.tools.wikipedia import WikipediaToolSpec
    from llama_index.core.tools.tool_spec.load_and_search import LoadAndSearchToolSpec
    WIKI_TOOLS_AVAILABLE = True
except ImportError:
    print("Warning: WikipediaToolSpec not found. Falling back to WikipediaReader with VectorStoreIndex.")
    WIKI_TOOLS_AVAILABLE = False

# Initialize tools
tools = []
if WIKI_TOOLS_AVAILABLE:
    try:
        wiki_spec = WikipediaToolSpec()
        for tool in wiki_spec.to_tool_list():
            wrapped_tools = LoadAndSearchToolSpec.from_defaults(tool).to_tool_list()
            tools.extend(wrapped_tools)
    except Exception as e:
        print(f"Warning: Failed to initialize WikipediaToolSpec: {str(e)}. Falling back to WikipediaReader.")
        WIKI_TOOLS_AVAILABLE = False

if not WIKI_TOOLS_AVAILABLE:
    try:
        wiki_reader = WikipediaReader()
        documents = wiki_reader.load_data(pages=["Sri Lanka"])
        index = VectorStoreIndex.from_documents(documents)

        def rag_search(query: str) -> str:
            """Search indexed Sri Lanka Wikipedia page for relevant information."""
            try:
                query_engine = index.as_query_engine(similarity_top_k=2)
                response = query_engine.query(query)
                return str(response)
            except Exception as e:
                return f"Error retrieving information: {str(e)}"

        tools.append(FunctionTool.from_defaults(fn=rag_search))
    except Exception as e:
        raise Exception(f"Error setting up WikipediaReader: {str(e)}")