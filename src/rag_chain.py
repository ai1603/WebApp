from langchain import hub
from langchain.chat_models import init_chat_model
from langgraph.graph import START, StateGraph
# from langchain_community.vectorstores import FAISS
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from typing_extensions import TypedDict, List
from langchain_core import documents
from dotenv import load_dotenv

load_dotenv()

class State(TypedDict):
    question: str
    context: List[documents.Document]
    answer: str
    
class RagChain:
    
    def __init__(self, vector_store: FAISS = None):
        self.llm = init_chat_model(model="gpt-4", model_provider="openai")
        self.prompt = hub.pull("rlm/rag-prompt")
        self.vector_store = vector_store
        self.graph = None
        
        if vector_store:
            self._build_graph()
    
    def set_vector_store(self, vector_store: FAISS):
        """Set the vector store and rebuild the graph"""
        self.vector_store = vector_store
        self._build_graph()
        
    def _build_graph(self):
        """Build the RAG graph"""
        graph_builder = StateGraph(State)
        graph_builder.add_sequence([self._retrieve, self._generate])
        graph_builder.add_edge(START, "_retrieve")
        self.graph = graph_builder.compile()
        
    def _retrieve(self, state: State) -> dict:
         if not self.vector_store:
             raise ValueError("Vector store not set")
         retrieve_docs = self.vector_store.similarity_search(state["question"])
         return {"context": retrieve_docs}
    
    def _generate(self, state: State) -> dict:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
        messages = self.prompt.invoke({
            "question": state["question"], 
            "context": docs_content
        })
        response = self.llm.invoke(messages)
        return {"answer": response.content}
    
    def ask_question(self, question: str) -> str:
        if not self.graph:
            raise ValueError("Graph not built. Set vector first")
        
        response = self.graph.invoke({"question": question})
        return response["answer"]