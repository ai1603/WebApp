import os
# from langchain.document_loaders import WebBaseLoader
from langchain_community.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
# from langchain.vectorstores import FAISS
from langchain_community.vectorstores import FAISS
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core import documents
from typing_extensions import List, TypedDict
from bs4 import SoupStrainer

load_dotenv()

llm = init_chat_model(model="gpt-4o-mini", model_provider="openai", add_start_index=True)

class DocumentProcessor:
    
    def __init__(self, vector_store: str):
        self.api_key = os.getenv("OPENAI_API_KEY")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large", api_key=self.api_key)
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

    def load_website(self, url:str) -> List[documents.Document]:
        # Load without class restrictions to get all content
        loader = WebBaseLoader(web_paths=(url,))
        loader.requests_kwargs = {'verify':False}
        
        try:
            docs = loader.load()
            print(f"Successfully loaded {len(docs)} documents")
            if docs:
                print(f"First document preview: {docs[0].page_content[:200]}... ")
                return docs
            else:
                print("No content found in the URL")
                return []
        except Exception as e:
            print(f"Error loading documents: {e}")
            return []
    
    def split_documents(self, documents: List[documents.Document]) -> List[documents.Document]:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200, add_start_index=True)
        split_docs = self.text_splitter.split_documents(documents)
        print(f"number of split docs : {len(split_docs)}")
        return split_docs    

    def create_vector_store(self, split_documents: List[documents.Document]) -> FAISS:
        """Create and save vector store from documents"""
        vector_store = FAISS.from_documents( split_documents, embedding=self.embeddings )
        vector_store.save_local("vectors")
        return vector_store
    
    def load_vector_store(self) -> FAISS:
        """Load existing vector store"""
        try:
            vector_store = FAISS.load_local("vectors", self.embeddings, allow_dangerous_deserialization=True)
            return vector_store
        except Exception as e:
            print(f"Error loading vector store: {e}")
            return None
        
    def process_url(self, url: str) -> FAISS | None:
        """Complete pipeline to process URL and create vector store"""
        try:
            # Load documents
            docs = self.load_website(url)
            if not docs:
                print("No documents were loaded from the URL")
                return None
            
            # Split documents
            split_docs = self.split_documents(documents=docs)
            if not split_docs:
                print("No document chunks were created")
                return None
            
            # Create vector store
            vector_store = self.create_vector_store(split_documents=split_docs)
            if not vector_store:
                print("Failed to create vector store")
                return None
                
            return vector_store
            
        except Exception as e:
            print(f"Error processing URL: {e}")
            return None
        