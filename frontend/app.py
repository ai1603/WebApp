import os
import sys
from pathlib import Path

# Add the src directory to the Python path
current_dir = Path(__file__).parent.absolute()
project_root = current_dir.parent
sys.path.append(str(project_root / "src"))

import streamlit as st
from document_processor import DocumentProcessor
from rag_chain import RagChain

#Initialize the session state
if 'processor' not in st.session_state:
    st.session_state.processor = DocumentProcessor(vector_store="vectors")
if 'rag' not in st.session_state:
    st.session_state.rag = RagChain()
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'url_processed' not in st.session_state:
    st.session_state.url_processed = False
if 'current_url' not in st.session_state:
    st.session_state.current_url = ""

st.title("Chat with Website")
st.markdown("Enter website URL and ask questions about it")

#URL input 
st.subheader("Website URL")

url = st.text_input("Enter website URL: ", placeholder="www.example.com")

if url:
    if not url.startswith(("https://", "https://")):
        st.error("Please enter a URL starting with https:// or http:// ")
    else:
        if url != st.session_state.current_url:
            st.session_state.url_processed = False
            st.session_state.current_url = url
            
        if not st.session_state.url_processed:
            with st.spinner("Processing website content.........."):
                
                try:
                    vector_store = st.session_state.processor.process_url(url=url)
                    if vector_store:
                        st.session_state.vector_store = vector_store
                        st.session_state.rag.set_vector_store(vector_store)
                        st.session_state.url_processed = True
                        st.success(f"Succesfully processed: {url}")
                    else:
                        st.error("Failed to process the website and please check")
                except Exception as e:
                    st.error(f"Error processing website : {str(e)}")
        else:
            st.success(f"Website is ready {url}")

#Question Section
if st.session_state.url_processed and st.session_state.vector_store:
    st.subheader("Ask question")
    
    #Initialize the chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
        
    #display the chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    #Accept user input
    if prompt := st.chat_input("Ask question about the website"):
        #Add user message to the chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
            
        #Get bot response 
        with st.chat_message("assistant"):
            with st.spinner("Thinking"):
                try:
                    response = st.session_state.rag.ask_question(prompt)
                    st.markdown(response)  
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({"role": "assistant", "content": response})   
                except Exception as e:
                    error_msg = f"Sorry, I encountered an error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({"role": "assistant", "content": error_msg})
    else:
        st.info("👆 Please enter a website URL above to start chatting!")

# Sidebar with information
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown("""
    This app allows you to:
    1. **Load** any website content
    2. **Ask questions** about the content
    3. **Get AI-powered answers** based on the website
    
    **How it works:**
    - The app extracts text from the website
    - Splits it into manageable chunks
    - Creates embeddings for semantic search
    - Uses AI to answer your questions
    """)
    if st.session_state.url_processed:
        st.success("🟢 Ready to answer questions!")
        if st.button("🔄 Process New URL"):
            st.session_state.url_processed = False
            st.session_state.current_url = ""
            st.session_state.messages = []
            st.rerun()
            
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    
    
            