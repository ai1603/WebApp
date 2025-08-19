from document_processor import DocumentProcessor
from rag_chain import RagChain

def main():
    """Main CLI application"""
    
    processor = DocumentProcessor(vector_store="vectors")
    rag = RagChain()
    
    url = input("Enter URL: ")
    
    print("Processing URL...")
    vector_store = processor.process_url(url)
    
    if not vector_store:
        print("Failed to process URL")
        return
    
    rag.set_vector_store(vector_store)
    print("Ready to answer questions!")
    
    while True:
        question = input("\nEnter question (or 'exit'/'quit' to stop): ")
        if question.lower() in ['exit', 'quit']:
            break
        if question.strip():
            try:
                answer = rag.ask_question(question)
                print(f"\nAnswer: {answer}")
            except Exception as e:
                print(f"Error: {e}")


if __name__ == "__main__":
    main()