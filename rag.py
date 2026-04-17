import os
from langchain_community.vectorstores import FAISS
# Note: Since sentence_transformers is installed directly, we can use HuggingFaceEmbeddings
# If langchain-huggingface isn't available, we can use langchain_community.embeddings
try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import MarkdownTextSplitter

class PedagogyRAG:
    """RAG system for pedagogical best practices."""
    
    def __init__(self, doc_path="pedagogy_guidelines.md"):
        self.doc_path = doc_path
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2", 
            model_kwargs={'device': 'cpu'}
        )
        self.retriever = self._build_index()

    def _build_index(self):
        """Builds a FAISS vector store index from the markdown document."""
        if not os.path.exists(self.doc_path):
            print(f"Warning: {self.doc_path} not found. RAG will return empty context.")
            return None
        
        with open(self.doc_path, "r", encoding="utf-8") as f:
            text = f.read()
            
        # Split document into chunks
        splitter = MarkdownTextSplitter(chunk_size=400, chunk_overlap=50)
        docs = splitter.create_documents([text])
        
        # Create Vector Store
        vectorstore = FAISS.from_documents(docs, self.embeddings)
        # Return retriever fetching top 2 chunks
        return vectorstore.as_retriever(search_kwargs={"k": 2})
        
    def retrieve_guidelines(self, query: str) -> str:
        """Retrieves formatted text context for a given query."""
        if not self.retriever:
            return "Pedagogy guidelines are unavailable."
        
        docs = self.retriever.invoke(query)
        context = "\n\n---\n\n".join([doc.page_content for doc in docs])
        return context
