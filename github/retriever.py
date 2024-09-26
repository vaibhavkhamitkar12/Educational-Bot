from langchain_core.documents import Document
import os
from pinecone import Pinecone
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

# Initialize the vector store
# Set up API Keys
os.environ["PINECONE_API_KEY"] = "534349d3-964c-4282-a45c-6a5a23ec8764"
os.environ['GROQ_API_KEY'] = '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'

# Initialize Pinecone and the HuggingFace embedding model
api_key = os.environ.get("PINECONE_API_KEY")
pc = Pinecone(api_key=api_key, environment="us-east-1-aws")
vectorstore_from_docs = pc.Index("sarvamaisumarry")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

class Retriever:
    def __init__(self, vectorstore, embedding_model):
        self.vecstore = vectorstore
        self.embedding_model = embedding_model

    def sim_search(self, query):
        query_vector = self.embedding_model.encode(query)
        response = self.vecstore.query(
            vector=query_vector,
            top_k=5,
            include_metadata=True
        )
        # Convert the retrieved matches into Document instances
        documents = [
            Document(page_content=match['metadata']['text'], metadata={"source": match['metadata']})
            for match in response['matches']
        ]
        return documents

retriever = Retriever(vectorstore_from_docs, embedding_model)
